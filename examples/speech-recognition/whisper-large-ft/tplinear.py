import poptorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from poptorch_experimental_addons import collectives as collectives
from typing import List, Optional, Tuple, Union
from optimum.graphcore.quantization.group_quantize import group_quantize_compress, group_quantize_decompress
import numpy as np
# from remap import tensor_remap

def remap(x):
    # workaround
    x = torch.stack([x,x])
    return x[0]

class GroupQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, w_packed, w_scale, w_bias, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_packed = nn.Parameter(w_packed, requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)
        self.w_bias = nn.Parameter(w_bias, requires_grad=False)
        self.bias = bias  # Bias is uncompressed


    @classmethod
    def from_model(cls, linear: nn.Linear, num_groups: int):
        w = linear.weight.data
        bias = linear.bias
        w_packed, w_scale, w_bias = group_quantize_compress(w, num_groups)
        return cls(linear.in_features, linear.out_features, w_packed, w_scale, w_bias, bias)

    def forward(self, input):
        weight = group_quantize_decompress(self.w_packed.data, self.w_scale.data, self.w_bias.data, dtype=input.dtype)
        weight = poptorch.ipu_print_tensor(weight, 'weight', print_gradient=False)
        return F.linear(input, weight, self.bias)

class TPGroupQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, w_packed, w_scale, w_bias, bias=None, tp=4, axis=0, allgather_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_packed = nn.Parameter(w_packed, requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)
        self.w_bias = nn.Parameter(w_bias, requires_grad=False)
        self.bias = bias  # Bias is uncompressed
        self.tp = tp
        self.axis = axis
        self.allgather_output = allgather_output
        self.serialization = 1
        self.serialization_mode = poptorch.MatMulSerializationMode.OutputChannels

    @classmethod
    def from_model(cls, linear: nn.Linear, num_groups: int, tp: int, axis: int, allgather_output:bool = False):
        w = linear.weight.data
        bias = linear.bias
        print("---> Quantizing weights of shape", w.shape)
        w_packed, w_scale, w_bias = group_quantize_compress(w, num_groups)
        return cls(linear.in_features, linear.out_features, w_packed, w_scale, w_bias, bias, tp, axis, allgather_output)

    def parallelize(self):
        with torch.no_grad():
            axis = self.axis
            w_packed = self.w_packed.data
            w_scale = self.w_scale.data
            w_bias = self.w_bias.data

            shape = w_packed.shape
            (num_row, num_group, group_size) = shape
            assert shape[axis] % self.tp == 0
            print("W shape before", self.w_packed.shape)
            if axis == 0:
                self.w_packed.data = w_packed.reshape(self.tp, num_row // self.tp, num_group, group_size).contiguous()
                self.w_scale.data = w_scale.reshape(self.tp, num_row // self.tp, num_group, 1).contiguous()
                self.w_bias.data = w_bias.reshape(self.tp, num_row // self.tp, num_group, 1).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.reshape(self.tp, -1)

            elif axis == 1:
                self.w_packed.data = w_packed.reshape(num_row, self.tp, num_group//self.tp, group_size).permute(1, 0, 2, 3).contiguous()
                self.w_scale.data = w_scale.reshape(num_row, self.tp, num_group//self.tp, 1).permute(1, 0, 2, 3).contiguous()
                self.w_bias.data = w_bias.reshape(num_row, self.tp, num_group//self.tp, 1).permute(1, 0, 2, 3).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.unsqueeze(0).repeat(self.tp, 1)
                    self.out_bias = nn.Parameter(self.bias.data, requires_grad=self.bias.requires_grad)
                    self.bias = None
            print("W shape after", self.w_packed.shape)
        return self

    def serialized_mm(self, weight, x):
        output = poptorch.serializedMatMul(x, weight.t(), self.serialization_mode,  self.serialization)
        if self.bias is not None:
            output += self.bias
        return output


    def forward(self, input):
        weight = group_quantize_decompress(self.w_packed.data, self.w_scale.data, self.w_bias.data, dtype=input.dtype)
        print("decompressed weights: ", weight.shape)

        # weight = poptorch.ipu_print_tensor(weight, 'weight', print_gradient=False)
        if self.serialization > 1:
            y = self.serialized_mm(weight, input)
        else:
            y = F.linear(input, weight, self.bias)

        if self.axis == 1:
            y =  collectives.all_reduce_cross_replica_sum(y, self.tp)
        if hasattr(self, "out_bias"):
            y = y + self.out_bias

        if self.allgather_output and self.axis == 0:
                y = collectives.all_gather_cross_replica(y, self.tp)
                y = y.permute(0,3,1,2)
                y = y.reshape([y.shape[-4]* y.shape[-3],y.shape[-2], y.shape[-1]])
                y = y.permute(1,2,0)
        return y

class TPLinear(nn.Linear):
    def __init__(self, layer: nn.Linear, axis=0, tp=4, dp=1, allgather_output=False, serialization=1):
        shape = layer.weight.data.shape
        super().__init__(shape[1], shape[0], dtype=layer.weight.data.dtype, bias=layer.bias is not None)
        self.allgather_output=allgather_output
        self.axis = axis
        self.tp = tp # tensor parallel replicas
        self.dp = dp # data parallel replicas
        self.serialization = serialization
        for param_src, param_dst in zip(layer.parameters(), self.parameters()):
            param_dst.requires_grad = param_src.requires_grad
        self.serialization_mode = poptorch.MatMulSerializationMode.OutputChannels
        self.remap = False

    def parallelize(self):
        with torch.no_grad():
            axis = self.axis
            weights = self.weight.data
            shape = weights.shape
            assert shape[axis] % self.tp == 0
            print("W shape before", self.weight.shape)
            if axis == 0:
                self.weight.data = weights.reshape(self.tp, shape[0] // self.tp, shape[1]).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.reshape(self.tp, -1)

            elif axis == 1:
                self.weight.data = weights.reshape(shape[0], self.tp, shape[1] // self.tp).permute(1, 0, 2).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.unsqueeze(0).repeat(self.tp, 1)
                    self.out_bias = nn.Parameter(self.bias.data, requires_grad=self.bias.requires_grad)
                    self.bias = None

            print("W shape after", self.weight.shape)
            return self

    def serialized_version(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.serialization_mode,  self.serialization)
        if self.bias is not None:
            output += self.bias
        return output

    def forward(self, x):
        if self.remap:
            x = remap(x)
        if self.serialization >1:
            y = self.serialized_version(x)
        else:
            y = super().forward(x)

        if self.axis == 1:
            y =  collectives.all_reduce_cross_replica_sum(y, self.tp)
        if hasattr(self, "out_bias"):
            y = y + self.out_bias

        if self.allgather_output and self.axis == 0:
            y = collectives.all_gather_cross_replica(y, self.tp)
            y = y.permute(0,3,1,2)
            y = y.reshape([y.shape[-4]* y.shape[-3],y.shape[-2], y.shape[-1]])
            y = y.permute(1,2,0)
        return y


class TPLinearLora(TPLinear):
    def __init__(self, layer, axis=0, tp=4, dp=1, allgather_output=False):
        super().__init__(layer, axis, tp, dp, allgather_output)
        self.active_adapter = layer.active_adapter
        # self.lora_A = TPLinear(layer.lora_A[self.active_adapter], axis, tp, dp, allgather_output=True).parallelize()
        self.lora_A = layer.lora_A[self.active_adapter]
        self.lora_B = TPLinear(layer.lora_B[self.active_adapter], axis, tp, dp, allgather_output=allgather_output).parallelize()
        self.lora_dropout = layer.lora_dropout
        self.scaling = layer.scaling

    def forward(self, x):
        previous_dtype = x.dtype
        result = super().forward(x)
        result += (
            self.lora_B(
                self.lora_A(self.lora_dropout[self.active_adapter](x))
            )
            * self.scaling[self.active_adapter]
        )
        result = result.to(previous_dtype)
        return result


class TPQuantLinearLora(nn.Module):
    def __init__(self, layer, num_group, axis=0, tp=4, dp=1, allgather_output=False):
        super().__init__()
        self.layer = TPGroupQuantLinear.from_model(layer, num_group, tp, axis, allgather_output)
        self.active_adapter = layer.active_adapter
        self.lora_A = layer.lora_A[self.active_adapter]
        self.lora_B = TPLinear(layer.lora_B[self.active_adapter], axis, tp, dp, allgather_output=allgather_output).parallelize()
        self.lora_dropout = layer.lora_dropout
        self.scaling = layer.scaling

    def parallelize(self):
        self.layer.parallelize()
        return self

    def forward(self, x):
        previous_dtype = x.dtype
        result = self.layer(x)
        result += (
            self.lora_B(
                self.lora_A(self.lora_dropout[self.active_adapter](x))
            )
            * self.scaling[self.active_adapter]
        )
        result = result.to(previous_dtype)
        return result


class QuantLinearLora(nn.Module):
    def __init__(self, layer, num_group):
        super().__init__()
        self.layer = GroupQuantLinear.from_model(layer, num_group)
        self.active_adapter = layer.active_adapter
        self.lora_A = layer.lora_A[self.active_adapter]
        self.lora_B = layer.lora_B[self.active_adapter]
        self.lora_dropout = layer.lora_dropout
        self.scaling = layer.scaling


    def forward(self, x):
        previous_dtype = x.dtype
        result = self.layer(x)
        result += (
            self.lora_B(
                self.lora_A(self.lora_dropout[self.active_adapter](x))
            )
            * self.scaling[self.active_adapter]
        )
        result = result.to(previous_dtype)
        return result

class TPEmbedding(nn.Module):
    def __init__(self, layer: nn.Embedding, tp=4):
        super().__init__()
        self.layer = layer
        self.tp = tp

    def __getattribute__(self, name):
        if name == "weight":
            return self.layer.weight
        else:
            return object.__getattribute__(self, name)
    def parallelize(self):
        with torch.no_grad():
            # Slice along the hidden size
            shape = self.layer.weight.data.shape
            print("W shape before", self.weight.data.shape)
            self.layer.weight.data = self.layer.weight.reshape(shape[0], self.tp, shape[1] // self.tp).permute(1, 0, 2).contiguous()
            print("W shape after", self.weight.data.shape)
        return self
    def forward(self, x):
        y = self.layer.forward(x)
        y = collectives.all_gather_cross_replica(y, self.tp)
        y = y.permute(0,3,1,2)
        y = y.reshape([y.shape[-4]* y.shape[-3],y.shape[-2], y.shape[-1]])
        y = y.permute(1,2,0)

        return y

# Cause segfault:
# class TPConv1d(nn.Module):
#     def __init__(self, layer: nn.Embedding, axis=0, tp=4, split_bias=False):
#         super().__init__()
#         self.layer = layer
#         self.tp = tp
#         self.axis = axis
#         self.split_bias = split_bias

#     def parallelize(self):
#         with torch.no_grad():
#             weights = self.layer.weight.data
#             shape = weights.shape
#             bias = self.layer.bias.data

#             print("weights shape before", self.layer.weight.shape)
#             if self.axis == 0: # split along "out_channels" -> need allgather to return to the original
#                 self.layer.weight.data = self.layer.weight.data.reshape(self.tp , shape[0] // self.tp , shape[1], *shape[2:]).contiguous()
#             elif self.axis == 1: # split along "in_channels/groups" -> need allreduce(Sum) to return to the original
#                 self.layer.weight.data = self.layer.weight.data.reshape(shape[0], self.tp , shape[1] // self.tp , *shape[2:]).permute(1, 0, 2, 3, 4).contiguous()
#             else:
#                 raise ValueError(f"Invalid axis '{self.axis}'. Only accepts '{list(range(len(shape[:-2])))}'")

#             if self.split_bias:
#                 self.layer.bias.data = self.layer.bias.data.reshape(self.tp , -1).contiguous()
#             else:
#                 self.out_bias = nn.Parameter(self.layer.bias.data.unsqueeze(0).repeat(self.tp , 1).contiguous())
#                 self.out_bias.requires_grad = self.layer.bias.requires_grad
#                 self.layer.bias = None
#             print("weights shape after", self.layer.weight.shape)
#             # self.layer.out_channels = self.layer.out_channels // self.tp

#     def forward(self, hidden_states):
#         print(hidden_states.shape)
#         y = self.layer.forward(hidden_states)
#         # print(y.shape)
#         # y = collectives.all_gather_cross_replica(y, self.tp)
#         # # print(y.shape)
#         # y = y.permute(0,2,1,3)
#         # y = y.reshape([y.shape[-4]* y.shape[-3],y.shape[-2], y.shape[-1]])
#         # y = y.permute(1,0,2)
#         # print(y.shape)
#         # y = all_reduce(y, self.tp_dim, self.dp_dim)
#         # if hasattr(self, "out_bias"):
#         #     y = y + self.out_bias[None, :, None, None]
#         # allgather here?
#         return y
