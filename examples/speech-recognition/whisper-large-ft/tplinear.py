import poptorch
import torch
import torch.nn as nn
from poptorch_experimental_addons import collectives as collectives
from typing import List, Optional, Tuple, Union
from remap import tensor_remap

class TPLinear(nn.Linear):
    def __init__(self, layer: nn.Linear, axis=0, tp=4, dp=1, allgather_output=False, serialization=1):
        shape = layer.weight.data.shape
        super().__init__(shape[1], shape[0], dtype=layer.weight.data.dtype, bias=layer.bias is not None)
        self.weight.data = layer.weight.data
        if layer.bias is not None:
            self.bias.data = layer.bias.data
        self.allgather_output=allgather_output
        self.axis = axis
        self.tp = tp # tensor parallel replicas
        self.dp = dp # data parallel replicas
        self.serialization = serialization
        for param_src, param_dst in zip(layer.parameters(), self.parameters()):
            param_dst.requires_grad = param_src.requires_grad

    def parallelize(self):
        with torch.no_grad():
            axis = self.axis
            weights = self.weight.data
            shape = weights.shape
            assert shape[axis] % self.tp == 0
            print("W shape before", self.weight.data.shape)
            if axis == 0:
                self.weight.data = weights.reshape(self.tp, shape[0] // self.tp, shape[1]).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.reshape(self.tp, -1)


            elif axis == 1:
                self.weight.data = weights.reshape(shape[0], self.tp, shape[1] // self.tp).permute(1, 0, 2).contiguous()
                if self.bias is not None:
                    self.bias.data = self.bias.data.unsqueeze(0).repeat(self.tp, 1)
                    self.out_bias = nn.Parameter(self.bias.data)

                    self.out_bias.requires_grad = self.bias.requires_grad
                    self.bias = None

            print("W shape after", self.weight.data.shape)
            return self

    def serialized_version(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), poptorch.MatMulSerializationMode.OutputChannels,  self.serialization)
        # output = poptorch.serializedMatMul(x, self.weight.t(), poptorch.MatMulSerializationMode.ReducingDim,  self.serialization)
        #
        if self.bias is not None:
            output += self.bias
        return output

    def forward(self, x):
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
        print(x.shape)
        print(self.lora_A.weight.shape)
        print(self.lora_B.weight.shape)
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

class TPConv1d(nn.Module):
    def __init__(self, layer: nn.Embedding, axis=0, tp=4, split_bias=False):
        super().__init__()
        self.layer = layer
        self.tp = tp
        self.axis = axis
        self.split_bias = split_bias

    def parallelize(self):
        with torch.no_grad():
            weights = self.layer.weight.data
            shape = weights.shape
            bias = self.layer.bias.data

            print("weights shape before", self.layer.weight.shape)
            if self.axis == 0: # split along "out_channels" -> need allgather to return to the original
                self.layer.weight.data = self.layer.weight.data.reshape(self.tp , shape[0] // self.tp , shape[1], *shape[2:])
            elif self.axis == 1: # split along "in_channels/groups" -> need allreduce(Sum) to return to the original
                self.layer.weight.data = self.layer.weight.data.reshape(shape[0], self.tp , shape[1] // self.tp , *shape[2:]).permute(1, 0, 2, 3, 4)
            else:
                raise ValueError(f"Invalid axis '{self.axis}'. Only accepts '{list(range(len(shape[:-2])))}'")

            if self.split_bias:
                self.layer.bias.data = self.layer.bias.data.reshape(self.tp , -1).contiguous()
            else:
                self.out_bias = nn.Parameter(self.layer.bias.data.unsqueeze(0).repeat(self.tp , 1))
                self.out_bias.requires_grad = self.layer.bias.requires_grad
                self.layer.bias = None
            print("weights shape after", self.layer.weight.shape)
            # self.layer.out_channels = self.layer.out_channels // self.tp

    def forward(self, hidden_states):
        print(hidden_states.shape)
        y = self.layer.forward(hidden_states)
        # print(y.shape)
        # y = collectives.all_gather_cross_replica(y, self.tp)
        # print(y.shape)
        # y = y.permute(0,2,1,3)
        # y = y.reshape([y.shape[-4]* y.shape[-3],y.shape[-2], y.shape[-1]])
        # y = y.permute(1,0,2)
        # print(y.shape)
        # y = all_reduce(y, self.tp_dim, self.dp_dim)
        # if hasattr(self, "out_bias"):
        #     y = y + self.out_bias[None, :, None, None]
        # allgather here?
        return y
