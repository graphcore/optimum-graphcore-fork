from optimum.graphcore.models.whisper.modeling_whisper import IPUWhisperAttention, _WhisperEncoderLayerClamp, PipelinedWhisperForConditionalGeneration
from optimum.graphcore import IPUConfig
import torch
import torch.nn as nn
import poptorch
from poptorch import CommGroupType, VariableRetrievalMode
import copy
from tplinear import TPLinear, TPLinearLora, TPEmbedding, TPGroupQuantLinear, TPQuantLinearLora, QuantLinearLora, GroupQuantLinear
from transformers import AutoConfig
from optimum.graphcore.modeling_utils import recomputation_checkpoint, outline_attribute
from peft import LoraConfig, LoraConfig, get_peft_model
import functools
from transformers.trainer_pt_utils import get_parameter_names
from poptorch_experimental_addons import collectives as collectives
from typing import List
# ----------------- #
bs = 1
seq_len_encoder = 1500
seq_len_decoder = 224
hidd_size = 1280

tp = 4
dp = 1
pipeline = True
gradient_accumulation = 9
lora = True

use_encoders = True
use_decoders = True
tp_layers = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2", "embed_tokens", "proj_out"]
exclude = ["lora"]
lora_targets = ["q_proj", "v_proj"]
ipu_config = IPUConfig()
# ipu_config._projection_serialization_factor = 2
# ----------------- #

# inputs_enc = [torch.rand([bs*gradient_accumulation, seq_len_encoder, hidd_size], dtype=torch.float16)]
inputs_enc = [torch.randint(1, 1000, [bs*gradient_accumulation, 80, seq_len_encoder*2], dtype=torch.float16)]
inputs_dec = [torch.rand([bs*gradient_accumulation, seq_len_decoder, hidd_size], dtype=torch.float16)]
model = None
input_ids_dec = None

class DummyLoss(torch.nn.Module):
    def forward(self, x):
        return x, poptorch.identity_loss(x**2, reduction="sum")

def maybe_make_model(args):
    global model
    if model is not None:
        return model

    config = AutoConfig.from_pretrained("openai/whisper-large-v2")
    config.encoder_layers = args.num_encoders
    config.decoder_layers = args.num_decoders
    config.max_length = seq_len_decoder

    whisper_ipu = PipelinedWhisperForConditionalGeneration(config=config).half()
    whisper_ipu.change_encoder_layer_class(False)
    whisper_ipu.change_decoder_layer_class(restore=False)
    whisper_ipu.change_attention_class(False)
    whisper_ipu.change_decoder_class(restore=False)
    # whisper_ipu.change_encoder_class(restore=False)
    whisper_ipu.change_decoder_positional_embedding(restore=False)

    whisper_ipu.ipu_config = ipu_config
    whisper_ipu.change_lm_head(False, use_cache=True)
    whisper_ipu.tie_weights()
    whisper_ipu.generation_config.max_length = seq_len_decoder


    if lora:
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_targets, lora_dropout=0.05, bias="none")
        whisper_ipu = get_peft_model(whisper_ipu, lora_config)
        whisper_ipu.print_trainable_parameters()
        whisper_ipu = whisper_ipu.model


    def get_placement(split: List[int]):
        assert args.num_encoders == sum(args.encoder_splits)
        assert args.num_decoders == sum(args.decoder_splits)
        placement = {}
        layer_idx = 0
        for ipu in range(len(split)):
            for k in range(split[ipu]):
                placement[layer_idx] = ipu
                layer_idx +=1
        return placement

    class TestWhisperModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self._hooks = []

            # self.decoders = []
            if use_encoders:
                # self.encoders = whisper_ipu.model.encoder.layers
                self.encoder = whisper_ipu.model.encoder
            if use_decoders:
                self.decoder = whisper_ipu.model.decoder

            if use_encoders:
                for n, encoder in enumerate(self.encoder.layers):
                    # group = n % 2
                    self._hooks.append(recomputation_checkpoint(encoder))
                    # self._hooks.append(outline_attribute(encoder, f"encoder_{n}"))
            if use_decoders:
                for decoder in self.decoder.layers:
                    self._hooks.append(recomputation_checkpoint(decoder))
            self.proj_out = whisper_ipu.proj_out
            self.loss = DummyLoss()


        def pipeline(self):
            stages_dict = {}
            # for encoders [14,16,2,0] is fitting.
            encoders_placement = get_placement(args.encoder_splits)
            decoders_placement = get_placement(args.decoder_splits)
            self.encoder.conv1 = poptorch.BeginBlock(self.encoder.conv1, "Conv1", ipu_id=0)
            self.encoder.conv2 = poptorch.BeginBlock(self.encoder.conv2, "Conv2", ipu_id=0)
            self.encoder.embed_positions = poptorch.BeginBlock(
                self.encoder.embed_positions, "Embed Positions", ipu_id=0
            )
            last_ipu = 0
            stage_id = 0
            stages_dict[stage_id] = ["Conv1", "Conv2", "Embed Positions"]

            for k, ipu in encoders_placement.items():
                self.encoder.layers[k] = poptorch.BeginBlock(self.encoder.layers[k], f"Encoder{k}", ipu_id=ipu)
                print("Placing encoder ", k, "on IPU ", ipu)
                if ipu != last_ipu:
                    stage_id += 1
                    stages_dict[stage_id]=[f"Encoder{k}"]

                else:
                    stages_dict[stage_id].append(f"Encoder{k}")

                last_ipu = ipu


            self.encoder.layer_norm = poptorch.BeginBlock(
                self.encoder.layer_norm, "Encoder Layer Norm", ipu_id=last_ipu
            )
            stages_dict[stage_id].append("Encoder Layer Norm")

            print("Placing decoder embedding", "on IPU ", decoders_placement[0])
            self.decoder.embed_tokens = poptorch.BeginBlock(
                                            self.decoder.embed_tokens, "Decoder Embedding", ipu_id=decoders_placement[0]
                                        )
            if decoders_placement[0] != last_ipu:
                stage_id +=1
            stages_dict[stage_id]=["Decoder Embedding"]
            last_ipu = decoders_placement[0]
            for k, ipu in decoders_placement.items():
                self.decoder.layers[k] = poptorch.BeginBlock(self.decoder.layers[k], f"Decoder{k}", ipu_id=ipu)
                print("Placing decoder ", k, "on IPU ", ipu)
                if ipu != last_ipu:
                    stage_id += 1
                    stages_dict[stage_id]=[f"Decoder{k}"]

                else:
                    stages_dict[stage_id].append(f"Decoder{k}")

                last_ipu = ipu

            self.decoder.layer_norm = poptorch.BeginBlock(
                self.decoder.layer_norm, "Decoder Layer Norm", ipu_id=last_ipu
            )
            stages_dict[stage_id].append("Decoder Layer Norm")
            stage_id+=1
            print("Placing dout projection", k, "on IPU ", decoders_placement[0])

            self.proj_out = poptorch.BeginBlock(self.proj_out, "Output Projection", ipu_id=decoders_placement[0])
            self.loss = poptorch.BeginBlock(self.loss, "Loss", ipu_id=decoders_placement[0])
            stages_dict[stage_id]=["Output Projection", "Loss"]
            print("Pipeline stages dict", stages_dict)
            return stages_dict

        def tie_weights(self):
            self.proj_out.weight = self.decoder.embed_tokens.weight

        def forward(self, x, ids=None):
            print(self.decoder.embed_tokens)
            y = [x]
            if use_encoders:
                y = self.encoder(y[0])
            if use_decoders:
                hidden_states = self.decoder(input_ids=ids, encoder_hidden_states=y[0])[0]
                hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], tp, hidden_states.shape[2] // tp)
                hidden_states = hidden_states.permute(2,0,1,3)
                hidden_states = collectives.all_to_all_single_cross_replica(hidden_states, tp)[0]
                print("before proj", hidden_states.shape)
                y = self.proj_out(hidden_states)
            out, loss = self.loss(y)
            return out, loss

    model = TestWhisperModel().half()

    if use_encoders:
        for encoder in model.encoder.layers:
            encoder.self_attn._sequence_serialization_factor = 5

    if use_decoders:
        for decoder in model.decoder.layers:
            decoder.self_attn._sequence_serialization_factor = 4
    return model

def contains_any(name:str, layers):
    for l in layers:
        if l in name:
            return True
    return False

def check_layer(name, exclude=[]):
    if contains_any(name, exclude):
            return False
    if contains_any(name,tp_layers):
            return True
    return False

def replace_linear_layers(model):
    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)
    # using wonder's beautiful simplification:
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [obj] + attr.split('.'))

    layers = []
    for n, l in model.named_modules():
        if check_layer(n, exclude=exclude) and isinstance(l, nn.Linear):
            print("Parallelizing Linear layer ", n)
            allgather_output = False
            if lora and contains_any(n, lora_targets):
                l  = TPLinearLora(l, tp=tp, axis=0, allgather_output=allgather_output)
                # l = TPQuantLinearLora(l, 16, tp=tp, axis=0, allgather_output=allgather_output)
                # l = QuantLinearLora(l, 16)
            else:
                l = TPLinear(l, tp=tp, axis=0, serialization=1, allgather_output=allgather_output)
                # l = TPGroupQuantLinear.from_model(l,16, tp=tp, axis=0, allgather_output=allgather_output)
                # l = GroupQuantLinear.from_model(l, 16)


            if "fc2" in n or "out_proj" in n:
                l.axis = 1
            if "fc" in n and "encoder" in n:
                l.serialization = 3
                l.serialization_mode = poptorch.MatMulSerializationMode.InputChannels
            if "fc" in n and "decoder" in n:
                l.serialization = 4
            if "encoder" in n:
                l.remap = True

            l.parallelize()
            layers.append((n,l))
            print("-----------")

        if check_layer(n, exclude=exclude) and "embed_tokens" in n :
            print("Parallelizing Embedding layer ", n)
            l = TPEmbedding(l, tp=tp)
            l.parallelize()
            layers.append((n,l))
            print("-----------")

    for n, l in layers:
        rsetattr(model, n, l)
    print(model)
    return model

def apply_replica_grouping(model, comm_group_type=CommGroupType.Consecutive, shards=1):
    for n, p in model.named_parameters():
        if check_layer(n, exclude = ["lora_A"]):
            model.per_replica_params[n] = (
                comm_group_type,
                shards,
                VariableRetrievalMode.OnePerGroup,
            )
            print("Applying tensor parallel to ", n)
    return model

def get_input(config):
    global input_ids_dec
    if input_ids_dec is None:
        input_ids_dec = torch.Tensor([config.pad_token_id for k in range(seq_len_decoder)]*bs).reshape(1,
            seq_len_decoder).expand(bs*gradient_accumulation, -1).to(torch.long)

    if not use_encoders:
        return (inputs_dec[0], input_ids_dec)
    else:
        return (inputs_enc[0], input_ids_dec)


def replace_layer_norm(model):
    class GatherNorm(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.norm = nn.LayerNorm(model.config.d_model)
            for param_src, param_dst in zip(layer.parameters(), self.norm.parameters()):
                param_dst.requires_grad = param_src.requires_grad
            self.norm.weight = layer.weight
            if layer.bias is not None:
                self.norm.bias = layer.bias

        def forward(self, x):
            x = collectives.all_gather_cross_replica(x, tp)
            x = x.permute(0,3,1,2)
            x = x.reshape([x.shape[-4]* x.shape[-3],x.shape[-2], x.shape[-1]])
            x = x.permute(1,2,0)
            x = self.norm(x)
            return x

    model.encoder.layer_norm = GatherNorm(model.encoder.layer_norm).half()
    model.decoder.layer_norm = GatherNorm(model.decoder.layer_norm).half()
    return model

def get_model(args):
    return copy.deepcopy(maybe_make_model(args))

def get_optimizer(model):
    # mimics optimum default AdamW with fp16
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = {name for name in decay_parameters if "bias" not in name}

    optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },

                ]
    print([n for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)])
    print([n for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)])
    optimizer =  poptorch.optim.AdamW(optimizer_grouped_parameters,
                               lr=0.2,
                               bias_correction=False,
                               accum_type=torch.float16,
                               first_order_momentum_accum_type=torch.float16,
                               second_order_momentum_accum_type=torch.float32
                               )
    optimizer.variable_attrs.markAsConstant("weight_decay")
    return optimizer

def get_options(stages_dict = None):
    options = poptorch.Options()
    if not pipeline:
        options._Popart.set("autoRecomputation", 4) # 4 = recompute all
    options._Popart.set("saveInitializersToFile", "weights.onnx")
    options._Popart.setEngineOptions({'opt.internalExchangeOptimisationTarget': 'memory'})
    # options._Popart.set("outlineThreshold", float(10))
    options._Popart.set("decomposeGradSum", True)
    # options.enableExecutableCaching("./exe_cache")
    options.setAvailableMemoryProportion({"IPU0": 0.2})
    # options.setAvailableMemoryProportion({"IPU0": 0.1, "IPU1": 0.1, "IPU2": 0.1, "IPU3": 0.1})


    options.Precision.setPartialsType(torch.half)
    # options.outputMode(poptorch.OutputMode.All)
    options.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            .useOnChipStorage(True)
            .useReplicatedTensorSharding(False) # DISABLE RTS
        )
    # options.Training.setConvolutionDithering(True)
    # import popart
    # # # PopART performance options from optimum #
    # # # Only stream needed tensors back to host
    # options._Popart.set("disableGradAccumulationTensorStreams", True)
    # # # Parallelize optimizer step update across IPUs
    # options._Popart.set(
    #     "accumulateOuterFragmentSettings.schedule",
    #     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized),
    # )
    # options._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # # # Enable patterns for better throughput and memory reduction
    # # # options._Popart.set("outlineThreshold", 10.0)
    # options._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    # options._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    # options._Popart.setPatterns(
    #     {"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True}
    # )

    # options.TensorLocations.setActivationLocation(poptorch.TensorLocationSettings().useOnChipStorage(False))
    if pipeline:
        assert stages_dict is not None
        options.Training.gradientAccumulation(gradient_accumulation)
        options.setExecutionStrategy(poptorch.PipelinedExecution(
            poptorch.Stage(*stages_dict[0]),
            poptorch.Stage(*stages_dict[1]),
            poptorch.Stage(*stages_dict[2]),
            poptorch.Stage(*stages_dict[3]),
            poptorch.Stage(*stages_dict[4]),

        ))
        # To cut compilation time:
        # options._Popart.set("timeLimitScheduler", 60.0)
        # options._Popart.set("swapLimitScheduler", 1000)
    return options

def run_normal(args):
    torch.manual_seed(0)
    model = get_model(args)
    options = get_options()
    input_tensor = get_input(model.config)
    optimizer = get_optimizer(model)
    ipu_model = poptorch.trainingModel(model, options, optimizer)

    result = ipu_model(input_tensor)
    print(result)

def run_tp(args):
    # configure the default attention method for tensor-parallel
    IPUWhisperAttention.tensor_parallel = tp

    torch.manual_seed(0)
    model = replace_linear_layers(get_model(args))
    model = replace_layer_norm(model)
    model.tie_weights()
    stages_dict = model.pipeline()

    input_tensors = get_input(model.config)
    options = get_options(stages_dict)

    # tp specific options
    options.replicationFactor(tp*dp)
    options.inputReplicaGrouping(tp, CommGroupType.Consecutive)
    options.outputMode(poptorch.OutputMode.All)
    options._Popart.setPatterns({"OpToIdentity": True})
    options.setExecutionStrategy(poptorch.ShardedExecution())

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, "is trainable")
    optimizer = get_optimizer(model)
    ipu_model = poptorch.trainingModel(model, options, optimizer)
    apply_replica_grouping(ipu_model)



    result = ipu_model(*input_tensors)
    print(result[0][0])



if __name__ == "__main__":
    import os
    from tap import Tap

    class SimpleArgumentParser(Tap):
        report_name: str  = "test" # popvision report
        num_encoders: int = 32 # total encoder layers
        num_decoders: int = 32 # total decder layers
        encoder_splits: List[int] = [15, 17, 0, 0] # encoder layers per ipu
        decoder_splits: List[int] = [0, 0, 16, 16] # decoder layers per ipu


    args = SimpleArgumentParser().parse_args()
    # os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "debug.instrument":"false", "autoReport.directory":"./report/full-encoder"}'
    # run_normal()
    os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "debug.instrument":"false", "autoReport.directory":"./report/'+args.report_name+'","profiler.replicaToProfile": "0"}'
    os.environ["POPART_LOG_LEVEL"] = 'INFO'

    run_tp(args)
    print("profile available under ./report/", args.report_name)

