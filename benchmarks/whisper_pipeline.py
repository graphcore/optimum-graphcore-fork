import argparse
import os
import time

# Suppress spurious PipelineChunkIterator warnings
import warnings

import numpy as np
import requests
import torch
from transformers import WhisperForConditionalGeneration

from optimum.graphcore import pipeline
from optimum.graphcore.models.whisper import WhisperProcessorTorch

from .whisper_configs import WHISPER_IPU_CONFIGS


warnings.filterwarnings("ignore", message="Length of IterableDataset.*was reported to be")

torch.set_num_threads(4)


WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2"]
MAX_LENGTH = 448
ENCODER_MAX_LENGTH = 1500
TEST_IDX = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper pipelines IPU Benchmarks")
    parser.add_argument(
        "--input-file", default="http://www.archive.org/download/greatexpectations_01_dickens_128kb.mp3", type=str
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=WHISPER_MODEL_SIZES,
        help=f"Model size, one of {WHISPER_MODEL_SIZES}.",
    )
    parser.add_argument("--use-cross-cache", action="store_true", help="Enable cross KV caching.")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--replication-factor", default=1, type=int)
    parser.add_argument("--on-device-generation-steps", default=16, type=int)
    parser.add_argument("--batch-serialization-factor", default=1, type=int)
    parser.add_argument("--sequence-serialization-factor", default=1, type=int)
    parser.add_argument("--use-cond-encoder", action="store_true")
    parser.add_argument("--use-group-quantized-linears", action="store_true")
    parser.add_argument("--chunk-length-s", type=int, default=30)
    args = parser.parse_args()

    input_file = args.input_file
    fn = os.path.basename(input_file)
    if not os.path.exists(fn):
        print(f"{fn} not found locally.")
        if not (input_file.startswith("http://") or input_file.startswith("https://")):
            print("Please provide a valid URL, expecting a prefix of `http://` or `https://`.")
            exit(1)
        print(f"Trying to download {input_file} to current directory.")
        r = requests.get(input_file)
        with open(fn, "wb") as fd:
            fd.write(r.content)

    model_name = f"openai/whisper-{args.model_size}"
    global_batch_size = args.batch_size * args.replication_factor

    processor = WhisperProcessorTorch.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    ipu_config_name = args.model_size + ("-quantized" if args.use_group_quantized_linears else "")
    ipu_config = WHISPER_IPU_CONFIGS[ipu_config_name]
    if not args.use_cond_encoder and ipu_config.inference_ipus_per_replica == 1:
        raise ValueError("ipu_config.inference_ipus_per_replica=1 requires --use-cond-encoder.")
    elif args.use_cond_encoder and ipu_config.inference_ipus_per_replica > 1:
        print("Overriding ipu_config.inference_ipus_per_replica to 1 since --use-cond-encoder was provided.")
        ipu_config.inference_ipus_per_replica = 1
    ipu_config.inference_replication_factor = args.replication_factor
    ipu_config.explicit_ir_inference = args.use_group_quantized_linears

    ipu_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        framework="pt",
        ipu_config=ipu_config,
        fp16=True,
        parallelize_kwargs={
            "use_cache": True,
            "batch_size": args.batch_size,
            "num_beams": args.num_beams,
            "max_length": MAX_LENGTH,
            "use_cross_cache": args.use_cross_cache,
            "encoder_max_length": ENCODER_MAX_LENGTH,
            "on_device_generation_steps": args.on_device_generation_steps,
            "use_encoder_output_buffer": args.on_device_generation_steps > 0 and not args.use_cond_encoder,
            "use_cond_encoder": args.use_cond_encoder,
            "use_group_quantized_linears": args.use_group_quantized_linears,
            "batch_serialization_factor": args.batch_serialization_factor,
            "sequence_serialization_factor": args.sequence_serialization_factor,
        },
        batch_size=args.batch_size * args.replication_factor,
        chunk_length_s=args.chunk_length_s,
        ignore_warning=True,
    )

    print(f"Benchmarking transcription of {input_file}.")
    print(
        f"Using {model_name} with {ipu_config.inference_replication_factor} replicas, each taking {ipu_config.inference_ipus_per_replica} IPU(s)."
    )
    print(f"Micro batch size {args.batch_size} - num beams {args.num_beams}.")
    print(f"{args=}.")

    total_times = []
    for _ in range(10 + 1):
        start_time = time.perf_counter()
        whisper_output = ipu_pipeline(
            [fn],
            generate_kwargs={
                "do_sample": False,
                "num_beams": args.num_beams,
                "use_cache": True,
                "max_length": MAX_LENGTH,
            },
        )
        end_time = time.perf_counter()
        total_times.append(end_time - start_time)

    total_mean = np.mean(total_times[1:]) * 1000
    total_std = np.std(total_times[1:]) * 1000
    print(f"Total (pre-process, generation, decoding): mean {total_mean:1.1f}ms - std {total_std:1.1f}ms.")
