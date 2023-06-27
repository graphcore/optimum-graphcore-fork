import argparse
import time

import numpy as np
import torch
from datasets import load_dataset

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined

from optimum.graphcore.models.whisper import WhisperProcessorTorch
from transformers import WhisperForConditionalGeneration


torch.set_num_threads(4)


WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2"]
IPU_CONFIGS = {
    "tiny": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "base": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "small": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "medium": IPUConfig(
        executable_cache_dir="./whisper_exe_cache", 
        inference_ipus_per_replica=4,
        inference_layers_per_ipu=[12, 12, 13, 11]),
    "large": IPUConfig(
        executable_cache_dir="./whisper_exe_cache",
        inference_ipus_per_replica=8,
        inference_layers_per_ipu=[8, 8, 8, 8, 6, 9, 9, 8],
    ),
    "large-v2": IPUConfig(
        executable_cache_dir="./whisper_exe_cache",
        inference_ipus_per_replica=8,
        inference_layers_per_ipu=[8, 8, 8, 8, 6, 9, 9, 8],
    ),
}
MAX_LENGTH = 448
ENCODER_MAX_LENGTH = 1500
TEST_IDX = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper `generate` IPU Benchmarks")
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
    args = parser.parse_args()

    model_name = f"openai/whisper-{args.model_size}"
    global_batch_size = args.batch_size * args.replication_factor

    processor = WhisperProcessorTorch.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    ipu_config = IPU_CONFIGS[args.model_size]
    ipu_config.inference_ipus_per_replica = 1 if args.use_cond_encoder else ipu_config.inference_ipus_per_replica
    ipu_config.inference_replication_factor = args.replication_factor
    ipu_config.eval()
    pipelined_model = to_pipelined(model, ipu_config)
    pipelined_model = pipelined_model.parallelize(
        for_generation=True,
        use_cache=True,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=MAX_LENGTH,
        use_cross_cache=args.use_cross_cache,
        encoder_max_length=ENCODER_MAX_LENGTH,
        on_device_generation_steps=args.on_device_generation_steps,
        use_encoder_output_buffer=args.on_device_generation_steps > 0 and not args.use_cond_encoder,
        use_cond_encoder=args.use_cond_encoder,
        batch_serialization_factor=args.batch_serialization_factor,
        sequence_serialization_factor=args.sequence_serialization_factor,
    )
    pipelined_model = pipelined_model.half()

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    test_examples = ds[TEST_IDX : TEST_IDX + global_batch_size]["audio"]

    print(f"Benchmarking transcription using librispeech examples.")
    print(f"Using {model_name} with {ipu_config.inference_replication_factor} replicas, each taking {ipu_config.inference_ipus_per_replica} IPU(s).")
    print(f"Micro batch size {args.batch_size} - num beams {args.num_beams}.")
    print(f"{args=}.")

    max_num_generated_tokens = 0
    preprocess_times = []
    generate_times = []
    decode_times = []
    total_times = []
    for _ in range(100):
        start_total_time = time.perf_counter()
        start_time = time.perf_counter()
        input_features = processor([test_example["array"] for test_example in test_examples], return_tensors="pt", sampling_rate=16000).input_features.half()
        end_time = time.perf_counter()
        preprocess_times.append(end_time - start_time)
    
        start_time = time.perf_counter()
        generated_tokens = pipelined_model.generate(
            input_features,
            use_cache=True,
            num_beams=args.num_beams,
            do_sample=False,
            max_length=MAX_LENGTH,
            min_length=3,
        )
        end_time = time.perf_counter()
        max_num_generated_tokens = generated_tokens.shape[1]
        generate_times.append(end_time - start_time)

        start_time = time.perf_counter()
        processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        end_time = time.perf_counter()
        decode_times.append(end_time - start_time)
        end_total_time = time.perf_counter()
        total_times.append(end_total_time - start_total_time)
    
    preprocess_mean = np.mean(preprocess_times[1:]) * 1000
    preprocess_std = np.std(preprocess_times[1:]) * 1000
    generate_mean = np.mean(generate_times[1:]) * 1000
    generate_std = np.std(generate_times[1:]) * 1000
    decode_mean = np.mean(decode_times[1:]) * 1000
    decode_std = np.std(decode_times[1:]) * 1000
    total_mean = np.mean(total_times[1:]) * 1000
    total_std = np.std(total_times[1:]) * 1000
    print(f"Preprocess: mean {preprocess_mean:1.1f}ms - std {preprocess_std:1.1f}ms.")
    print(f"Generate: mean {generate_mean:1.1f}ms - std {generate_std:1.1f}ms.")
    print(f"Decode: mean {decode_mean:1.1f}ms - std {decode_std:1.1f}ms.")
    print(f"Total: mean {total_mean:1.1f}ms - std {total_std:1.1f}ms.")
    print(f"Longest generated sequence has {max_num_generated_tokens} tokens.")
