from optimum.graphcore import IPUConfig


WHISPER_IPU_CONFIGS = {
    "tiny": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "base": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "small": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "medium": IPUConfig(
        executable_cache_dir="./whisper_exe_cache",
        inference_ipus_per_replica=4,
        inference_layers_per_ipu=[12, 12, 13, 11],
    ),
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
    "tiny-quantized": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=1),
    "base-quantized": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=1),
    "small-quantized": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=1),
    "medium-quantized": IPUConfig(executable_cache_dir="./whisper_exe_cache", inference_ipus_per_replica=2),
    "large-quantized": IPUConfig(
        executable_cache_dir="./whisper_exe_cache",
        inference_ipus_per_replica=4,
        inference_layers_per_ipu=[16, 16, 14, 18],
        inference_matmul_proportion=0.1,
        inference_projection_serialization_factor=5,
    ),
    "large-v2-quantized": IPUConfig(
        executable_cache_dir="./whisper_exe_cache",
        inference_ipus_per_replica=4,
        inference_layers_per_ipu=[16, 16, 14, 18],
        inference_matmul_proportion=0.1,
        inference_projection_serialization_factor=5,
    ),
}
