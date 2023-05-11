# coding=utf-8
# Copyright 2021 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import random
import re
import string
import unittest
from collections.abc import Iterable
from typing import Any, Dict, Optional, Set

import pytest

from optimum.graphcore import IPUConfig
from optimum.graphcore.ipu_configuration import IncompatibleIPUConfigError
from optimum.graphcore.modeling_utils import get_layer_ipu, split_encoder_decoder_ipu_config
from poptorch import OutputMode


def create_ipu_config() -> IPUConfig:
    initial_dict = IPUConfig().to_dict()
    allowed_output_modes = ["all", "sum", "final"]
    initial_dict["output_mode"] = random.choice(allowed_output_modes)
    # Setting this setting to False as it is currently not supported and will throw an error.
    initial_dict["execute_encoder_on_cpu_for_generation"] = False
    return IPUConfig.from_dict(initial_dict)


class IPUConfigTester(unittest.TestCase):
    def _test_to_options(self, for_inference: bool):
        def make_poptorch_options_comparable_to_ipu_config(options_dict):
            options_dict = copy.deepcopy(options_dict)
            # mapping specifies how to transform an options dict entry to something that can be compared to an IPUConfig.
            # It maps a string to either another string or a function:
            #   1. String -> string: it specifies how to align values for attribute names that differ between
            #       poptorch.Options and IPUConfig.
            #   2. String -> function: the function must return a tuple of type (str, Any), the first element being the
            #       name of the attribute to update, and the second one being the actual value to use for the update.
            mapping = {
                "gradient_accumulation": "gradient_accumulation_steps",
                # Seed?
                "location_optimizer.onChip": lambda d: (
                    "optimizer_state_offchip",
                    not d["location_optimizer"]["onChip"],
                ),
                "location_optimizer.useReplicatedTensorSharding": lambda d: (
                    "replicated_tensor_sharding",
                    bool(d["location_optimizer"]["useReplicatedTensorSharding"]),
                ),
                # This works for matmul_proportion because the randomly generated IPUConfig will set matmul_proportion
                # to some float value, meaning that this value will be replicated for all the IPUs, so there is only
                # one value in d["available_memory_proportion"].values(), which should be equal to
                # ipu_config.matmul_proportion
                "available_memory_proportion": lambda d: (
                    "matmul_proportion",
                    set(d["available_memory_proportion"].values()).pop(),
                ),
                "cachePath": "executable_cache_dir",
                "output_mode": lambda d: ("output_mode", OutputMode(d["output_mode"]).name.lower()),
            }
            for k, v in mapping.items():
                try:
                    if isinstance(v, str):
                        value = options_dict
                        keys = k.split(".")
                        for key in keys:
                            value = value[key]
                        options_dict[v] = value
                    else:
                        key_name, value = v(options_dict)
                        options_dict[key_name] = value
                except KeyError:
                    continue

            return options_dict

        def intersection_of_dicts(d1, d2):
            d1 = copy.deepcopy(d1)
            d2 = copy.deepcopy(d2)
            d1 = {k: v for k, v in d1.items() if k in d2}
            d2 = {k: v for k, v in d2.items() if k in d1}
            return d1, d2

        ipu_config = create_ipu_config()
        options = ipu_config.to_options(for_inference=for_inference)
        ipu_config_dict = ipu_config.to_dict()
        if for_inference:
            ipu_config_dict["replication_factor"] = ipu_config_dict["inference_replication_factor"]
            ipu_config_dict["device_iterations"] = ipu_config_dict["inference_device_iterations"]
            ipu_config_dict["gradient_accumulation_steps"] = 1
            ipu_config_dict["output_mode"] = "all"
        ipu_config_dict, options_dict = intersection_of_dicts(
            ipu_config_dict, make_poptorch_options_comparable_to_ipu_config(options.toDict())
        )
        self.assertEqual(ipu_config_dict, options_dict)

    def test_to_options(self):
        return self._test_to_options(False)

    def test_to_options_for_inference(self):
        return self._test_to_options(True)

    def _test_batch_size_factor(self, for_inference: bool):
        # Case 1: the IPUConfig is not "specialized" and contains values for many pod types.
        ipu_config = create_ipu_config()
        batch_size_factor = ipu_config.batch_size_factor(for_inference=for_inference)
        replication_factor = (
            ipu_config.inference_replication_factor if for_inference else ipu_config.replication_factor
        )
        gradient_accumulation_steps = 1 if for_inference else ipu_config.gradient_accumulation_steps
        device_iterations = ipu_config.inference_device_iterations if for_inference else ipu_config.device_iterations
        self.assertEqual(
            replication_factor * gradient_accumulation_steps * device_iterations,
            batch_size_factor,
        )
        # Case 2: the IPUConfig is specialized, no pod type needs to be specified to compute the batch size factor.
        ipu_config = create_ipu_config()
        batch_size_factor = ipu_config.batch_size_factor(for_inference=for_inference)
        replication_factor = (
            ipu_config.inference_replication_factor if for_inference else ipu_config.replication_factor
        )
        gradient_accumulation_steps = 1 if for_inference else ipu_config.gradient_accumulation_steps
        device_iterations = ipu_config.inference_device_iterations if for_inference else ipu_config.device_iterations
        self.assertEqual(
            replication_factor * gradient_accumulation_steps * device_iterations,
            batch_size_factor,
        )

    def test_batch_size_factor(self):
        self._test_batch_size_factor(False)

    def test_batch_size_factor_for_inference(self):
        self._test_batch_size_factor(True)

    def test_layers_per_ipu(self):
        # ipus_per_replica inferred from layers_per_ipu
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        self.assertEqual(ipu_config.ipus_per_replica, 2)

        # get_layer_ipu with specific layers_per_ipu
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 3)
        self.assertEqual(layer_ipu, [0, 1, 1])

        # No target number of layers specified
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        layer_ipu = get_layer_ipu(ipu_config)
        self.assertEqual(layer_ipu, [0, 1, 1])

        # Raises exception if number of layers is too few
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match="layers_per_ipu does not define the correct number of layers for the current model",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 2)

        # Raises exception if number of layers is too many
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match="layers_per_ipu does not define the correct number of layers for the current model",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 4)

        # layers_per_ipu and ipus_per_replica mismatch raises
        ipu_config = IPUConfig(layers_per_ipu=[1, -1], ipus_per_replica=4)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"layers_per_ipu has non-default value set, but its length does not match ipus_per_replica",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 3)

        # Default config everything should be on single IPU
        ipu_config = IPUConfig()
        layer_ipu = get_layer_ipu(ipu_config, 9)
        self.assertEqual(layer_ipu, [0] * 9)

        # Spread layers across 2 IPUs
        ipu_config = IPUConfig(ipus_per_replica=2)
        layer_ipu = get_layer_ipu(ipu_config, 6)  # even
        self.assertEqual(layer_ipu, [0, 0, 0, 1, 1, 1])
        layer_ipu = get_layer_ipu(ipu_config, 7)  # odd
        self.assertEqual(layer_ipu, [0, 0, 0, 1, 1, 1, 1])

        # Wild card
        ipu_config = IPUConfig(layers_per_ipu=[2, -1])
        layer_ipu = get_layer_ipu(ipu_config, 6)
        self.assertEqual(layer_ipu, [0, 0, 1, 1, 1, 1])
        ipu_config = IPUConfig(layers_per_ipu=[-1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 6)
        self.assertEqual(layer_ipu, [0, 0, 0, 0, 1, 1])
        ipu_config = IPUConfig(layers_per_ipu=[-1, 2, -1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 7)
        self.assertEqual(layer_ipu, [0, 1, 1, 2, 2, 3, 3])

    def test_execution_mode_specific_options(self):
        base_expected_values = {
            "layers_per_ipu": [0, 1, 2, 2, 2, 2, 1, 0],
            "inference_layers_per_ipu": [0, 3, 7, 0],
            "matmul_proportion": [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.1],
            "inference_matmul_proportion": [0.2, 0.3, 0.7, 0.2],
            "ipus_per_replica": 8,
            "inference_ipus_per_replica": 4,
        }

        # serialized_layer_splits_per_ipu and serialization_factor options are
        # mutually exclusive
        serialized_splits_per_ipu_expected_values = {
            "serialized_projection_splits_per_ipu": [0, 0, 0, 0, 0, 0, 2, 2],
            "inference_serialized_projection_splits_per_ipu": [0, 0, 1, 2],
            "serialized_embedding_splits_per_ipu": [2, 2, 0, 0, 0, 0, 0, 0],
            "inference_serialized_embedding_splits_per_ipu": [1, 1, 0, 0],
        }

        serialization_factor_expected_values = {
            "projection_serialization_factor": 4,
            "inference_projection_serialization_factor": 3,
            "embedding_serialization_factor": 4,
            "inference_embedding_serialization_factor": 2,
        }

        for extended_expected_values in (
            serialized_splits_per_ipu_expected_values,
            serialization_factor_expected_values,
        ):
            expected_values = {**base_expected_values, **extended_expected_values}
            ipu_config = IPUConfig(**expected_values)

            # Default mode is training
            self.assertEqual(ipu_config.mode, "training")

            # Training versions retreived
            for option, value in expected_values.items():
                if "inference" not in option:
                    self.assertEqual(getattr(ipu_config, f"_{option}"), value)

            # Inference options created when mode is training
            opts = ipu_config.train().to_options(for_inference=True)
            self.assertEqual(
                opts._values["available_memory_proportion"],
                {
                    ipu_id: matmul_proportion
                    for ipu_id, matmul_proportion in zip(
                        range(expected_values["inference_ipus_per_replica"]),
                        expected_values["inference_matmul_proportion"],
                    )
                },
            )

            self.assertEqual(ipu_config.mode, "training")

            # Inference versions retreived
            ipu_config.eval()
            for option, value in expected_values.items():
                if "inference" in option:
                    self.assertEqual(getattr(ipu_config, f"_{option[len('inference_'):]}"), value)

            # Test encoder decoder model IPUConfig splitting for generation
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 3, 7)
            self.assertEqual(e_ipu_config._layers_per_ipu, [0, 3])
            self.assertEqual(e_ipu_config._ipus_per_replica, 2)
            self.assertEqual(d_ipu_config._layers_per_ipu, [7, 0])
            self.assertEqual(d_ipu_config._ipus_per_replica, 2)

            if "serialized_projection_splits_per_ipu" in expected_values:
                self.assertEqual(e_ipu_config._serialized_embedding_splits_per_ipu, [1, 1])
                self.assertEqual(e_ipu_config._serialized_projection_splits_per_ipu, None)
                self.assertEqual(d_ipu_config._serialized_embedding_splits_per_ipu, None)
                self.assertEqual(d_ipu_config._serialized_projection_splits_per_ipu, [1, 2])

        # ipus_per_replica not specified
        ipu_config = IPUConfig(layers_per_ipu=[1, 2, 3, 4])
        self.assertEqual(ipu_config._ipus_per_replica, 4)

        # layers_per_ipu wildcard
        ipu_config = IPUConfig(ipus_per_replica=4, layers_per_ipu=[-1])
        layer_ipu = get_layer_ipu(ipu_config, 8)
        self.assertEqual(ipu_config._ipus_per_replica, 4)
        self.assertEqual(layer_ipu, [0, 0, 1, 1, 2, 2, 3, 3])

        # inference_matmul_proportion not specified but matmul_proportion is
        ipu_config = IPUConfig(
            layers_per_ipu=[1, 2, 3, 4],
            matmul_proportion=[0.1, 0.2, 0.3, 0.4],
            inference_layers_per_ipu=[3, 7],
        )
        self.assertEqual(ipu_config.inference_matmul_proportion, 0.2)

    def test_split_encoder_decoder_ipu_config(self):
        # Test splitting two IPUs
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 1, 2)
        self.assertEqual(e_ipu_config.layers_per_ipu, [1])
        self.assertEqual(e_ipu_config.ipus_per_replica, 1)
        self.assertEqual(d_ipu_config.layers_per_ipu, [2])
        self.assertEqual(d_ipu_config.ipus_per_replica, 1)

        # Test splitting matmul_proportion
        ipu_config = IPUConfig(layers_per_ipu=[2, 2, 2, 2], matmul_proportion=[0.1, 0.2, 0.3, 0.4])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 4, 4)
        self.assertEqual(e_ipu_config.matmul_proportion, [0.1, 0.2])
        self.assertEqual(d_ipu_config.matmul_proportion, [0.3, 0.4])

        # For generation using encoder decoder models and layers `SplitLinear`
        # and `SerializedEmbedding` placed on different IPUs, cannot
        # have serialized_{linear/embedding}_splits_per_ipu present in
        # both the encoder and decoder
        for test_attr in ("projection", "embedding"):
            with pytest.raises(ValueError, match=f"must have all splits placed on the"):
                failing_ipu_config = IPUConfig(
                    layers_per_ipu=[0, 2, 2, 0],
                    **{f"serialized_{test_attr}_splits_per_ipu": [0, 2, 2, 0]},
                )
                e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(failing_ipu_config, 2, 2)

        # Test that all the other values from the original ipu_config are intact
        ipu_config = ipu_config.to_dict()
        e_ipu_config = e_ipu_config.to_dict()
        d_ipu_config = d_ipu_config.to_dict()
        skip = {"layers_per_ipu", "ipus_per_replica", "matmul_proportion"}
        for k in ipu_config.keys():
            if re.search("|".join(skip), k):
                continue
            self.assertEqual(ipu_config[k], e_ipu_config[k], k)
            self.assertEqual(ipu_config[k], d_ipu_config[k], k)

        # Test that wildcards work
        ipu_config = IPUConfig(layers_per_ipu=[-1, -1])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 3)
        self.assertEqual(e_ipu_config.layers_per_ipu, [2])
        self.assertEqual(d_ipu_config.layers_per_ipu, [3])

        # Wrong number of layers should raise an exception
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"layers_per_ipu does not define the correct number of layers for the current model",
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 2)

        # Encoder and decoder layers defined on same IPU should raise an exception
        ipu_config = IPUConfig(layers_per_ipu=[4, 3])
        with pytest.raises(
            IncompatibleIPUConfigError, match=r"Unable to find valid split of ipu_config.layers_per_ipu"
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 3, 4)

        # If ipu_config only has 1 IPU then it should raise and exception
        ipu_config = IPUConfig(layers_per_ipu=[4])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"Need ipus_per_replica of at least 2 to split ipu_config into encoder and decoder configs",
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 2)

        # Handle empty IPU in last stage of encoder
        ipu_config = IPUConfig(layers_per_ipu=[1, 2, 3, 0, 5, 6, 7, 8])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 26)
        self.assertEqual(e_ipu_config.layers_per_ipu, [1, 2, 3, 0])
        self.assertEqual(e_ipu_config.ipus_per_replica, 4)
        self.assertEqual(d_ipu_config.layers_per_ipu, [5, 6, 7, 8])
        self.assertEqual(d_ipu_config.ipus_per_replica, 4)

        # Split where first IPU has 0 layers
        ipu_config = IPUConfig(layers_per_ipu=[0, 6, 0, 6])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 6)
        self.assertEqual(e_ipu_config.layers_per_ipu, [0, 6])
        self.assertEqual(e_ipu_config.ipus_per_replica, 2)
        self.assertEqual(d_ipu_config.layers_per_ipu, [0, 6])
        self.assertEqual(d_ipu_config.ipus_per_replica, 2)

        # Split where there are many zeros
        ipu_config = IPUConfig(layers_per_ipu=[3, 0, 3, 0, 0, 7])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 7)
        self.assertEqual(e_ipu_config.layers_per_ipu, [3, 0, 3, 0])
        self.assertEqual(e_ipu_config.ipus_per_replica, 4)
        self.assertEqual(d_ipu_config.layers_per_ipu, [0, 7])
        self.assertEqual(d_ipu_config.ipus_per_replica, 2)

    def test_attribute_value_validation(self):
        ipu_config = IPUConfig()

        # *layers_per_ipu attributes (List[int>=-1]) cannot contain
        # values less than -1
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            test_attr = ipu_config._get_managed_attr_mode_name("layers_per_ipu")
            with pytest.raises(ValueError, match=f"`IPUConfig` attribute `{test_attr}` must have all elements >= -1"):
                setattr(ipu_config, test_attr, [3, 5, -2])
            # should not raise
            setattr(ipu_config, test_attr, [1, 2, 3])

        # *matmul proportion attributes cannot contain values less than 0
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            test_attr = ipu_config._get_managed_attr_mode_name("matmul_proportion")
            with pytest.raises(ValueError, match=f"`IPUConfig` attribute `{test_attr}` must have all elements >= 0"):
                setattr(ipu_config, test_attr, [-0.5, 0, 0.5])
            # should not raise
            setattr(ipu_config, test_attr, [0.5, 0.5])

        # Scalar attributes like *replication_factor must be atleast 1
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            for attribute in (
                "replication_factor",
                "gradient_accumulation_steps",
                "ipus_per_replica",
                "projection_serialization_factor",
                "embedding_serialization_factor",
                "device_iterations",
            ):
                test_attr = ipu_config._get_managed_attr_mode_name(attribute)
                with pytest.raises(ValueError, match=f"`IPUConfig` attribute `{test_attr}` must be >= 1"):
                    setattr(ipu_config, test_attr, 0)
                # should not raise
                setattr(ipu_config, test_attr, 8)

        # output mode must be one of ("all", "sum", "final", "default")
        with pytest.raises(ValueError, match=f"`IPUConfig` attribute `output_mode` can only take values in"):
            ipu_config.output_mode = "reduce"
        # should not raise
        ipu_config.output_mode = "final"

        # Validation of serialized_{projection/embedding}_splits_per_ipu
        # attributes

        # Must be of type List[int>=0]
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            for attribute in ("projection", "embedding"):
                test_attr = ipu_config._get_managed_attr_mode_name(f"serialized_{attribute}_splits_per_ipu")
                with pytest.raises(
                    ValueError, match=f"`IPUConfig` attribute `{test_attr}` must have all elements >= 0"
                ):
                    setattr(ipu_config, test_attr, [0, 0, -1, 1])
                # Should not raise
                setattr(ipu_config, test_attr, [0, 0, 1, 1])

        # Must have atleast 1 split if the pipeline is provided
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            for attribute in ("projection", "embedding"):
                test_attr = ipu_config._get_managed_attr_mode_name(f"serialized_{attribute}_splits_per_ipu")
                with pytest.raises(
                    ValueError,
                    match=re.escape(f"`IPUConfig` attribute `{test_attr}=[0, 0]` must have atleast 1 split"),
                ):
                    setattr(ipu_config, test_attr, [0, 0])
                # Should not raise
                setattr(ipu_config, test_attr, [1, 0])

        # Splits should be on consecutive IPUs
        for mode in ipu_config.modes:
            ipu_config.mode = mode
            for attribute in ("projection", "embedding"):
                test_attr = ipu_config._get_managed_attr_mode_name(f"serialized_{attribute}_splits_per_ipu")
                with pytest.raises(
                    ValueError,
                    match=re.escape(f"`IPUConfig` attribute `{test_attr}=[0, 3, 0, 2]` must have its splits on"),
                ):
                    setattr(ipu_config, test_attr, [0, 3, 0, 2])
                # Should not raise
                setattr(ipu_config, test_attr, [0, 3, 2, 0])

    def test_validate_ipu_config(self):
        # If *matmul_proportion is a List[float], it must
        # use the same number of IPUs as *ipus_per_replica
        ipus_per_replica = 4
        matmul_proportion = [0.2] * (ipus_per_replica + 1)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=re.escape(f"matmul_proportion={matmul_proportion} should use the same number"),
        ):
            IPUConfig(ipus_per_replica=ipus_per_replica, matmul_proportion=matmul_proportion)
        # Should not raise
        matmul_proportion = [0.2] * ipus_per_replica
        IPUConfig(ipus_per_replica=ipus_per_replica, matmul_proportion=matmul_proportion)

        inference_matmul_proportion = [0.2] * (ipus_per_replica - 1)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=re.escape(f"inference_matmul_proportion={inference_matmul_proportion} should use the same number"),
        ):
            IPUConfig(ipus_per_replica=ipus_per_replica, inference_matmul_proportion=inference_matmul_proportion)
        # Should not raise
        inference_matmul_proportion = [0.2] * ipus_per_replica
        IPUConfig(ipus_per_replica=ipus_per_replica, inference_matmul_proportion=inference_matmul_proportion)

        # If there are no wildcards in *layers_per_ipu, the pipeline length
        # should equal *ipus_per_replica
        layers_per_ipu = [2] * (ipus_per_replica + 1)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=re.escape(f"layers_per_ipu={layers_per_ipu} should use the same number"),
        ):
            IPUConfig(ipus_per_replica=ipus_per_replica, layers_per_ipu=layers_per_ipu)
        # Should not raise
        layers_per_ipu = [2] * ipus_per_replica
        IPUConfig(ipus_per_replica=ipus_per_replica, layers_per_ipu=layers_per_ipu)

        inference_layers_per_ipu = [2] * (ipus_per_replica - 1)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=re.escape(f"inference_layers_per_ipu={inference_layers_per_ipu} should use the same number"),
        ):
            IPUConfig(ipus_per_replica=ipus_per_replica, inference_layers_per_ipu=inference_layers_per_ipu)
        # Should not raise
        inference_layers_per_ipu = [2] * ipus_per_replica
        IPUConfig(ipus_per_replica=ipus_per_replica, inference_layers_per_ipu=inference_layers_per_ipu)

        # Validation of non-transformer layer placement configuration

        # If the user has not provided either {projection/embedding}_serialization_factor
        # or serialized_{projection/embedding}_splits_per_ipu, {projection/embedding}_serialization_factor}
        # should default to 1
        ipu_config = IPUConfig()
        self.assertEqual(ipu_config.projection_serialization_factor, 1)
        self.assertEqual(ipu_config.inference_projection_serialization_factor, 1)
        self.assertEqual(ipu_config.embedding_serialization_factor, 1)
        self.assertEqual(ipu_config.inference_embedding_serialization_factor, 1)

        # The user cannot provide both {projection/embedding}_serialization_factor and
        # serialized_{projection/embedding}_splits_per_ipu
        for test_pair in (
            ("projection_serialization_factor", "serialized_projection_splits_per_ipu"),
            ("inference_projection_serialization_factor", "inference_serialized_projection_splits_per_ipu"),
            ("embedding_serialization_factor", "serialized_embedding_splits_per_ipu"),
            ("inference_embedding_serialization_factor", "inference_serialized_embedding_splits_per_ipu"),
        ):
            with pytest.raises(ValueError, match=f"Only one of `{test_pair[0]}` and `{test_pair[1]}`"):
                IPUConfig(**{test_pair[0]: 1, test_pair[1]: [1, 1]})

        # The pipeline length of serialized_{projection/embedding}_splits_per_ipu must equal
        # the number of IPUs specified by IPUs per replica
        for test_attr in (
            "serialized_projection_splits_per_ipu",
            "inference_serialized_projection_splits_per_ipu",
            "serialized_embedding_splits_per_ipu",
            "inference_serialized_embedding_splits_per_ipu",
        ):
            with pytest.raises(
                ValueError, match=re.escape(f"{test_attr}=[0, 0, 1, 1] should use the same number of IPUs as")
            ):
                IPUConfig(ipus_per_replica=8, **{test_attr: [0, 0, 1, 1]})

        # Test validation after construction
        ipu_config = IPUConfig(ipus_per_replica=ipus_per_replica, layers_per_ipu=[-1])
        layers_per_ipu = [2, 2, 2]
        ipu_config.layers_per_ipu = layers_per_ipu
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=re.escape(f"layers_per_ipu={layers_per_ipu} should use the same number"),
        ):
            ipu_config.validate_ipu_config()
        # Should not raise
        ipu_config.layers_per_ipu = [2] * ipus_per_replica
        ipu_config.validate_ipu_config()
