# coding=utf-8
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

import unittest


class TestCustomOps(unittest.TestCase):
    def test_group_quantize_compiles(self):
        # Importing quantize should compile
        from optimum.graphcore.quantization import group_quantize  # noqa: F401

    def test_sdk_version_hash(self):
        from optimum.graphcore.custom_ops.sdk_version_hash import sdk_version_hash

        hash = sdk_version_hash()
        self.assertTrue(type(hash) is str)
