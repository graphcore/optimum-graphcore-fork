# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Optional, Tuple

import torch


FLOAT16_LIMIT = 1e4


class IPUAttentionMixin:
    """
    The aim of this class is to provide common, model-agnostic functionality such as KV caching and attention
    serialization to transformer attention layers.

    Currently, KV caching is implemented for self-attention, with more to follow.

    The intended usage is best demonstrated with an existing example, Whisper. There are roughly two steps:
    1. subclass the parent attention layer to inject this mixin, for example, `class IPUWhisperAttention(WhisperAttention, IPUAttentionMixin)`
    and use the `add_to_kv_cache` and `update_attention_mask` methods to add the KV values at the current time
    step to the cache.

    2. replace the existing attention layers with above via the provided class method `from_model`, e.g.
    `decoder_layer.self_attn = IPUWhisperAttention.from_model(decoder_layer.self_attn, use_cache=True, **kwargs)`.
    """

    _kv_cache_initialised: bool = False

    @property
    def kv_cache_initialised(self) -> bool:
        return self._kv_cache_initialised

    def _create_kv_cache(self, cache_shape: Tuple[int], dtype: torch.dtype, uses_beams=False):
        self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)
        self.register_buffer("_k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("_v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        if uses_beams:
            self.register_buffer("_beam_idx", torch.arange(cache_shape[0], dtype=torch.int32), persistent=False)
        self._kv_cache_initialised = True

    def _delete_kv_cache(self):
        if not self._kv_cache_initialised:
            return

        del self._generation_step
        del self._k_cache
        del self._v_cache
        if hasattr(self, "_beam_idx"):
            del self._beam_idx
        del self._kv_cache_initialised

    @classmethod
    def from_model(
        cls,
        attention_layer: torch.nn.Module,
        use_cache: bool = False,
        batch_size: int = 1,
        max_length: int = 128,
        num_beams: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        clone = copy.deepcopy(attention_layer)
        clone.__class__ = cls

        if use_cache:
            clone._create_kv_cache(
                (batch_size * num_beams, clone.num_heads, max_length, clone.head_dim),
                dtype=dtype,
                uses_beams=num_beams > 1,
            )

        return clone

    def to_model(self, cls) -> torch.nn.Module:
        self._delete_kv_cache()

        original = copy.deepcopy(self)
        original.__class__ = cls
        return original

    def add_to_kv_cache(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copies the key-value pair into their corresponding key-value caches. Each copy-cache pair is assumed
        to be of shape [batch_size, num_heads, 1, head_dim] and [batch_size, num_heads, max_length, head_dim] respectively.
        """
        if not self.kv_cache_initialised:
            raise ValueError(
                f"{self.__class__.__name__} assumes that self-attention has KV caching enabled. "
                f"Please instantiate using `{self.__class__.__name__}.from_model()` so the KV "
                "cache can be created."
            )

        if self.training:
            raise RuntimeError("KV caching is currently only supported for inference.")

        expected_key_shape, expected_value_shape = list(self._k_cache.shape), list(self._v_cache.shape)
        expected_key_shape[-2] = 1
        expected_value_shape[-2] = 1
        if list(key.shape) != expected_key_shape:
            raise ValueError(f"Expected key shape {expected_key_shape}, received {list(key.shape)}.")
        if list(value.shape) != expected_value_shape:
            raise ValueError(f"Expected value shape {expected_value_shape}, received {list(value.shape)}.")

        # For now assume that generation will always start from step 0.
        reset_kv_cache = self._generation_step == 0
        self._k_cache *= 1 - reset_kv_cache.to(self._k_cache.dtype)
        self._v_cache *= 1 - reset_kv_cache.to(self._v_cache.dtype)

        if hasattr(self, "_beam_idx"):
            # For beam search, permute the cache since inputs are permuted on host.
            _k_cache = torch.index_select(self._k_cache, 0, self._beam_idx)
            _v_cache = torch.index_select(self._v_cache, 0, self._beam_idx)
            self._k_cache.copy_(_k_cache)
            self._v_cache.copy_(_v_cache)

        # Dynamic update leads to uneven tile placement, and scatter leads to large re-arrangements,
        # so use a brute force matmul approach which empirically seems best for now.
        bsz, heads, src_len, head_dim = self._k_cache.shape
        mm_mask = (torch.arange(src_len) == self._generation_step).view(src_len, 1)
        _key = torch.matmul(mm_mask.to(key.dtype), key.view(bsz * heads, 1, head_dim))
        _value = torch.matmul(mm_mask.to(value.dtype), value.view(bsz * heads, 1, head_dim))
        self._k_cache += _key.view(self._k_cache.shape)
        self._v_cache += _value.view(self._v_cache.shape)

        return self._k_cache, self._v_cache

    def update_attention_mask(self, attention_mask: Optional[torch.Tensor] = None):
        """
        Creates a default mask up to and including the current generation step, marking the point
        up to which the caches have been populated.
        """
        bsz, _, src_len, _ = self._k_cache.shape
        mask = torch.full((1, src_len), -FLOAT16_LIMIT)
        mask_cond = torch.arange(src_len).view(1, src_len)
        mask.masked_fill_(mask_cond < self._generation_step + 1, 0)
        mask = mask.to(self._k_cache.dtype)
        mask = mask.expand(bsz, 1, 1, src_len)

        if attention_mask is not None:
            if attention_mask.size() != mask.size():
                raise ValueError(
                    f"Attention mask does not match expected KV cache mask dimensions. "
                    f"Received: {attention_mask.size()}, expected {mask.size()}."
                )
            mask = mask + attention_mask

        return mask
