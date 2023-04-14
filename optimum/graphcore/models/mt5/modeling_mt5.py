#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import poptorch
from optimum.utils import logging
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5Block, T5Stack

from ...generation_utils import IPUGenerationMixin, _IndexedInputLinear
from ...modeling_utils import (
    PipelineMixin,
    SerializedEmbedding,
    SharedEmbedding,
    SplitLinear,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
    split_encoder_decoder_ipu_config,
)
from ..t5.modeling_t5 import PipelinedT5ForConditionalGeneration


logger = logging.get_logger(__name__)


@register(MT5ForConditionalGeneration)
class PipelinedMT5ForConditionalGeneration(MT5ForConditionalGeneration, PipelinedT5ForConditionalGeneration):
    # exact copy from PipelinedT5ForConditionalGeneration
    @property
    def is_encoder_and_decoder_embeddings_computation_shared(self):
        return isinstance(self.shared, SharedEmbedding)
    
    def encoder_and_decoder_embeddings_computation(self, *args, **kwargs):
        return PipelinedT5ForConditionalGeneration.encoder_and_decoder_embeddings_computation(*args, **kwargs)

    # modified from PipelinedT5ForConditionalGeneration
    def parallelize(self, for_generation=False):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedMT5ForConditionalGeneration(config).parallelize().half()
        ```
        """
        # in training mode, this must be the case otherwise the model will not fit
        # this requires training and inference specific settings in the IPUConfig
        if self.ipu_config.embedding_serialization_factor < 2:
            raise ValueError("Embedding serialization factor must be greater than 1 for MT5 to run on the IPU.")

        self.shared = SerializedEmbedding(self.shared, self.ipu_config.embedding_serialization_factor)

        if for_generation:
            self.lm_head = SplitLinear(self.lm_head, splits=1, serialization_factor=self.ipu_config.embedding_serialization_factor)
        else:
            self.lm_head = SplitLinear(self.lm_head, splits=4, serialization_factor=self.ipu_config.embedding_serialization_factor)
        
        if self.config.tie_word_embeddings:
            raise ValueError("Tied input and output embeddings for MT5 are currently not supported on the IPU.")

        # Prevent `PipelinedT5ForConditionalGeneration.parallelize` from serialising the
        # embedding, we already split it above
        original_embedding_serialization_factor = self.ipu_config.embedding_serialization_factor
        self.ipu_config.embedding_serialization_factor = 1
        PipelinedT5ForConditionalGeneration.parallelize(for_generation=for_generation)
        self.ipu_config.embedding_serialization_factor = original_embedding_serialization_factor

        poptorch.removeBlocks(self.lm_head)
        if for_generation:
            last_ipu = self.decoder_ipu_config.ipus_per_replica - 1
            logger.info(f"LM Head Output --> IPU {last_ipu}")
            self.lm_head.split_linear_layers[0] = poptorch.BeginBlock(self.lm_head.split_linear_layers[0], f"LM Head Output {0}", ipu_id=last_ipu)
        else:
            last_ipu = self.ipu_config.ipus_per_replica - 1
            # TODO: need to make sure that splitting the lm head does not exceed ipus per replica for training
            logger.info(f"LM Head Output --> IPU {last_ipu - 1}-{last_ipu}")
            self.lm_head.split_linear_layers[0] = poptorch.BeginBlock(self.lm_head.split_linear_layers[0], f"LM Head Output {0}", ipu_id=last_ipu - 1)
            for i, _ in enumerate(self.lm_head.split_linear_layers[1:]):
                self.lm_head.split_linear_layers[i+1] = poptorch.BeginBlock(self.lm_head.split_linear_layers[i+1], f"LM Head Output {i+1}", ipu_id=last_ipu)
        
        logger.info("-----------------------------------------------------------")
        return self
    
    # modified from PipelinedT5ForConditionalGeneration
    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # T5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)

        self.encoder_and_decoder_embeddings_computation(False)
        # self.scale_down_weights(factor=1, restore=True)

        self.encoder.__class__ = T5Stack
        self.decoder.__class__ = T5Stack

        for block in self.encoder.block:
            block.__class__ = T5Block
        for block in self.decoder.block:
            block.__class__ = T5Block

        if self.lm_head.__class__ == _IndexedInputLinear:
            self.lm_head = self.lm_head.wrapped_linear
        self.lm_head = self.lm_head.deserialize()
        self.shared = self.shared.deserialize()


        return self

    def forward(self, *args, **kwargs) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return PipelinedT5ForConditionalGeneration.forward(*args, **kwargs)
