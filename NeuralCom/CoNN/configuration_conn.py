# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" CoNN model configuration"""
from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

CoNN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "parity": "https://huggingface.co/wengsyx/parity/resolve/main/config.json",
    "reverse": "https://huggingface.co/wengsyx/reverse/resolve/main/config.json",
    "last_letter": "https://huggingface.co/wengsyx/last_letter/resolve/main/config.json",
}


class CoNNConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CoNNModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    Examples:

    ```python
    >>> from transformers import CoNNConfig, CoNNModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = CoNNConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = CoNNModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "conn"

    def __init__(
        self,
        vocab_size=10,
        hidden_size=128,
        num_hidden_layers=12,
        num_attention_heads=1,
        intermediate_size=512,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=0,
        initializer_range=0.02,
        layer_norm=False,
        layer_norm_eps=0.02,
        pad_token_id=0,
        mlp_hidden_size=64,
        input_encoding_map={},
        output_encoding_map={},
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm = layer_norm
        self.layer_norm_eps = layer_norm_eps
        self.mlp_hidden_size = mlp_hidden_size
        self.input_encoding_map = input_encoding_map
        self.output_encoding_map = output_encoding_map
