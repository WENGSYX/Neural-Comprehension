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
"""PyTorch CoNN model."""


import math
import os
import copy
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_conn import CoNNConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "conn"
_CONFIG_FOR_DOC = "CoNNConfig"



BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    'CoNN_Parity',
    'CoNN_Reverse',
    'CoNN_Last_letter'
    # See all CoNN models at https://huggingface.co/models?filter=conn
]



class CoNNEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(torch.tensor(list(range(input_ids.size(1)))))
        embeddings += position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings




class CoNNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CoNNConfig
    base_model_prefix = "CoNN"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CoNNEncoder):
            module.gradient_checkpointing = value



CoNN_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CoNNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""



@add_start_docstrings(
    "The bare CoNN Model transformer outputting raw hidden-states without any specific head on top.",
    CoNN_START_DOCSTRING,
)
class CoNNModel(CoNNPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = CoNNEmbeddings(config)
        encoder_layer = TransformerEncoderLayer(config)
        self.transformer = TransformerEncoder(encoder_layer, config)
        self.unembedding = torch.nn.Parameter(torch.zeros(config.hidden_size, len(config.output_encoding_map.values())))
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )

    def set_unembedding(self,unembedding):
        self.unembedding = torch.nn.Parameter(unembedding)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        embedding_output = self.embeddings(
            input_ids=input_ids,
        )

        src = self.transformer(embedding_output, mask=None, src_key_padding_mask=None)
        output = (src[-1] @ self.unembedding)
        return output




class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.key_size = config.intermediate_size

        self.query_proj = nn.Linear(self.embed_dim, config.intermediate_size)
        self.key_proj = nn.Linear(self.embed_dim, config.intermediate_size)
        self.value_proj = nn.Linear(self.embed_dim, config.intermediate_size)

        self.out_proj = nn.Linear(config.intermediate_size, self.embed_dim)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        batch_size = query.size(0)

        # 通过投影获取 Q, K, V
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)

        # 计算 scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = nn.functional.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        # 加权求和
        attn_output = torch.matmul(attn_scores, v)

        # 将多头 attention 结果拼接
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.key_size)

        # 通过线性层计算输出
        attn_output = self.out_proj(attn_output)
        if key_padding_mask is not None:
            attn_output = attn_output.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        return attn_output, attn_scores
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiheadAttention(config)
        self.linear1 = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.mlp_hidden_size, config.hidden_size)
        self.config = config
        if self.config.layer_norm:
            self.ln = nn.LayerNorm(config.hidden_size)

    def layer_norm(self, src):
        if self.config.layer_norm:
            return self.ln(src)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.layer_norm(src)
        src2 = self.linear2(self.dropout(torch.nn.ReLU()(self.linear1(src))))
        src = src + self.dropout(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, config):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config.num_hidden_layers)])
        self.num_layers = config.num_hidden_layers
        self.config = config
        if self.config.layer_norm:
            self.ln = nn.LayerNorm(config.hidden_size)

    def layer_norm(self, src):
        if self.config.layer_norm:
            return self.ln(src)
        return src



    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        outputs = [output]
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = self.layer_norm(output)
            outputs.append(output)
        return outputs
