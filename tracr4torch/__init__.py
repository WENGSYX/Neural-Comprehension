import json
import torch
import torch
import torch.nn as nn
import re
import copy
import dill
from collections import OrderedDict
from CoNN.configuration_conn import CoNNConfig
from CoNN.modeling_conn import CoNNModel

def get_CoNN(file_path='',model=''):

    if model == '':
        with open(file_path, 'rb') as f:
            model = dill.load(f)
    config = CoNNConfig(vocab_size=model.params['token_embed']['embeddings'].shape[0],
        hidden_size=model.params['token_embed']['embeddings'].shape[1],
        num_hidden_layers= model.model_config.num_layers,
        num_attention_heads=model.model_config.num_heads,
        intermediate_size=model.model_config.key_size,
        mlp_hidden_size=model.model_config.mlp_hidden_size,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm=model.model_config.layer_norm,
        max_position_embeddings=model.params['pos_embed']['embeddings'].shape[0],
        pad_token_id=model.input_encoder.encoding_map['pad'],
        input_encoding_map=model.input_encoder.encoding_map,
        output_encoding_map=model.output_encoder.encoding_map)


    if 'NumericalEncoder' in str(type(model.output_encoder)):
        use_argmax = False
        unembedding = torch.zeros(config.hidden_size, 1)
        for kr, vr in {model.residual_labels[n]: n for n in range(len(model.residual_labels))}.items():
            if 'output_18' == kr:
                unembedding[vr, 0] = 1
    else:
        use_argmax = True
        unembedding = torch.zeros(config.hidden_size, len(model.output_encoder.encoding_map.values()))
        for k, v in model.output_encoder.encoding_map.items():
            for kr, vr in {model.residual_labels[n]: n for n in range(len(model.residual_labels))}.items():
                if re.search(r"^output_\d+:{}$".format(k), kr):
                    unembedding[vr, v] = 1

    model_state = OrderedDict()

    model_state['embeddings.position_embeddings.weight'] = torch.tensor(model.params['pos_embed']['embeddings'].tolist())
    model_state['embeddings.word_embeddings.weight'] = torch.tensor(model.params['token_embed']['embeddings'].tolist())
    for i in range(config.num_hidden_layers):
        model_state['transformer.layers.{}.self_attn.query_proj.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/query'.format(i)]['w'].tolist()).T
        model_state['transformer.layers.{}.self_attn.value_proj.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/value'.format(i)]['w'].tolist()).T
        model_state['transformer.layers.{}.self_attn.key_proj.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/key'.format(i)]['w'].tolist()).T
        model_state['transformer.layers.{}.self_attn.out_proj.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/linear'.format(i)]['w'].tolist()).T
        model_state['transformer.layers.{}.linear1.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/mlp/linear_1'.format(i)]['w'].tolist()).T
        model_state['transformer.layers.{}.linear2.weight'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/mlp/linear_2'.format(i)]['w'].tolist()).T

        model_state['transformer.layers.{}.self_attn.query_proj.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/query'.format(i)]['b'].tolist())
        model_state['transformer.layers.{}.self_attn.value_proj.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/value'.format(i)]['b'].tolist())
        model_state['transformer.layers.{}.self_attn.key_proj.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/key'.format(i)]['b'].tolist())
        model_state['transformer.layers.{}.self_attn.out_proj.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/attn/linear'.format(i)]['b'].tolist())
        model_state['transformer.layers.{}.linear1.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/mlp/linear_1'.format(i)]['b'].tolist())
        model_state['transformer.layers.{}.linear2.bias'.format(i)] = torch.tensor(
            model.params['transformer/layer_{}/mlp/linear_2'.format(i)]['b'].tolist())
    model_state['unembedding'] = unembedding
    conn = CoNNModel._from_config(config)
    conn.load_state_dict(model_state)
    conn.set_unembedding(unembedding)
    tokenizer = Tokenizer(model.config.input_encoding_map, model.config.output_encoding_map,model.config.max_position_embeddings)

    return conn,tokenizer


class Tokenizer:
    def __init__(self, encoding_map,decoding_map,max_length):
        self.vocab = encoding_map
        self.decoder_vocab = {v:k for k,v in decoding_map.items()}
        self.max_length = max_length
        if sum([type(n)==int for n in self.vocab.keys()]):
            self.vocab = {str(k):v for k,v in self.vocab.items()}

    def tokenize(self, text,return_tensor='pt'):
        tokens = []
        for word in text.strip().split():

            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens = [self.vocab.get('bos')] + tokens[-self.max_length+1:]
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens

    def __call__(self, text,return_tensor='pt'):
        tokens = []
        for word in text.strip().split():

            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens = [self.vocab.get('bos')] + tokens[-self.max_length+1:]
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens
    
    def decode(self,output):
        texts = [[] * output.size(0)]
        output = output.cpu().tolist()

        for n in range(len(output)):
            texts[n] = [str(self.decoder_vocab[x]) for x in output[n]]
        return texts
