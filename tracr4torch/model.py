import torch
import torch.nn as nn
import copy
import dill
from .tokenizer import Tokenizer
from collections import OrderedDict

def get_CoNN(file_path):
    """最后一层一定要设置为output"""
    with open(file_path+'/parity.pkl', 'rb') as f:
        model = dill.load(f)

    config = {
        'dropout': model.model_config.dropout_rate,
        'd_model': model.params['token_embed']['embeddings'].shape[1],
        'num_heads': model.model_config.num_heads,
        'mlp_hidden_size': model.model_config.mlp_hidden_size,
        'layer_norm': model.model_config.layer_norm,
        'num_layers': model.model_config.num_layers,
        'vocab_size': model.params['token_embed']['embeddings'].shape[0],
        'key_size': model.model_config.key_size,
        'max_seq_len': model.params['pos_embed']['embeddings'].shape[0],
        'activation_function': torch.nn.ReLU()
    }


    if 'NumericalEncoder' in str(type(model.output_encoder)):
        use_argmax = False
        unembedding = torch.zeros(config['d_model'], 1)
        for kr, vr in {model.residual_labels[n]: n for n in range(len(model.residual_labels))}.items():
            if 'output_1' == kr:
                unembedding[vr, 0] = 1
    else:
        use_argmax = True
        unembedding = torch.zeros(config['d_model'], len(model.output_encoder.encoding_map.values()))
        for k, v in model.output_encoder.encoding_map.items():
            for kr, vr in {model.residual_labels[n]: n for n in range(len(model.residual_labels))}.items():
                if 'map_8:' + str(k) == kr:
                    unembedding[vr, v] = 1

    model_state = OrderedDict()

    model_state['pos_embed.weight'] = torch.tensor(model.params['pos_embed']['embeddings'].tolist())
    model_state['token_embed.weight'] = torch.tensor(model.params['token_embed']['embeddings'].tolist())
    for i in range(config['num_layers']):
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
    model2 = Transformer(config)
    model2.load_state_dict(model_state)
    model2.set_unembedding(unembedding)
    tokenizer = Tokenizer(model.input_encoder.encoding_map, model.output_encoder.encoding_map)

    return model2,tokenizer


class CoNN():
    def __init__(self,model,tokenizer):

        self.model = model
        self.tokenizer = tokenizer


    def __call__(self,texts):
        outputs = []
        for i in texts:
            input_ids = self.tokenizer.tokenize(i)
            output = self.model(torch.tensor(input_ids).unsqueeze(0)).argmax(-1)
            text = self.tokenizer.decode(output)[0][-1]
            outputs.append(text)
        return outputs

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config['d_model']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        self.key_size = config['key_size']

        self.query_proj = nn.Linear(self.embed_dim, config['key_size'])
        self.key_proj = nn.Linear(self.embed_dim, config['key_size'])
        self.value_proj = nn.Linear(self.embed_dim, config['key_size'])

        self.out_proj = nn.Linear(config['key_size'], self.embed_dim)

        self.dropout = nn.Dropout(config['dropout'])

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
        self.linear1 = nn.Linear(config['d_model'], config['mlp_hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.linear2 = nn.Linear(config['mlp_hidden_size'], config['d_model'])
        self.config = config
        if self.config['layer_norm']:
            self.ln = nn.LayerNorm(config.d_model)

    def layer_norm(self, src):
        if self.config['layer_norm']:
            return self.ln(src)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.layer_norm(src)
        src2 = self.linear2(self.dropout(self.config['activation_function'](self.linear1(src))))
        src = src + self.dropout(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, config):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config['num_layers'])])
        self.num_layers = config['num_layers']
        self.config = config
        if self.config['layer_norm']:
            self.ln = nn.LayerNorm(config['d_model'])

    def layer_norm(self, src):
        if self.config['layer_norm']:
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

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Embedding(config['max_seq_len'],config['d_model'])
        encoder_layer = TransformerEncoderLayer(config)
        self.transformer = TransformerEncoder(encoder_layer, config)
        self.config = config

    def set_unembedding(self,unembedding):
        self.unembedding = torch.nn.Parameter(unembedding)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        token = self.token_embed(src)
        pos = self.pos_embed(torch.tensor(list(range(src.size(1)))))
        src = token + pos
        src = self.transformer(src, mask=mask, src_key_padding_mask=src_key_padding_mask)

        output = (src[-1] @ self.unembedding)
        return output