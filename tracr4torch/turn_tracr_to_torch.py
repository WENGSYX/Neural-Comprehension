import torch
import torch.nn as nn
import jax
from model import Transformer
from tokenizer import Tokenizer
from prompts import *
from tracr.compiler import compiling
import dill
from collections import OrderedDict



class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


with open('../dataset/parity/parity.pkl','rb') as f:
    model = dill.load(f)



config_dict = {
    'dropout':model.model_config.dropout_rate,
    'd_model':model.params['token_embed']['embeddings'].shape[1],
    'num_heads':model.model_config.num_heads,
    'mlp_hidden_size':model.model_config.mlp_hidden_size,
    'layer_norm':model.model_config.layer_norm,
    'num_layers':model.model_config.num_layers,
    'vocab_size':model.params['token_embed']['embeddings'].shape[0],
    'key_size':model.model_config.key_size,
    'max_seq_len':model.params['pos_embed']['embeddings'].shape[0],
    'activation_function':torch.nn.ReLU()
}
config = Config(config_dict)

if 'NumericalEncoder' in str(type(model.output_encoder)):
    use_argmax = False
    unembedding = torch.zeros(config.d_model, 1)
    for kr,vr in {model.residual_labels[n]:n for n in range(len(model.residual_labels))}.items():
        if 'output_1' == kr:
            unembedding[vr,0] = 1
else:
    use_argmax = True
    unembedding = torch.zeros(config.d_model, len(model.output_encoder.encoding_map.values()))
    for k,v in model.output_encoder.encoding_map.items():
        for kr,vr in {model.residual_labels[n]:n for n in range(len(model.residual_labels))}.items():
            if 'map_8:'+str(k) == kr:
                unembedding[vr,v] = 1

torch_model = Transformer(config)
sd = torch_model.state_dict()
model_state = OrderedDict()


model_state['pos_embed.weight'] = torch.tensor(model.params['pos_embed']['embeddings'].tolist())
model_state['token_embed.weight'] = torch.tensor(model.params['token_embed']['embeddings'].tolist())
for i in range(config.num_layers):
    model_state['transformer.layers.{}.self_attn.query_proj.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/query'.format(i)]['w'].tolist()).T
    model_state['transformer.layers.{}.self_attn.value_proj.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/value'.format(i)]['w'].tolist()).T
    model_state['transformer.layers.{}.self_attn.key_proj.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/key'.format(i)]['w'].tolist()).T
    model_state['transformer.layers.{}.self_attn.out_proj.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/linear'.format(i)]['w'].tolist()).T
    model_state['transformer.layers.{}.linear1.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/mlp/linear_1'.format(i)]['w'].tolist()).T
    model_state['transformer.layers.{}.linear2.weight'.format(i)] = torch.tensor(model.params['transformer/layer_{}/mlp/linear_2'.format(i)]['w'].tolist()).T

    model_state['transformer.layers.{}.self_attn.query_proj.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/query'.format(i)]['b'].tolist())
    model_state['transformer.layers.{}.self_attn.value_proj.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/value'.format(i)]['b'].tolist())
    model_state['transformer.layers.{}.self_attn.key_proj.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/key'.format(i)]['b'].tolist())
    model_state['transformer.layers.{}.self_attn.out_proj.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/attn/linear'.format(i)]['b'].tolist())
    model_state['transformer.layers.{}.linear1.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/mlp/linear_1'.format(i)]['b'].tolist())
    model_state['transformer.layers.{}.linear2.bias'.format(i)] = torch.tensor(model.params['transformer/layer_{}/mlp/linear_2'.format(i)]['b'].tolist())


torch_model.load_state_dict(model_state)
torch_model.set_unembedding(unembedding)
tokenizer = Tokenizer(model.input_encoder.encoding_map,model.output_encoder.encoding_map)

with open('../dataset/parity/tokenizer.pt','wb') as f:
    dill.dump(tokenizer,f)
with open('../dataset/parity/model.pt','wb') as f:
    dill.dump(model,f)

"""
BCE = torch.nn.BCELoss(reduction='none')
tokenizer = Tokenizer(model.input_encoder.encoding_map,model.output_encoder.encoding_map)
input_ids = tokenizer.tokenize('The flip process of coins is: 0 1 1 0 ->')
input_ids = model.input_encoder.encode(['bos',1,1,0])
output = torch_model(torch.tensor(input_ids).unsqueeze(0))
text = tokenizer.decode(output)
"""
