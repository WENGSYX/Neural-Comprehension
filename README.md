# Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/WENGSYX/Neural-Comprehension.svg?color=blue&style=flat-square">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/WENGSYX/Neural-Comprehension">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/WENGSYX/Neural-Comprehension">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/WENGSYX/Neural-Comprehension">
</p>


**Authors**: Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, Jun Zhao 😎

**[Contact]** If you have any questions, feel free to contact me via (wengsyx@gmail.com).

This repository contains code, models, and other related resources of our paper ["Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks"](https://arxiv.org/abs/2304.01665).


****
* [2023/05/19] We have supported the one-click implementation of integration between CoNN and PLM!
* [2023/05/18] We have published the ["paper-v2"](https://arxiv.org/abs/2304.01665v2)!
* [2023/04/04] We have used huggingface to release the weight of the CoNN model!
* [2023/04/04] We have released the code for AutoCoNN!
* [2023/04/03] We have published the paper!
* [2023/03/26] We created the Github library!

****



### Install 

```
git clone https://github.com/WENGSYX/Neural-Comprehension
cd Neural-Comprehension
pip install .
```

To run neural comprehension, you need to install `PyTorch`, `Transformers`, `jax`, and `tracr`.
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

### Use AutoCoNN to create your CoNN

Please note that setting an OpenAI Key is required to use AutoCoNN (but not necessary if you're just experimenting with neural cognition and CoNN models).

```python
from NeuralCom.AutoCoNN import AutoCoNN

INSTRUCT = 'Create an SOp that is the last letter of a word'
VOCAB = ['a','b','c','d','e','f','g']
EXAMPLE = [[['a','b','c'],['c','c','c']],[['b','d'],['d','d']]]

auto = AutoCoNN()
model,tokenizer = auto(instruct=INSTRUCT,vocab=VOCAB,example=EXAMPLE)
```







### Use CoNN from huggingface

```python
from NeuralCom.CoNN.modeling_conn import CoNNModel
from NeuralCom.CoNN import Tokenizer


model = CoNNModel.from_pretrained('WENGSYX/CoNN_Reverse')
tokenizer = Tokenizer(model.config.input_encoding_map, model.config.output_encoding_map,model.config.max_position_embeddings)

output = model(tokenizer('r e v e r s e').unsqueeze(0))
print(tokenizer.decode(output.argmax(2)))
>>> [['bos', 'e', 's', 'r', 'e', 'v', 'e', 'r']]
```


### One-click implementation for Neural-Comprehension

```python
from transformers import AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM
from NeuralCom.CoNN.modeling_conn import CoNNModel
from NeuralCom.CoNN import Tokenizer as CoNNTokenizer
from NeuralCom.Model import NCModelForCoinFlip

PLM = AutoModelForSeq2SeqLM.from_pretrained('WENGSYX/PLM_T5_Base_coin_flip')
CoNN = CoNNModel.from_pretrained('WENGSYX/CoNN_Parity')
PLMTokenizer = AutoTokenizer.from_pretrained('WENGSYX/PLM_T5_Base_coin_flip')
CoNNTokenizer = CoNNTokenizer(CoNN.config.input_encoding_map, CoNN.config.output_encoding_map,CoNN.config.max_position_embeddings)

neural_comprehension = NCModelForCoinFlip(PLM, CoNN, PLMTokenizer, CoNNTokenizer).to('cuda:0')
input_text = "A coin is heads up. Aaron flips the coin. Julius does not flip the coin. Yixuan Weng flip the coin. Minjun Zhu does not flip the coin. Is the coin still heads up?"
input_tokens_PLM = PLMTokenizer.encode(input_text, return_tensors='pt')
generated_output = neural_comprehension.generate(input_tokens_PLM.to('cuda:0'))
generated_text = PLMTokenizer.decode(generated_output, skip_special_tokens=True)
print(f"Output: {generated_text}")
```


#### Huggingface Model

In each link, we provide detailed instructions on how to use the CoNN model.

| Model Name  | Model Size | Model Address                                             |
| ----------- | ---------- | --------------------------------------------------------- |
| Parity      | 2.2M       | [[link]](https://huggingface.co/WENGSYX/CoNN_Parity)      |
| Reverse     | 4.3M       | [[link]](https://huggingface.co/WENGSYX/CoNN_Reverse)     |
| Last Letter | 62.6K      | [[link]](https://huggingface.co/WENGSYX/CoNN_Last_Letter) |
| Copy        | 8.8K       | [[link]](https://huggingface.co/WENGSYX/CoNN_Copy)        |
| Add_Carry   | 117K       | [[link]](https://huggingface.co/WENGSYX/CoNN_Add_Carry)   |
| Sub_Carry   | 117K       | [[link]](https://huggingface.co/WENGSYX/CoNN_Sub_Carry)   |

###### If you have also created some amazing CoNN, you are welcome to share them publicly with us.


## 🌱 Neural-Comprehension's Roadmap 🌱


Our future plan includes but not limited to :
- [x] One-click implementation of integration between CoNN and PLM (huggingface)
- [ ] Combining CoNN with LLM (API-based)
- [ ] Demo Presentation

### 🙏Cite🙏


###### If you are interested in our paper, please feel free to cite it.
```
@misc{weng2023mastering,
      title={Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks}, 
      author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},
      year={2023},
      eprint={2304.01665},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
