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
# ==============================================================================

import os
import openai
import random
from tracr.compiler import compiling
from tracr.rasp import rasp
from tqdm import trange
from .prompts import *
from ..tracr4torch import get_CoNN
class AutoCoNN():
    def __init__(self, folder_name='./code_prompt'):

        self.code_prompt = [open(os.path.join(folder_name, file), 'r', encoding='utf-8').read() for file in
                            os.listdir(folder_name)]

        self.folder_name = folder_name
        self.CODE_PROMPT = """
from typing import List, Sequence
from tracr.rasp import rasp

Comparison.EQ: lambda key, query: key == query,
Comparison.LT: lambda key, query: key < query,
Comparison.LEQ: lambda key, query: key <= query,
Comparison.GT: lambda key, query: key > query,
Comparison.GEQ: lambda key, query: key >= query,
Comparison.NEQ: lambda key, query: key != query,
Comparison.TRUE: lambda key, query: True,
Comparison.FALSE: lambda key, query: False,
"""

    def pop_code_prompt(self, number=0):
        """pop the propmt code."""
        self.code_prompt.pop(number)

    def add_code_prompt(self, code, save=True, sudo_save=False):
        """add the new prompt code.
            code: the prompt code
            save: Whether to permanently save this code (meaning save to a folder)
            sudo_save: If the input prompt code is not an executable file, should it still be saved.
        """

        try:
            exec(code)
            if save:
                with open(os.path.join(self.folder_name, str(len(self.code_prompt) + 1) + '.py'), 'w',
                          encoding='utf-8') as f:
                    f.write('text')
            self.code_prompt.append(code)
        except:
            if sudo_save:
                if save:
                    with open(os.path.join(self.folder_name, str(len(self.code_prompt) + 1) + '.py'), 'w',
                              encoding='utf-8') as f:
                        f.write('text')
                self.code_prompt.append(code)
            else:
                exec(code)

    def __call__(self, instruct,vocab=['a','b','c'], example=[], LLM_name='text-davich-003', generation_number=10,
                 temperature=0.6, max_length=256, stop='\n\n', auto_save=True, return_torch=True):
        """The main process of generating a CoNN
        Args:
            instruct: the 'insturct' text for generated.
            vocab:
            example: the 'example' list for generated. example is a list containing at least one sample, each sample being a sub-list where the first item is the input to the CoNN model and the second item is the output of the CoNN model.
            LLM_name: the openai model (Recommended GPT-3.5 or GPT4).
            generation_number: Recommended GPT-3.5 or GPT4.
            temperature: the temperature for generated (The greater the possibility, the more diverse).
            max_length: the max generated text length (Complex CoNN requires longer text to be set).
            stop: auto stop for model. ('\n\n' is ok).
            auto_save: do you need to save the generated code as next prompt.
            return_torch: **IMPORTANT** Do you need to return the torch model (if False, return JAX model)
            """

        random.shuffle(self.code_prompt)
        function_text = ''
        prompt_text = ''.join(
            self.code_prompt[10:]) + f'def {model_name}(sop) -> rasp.SOp:\n    """{instruct}\n'
        if example != []:
            example_text = f'    Example usage:\n      model = CoNN()\n'
            for index in range(len(example)):
                example_text += f'      model({str(example[index][0])})\n      >> {str(example[index][1])}\n'
            prompt_text += example_text
        prompt_text += '"""\n'
        function_text = prompt_text
        prompt_text = self.CODE_PROMPT + prompt_text

        response = openai.Completion.create(
            engine=LLM_name,
            prompt=prompt_text,
            max_tokens=max_length,
            temperature=temperature,
            stop=stop,
            n=generation_number
        )

        res = [text["text"] for text in response["choices"]]
        tk = trange(len(res))
        models = []
        for index in tk:
            try:
                exec(function_text + res[index])
                program = CoNN(
                    sop=rasp.tokens)

                assembled_model = compiling.compile_rasp_to_model(
                    program=program,
                    vocab=vocab,
                    max_seq_len=10,
                    causal=False,
                    compiler_bos="bos",
                    compiler_pad="pad",
                    mlp_exactness=100)
                score = 0
                outputs = []
                for ex in example:
                    output = assembled_model.apply(["bos"]+ex[0]).decoded
                    outputs.append({'input':ex[0],'output':output,'target':ex[1]})
                    if sum([output[n] == ex[1][1+n] for n in range(len(ex[1]))]) == len(ex[1]):
                        score += 1

                RightModel = False
                if score == len(example):
                    RightModel = True
                models.append({'model':assembled_model,'RightModel':RightModel,'output':outputs,'code':function_text + res[index]})
                tk.set_postfix(model='{}/{}'.format(sum([model['RightModel'] for model in models]),index))
            except:
                pass




        if return_torch:
            model,tokenizer = get_CoNN(model=models[0])
            if auto_save:
                self.add_code_prompt(models[0]['code'])
                model.save_pretrained(instruct)
            return (model,tokenizer)
        else:
            return models






