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
    def __init__(self):

        self.code_prompt = ['def make_length() -> rasp.SOp:\n    """Creates the `length` SOp using selector width primitive.\n    Example usage:\n      length = make_length()\n      length("abcdefg")\n      >> [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]\n    Returns:\n      length: SOp mapping an input to a sequence, where every element\n        is the length of that sequence.\n    """\n    all_true_selector = rasp.Select(\n        rasp.indices, rasp.tokens, rasp.Comparison.TRUE).named(\n        "all_true_selector")  # Match tokens and tokens one by one, and calculate that they are completely equal.\n    return rasp.SelectorWidth(all_true_selector).named(\n        "length")  # which computes the number of elements in each row of a selector that with the value 1.\n\nnow_length = make_length()\n',
 'def make_reverse(sop: rasp.SOp) -> rasp.SOp:\n    """Create an SOp that reverses a sequence, using length primitive.\n    Example usage:\n      reverse = make_reverse(rasp.tokens)\n      reverse("Hello")\n      >> [\'o\', \'l\', \'l\', \'e\', \'H\']\n    Args:\n      sop: an SOp\n    Returns:\n      reverse : SOp that reverses the input sequence.\n    """\n    opp_idx = (now_length - rasp.indices).named("opp_idx")  # Get the indices from back to front.\n    opp_idx = (opp_idx - 1).named("opp_idx-1")  # opp_idx - 1, so that the first digit of indices = 0.\n    reverse_selector = rasp.Select(rasp.indices, opp_idx,\n                                   rasp.Comparison.EQ).named(\n        "reverse_selector")  # Use opp_idx to query indices, get the Select.\n    return rasp.Aggregate(reverse_selector, sop).named(\n        "reverse")  # Aggregate the reverse_selector and sop, so that output the reverse sop\n\n\nlength = make_length()\n\n\n',
 'def make_sort_freq(max_seq_len: int) -> rasp.SOp:\n    """Returns tokens sorted by the frequency they appear in the input.\n    Tokens the appear the same amount of times are output in the same order as in\n    the input.\n    Example usage:\n      sort = make_sort_freq(rasp.tokens, rasp.tokens, 5)\n      sort([2, 4, 2, 1])\n      >> [2, 2, 4, 1]\n    Args:\n      max_seq_len: Maximum sequence length (used to ensure keys are unique)\n    """\n    hist = -1 * make_hist().named("hist")\n    return make_sort(\n        rasp.tokens, hist, max_seq_len=max_seq_len, min_key=1).named("sort_freq")\n\n\n### Programs that work under both causal and regular evaluation.\n\n\n',
 'def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:\n    """Count the fraction of previous tokens where a specific condition was True.\n     (As implemented in the RASP paper.)\n    Example usage:\n      num_l = make_frac_prevs(rasp.tokens=="l")\n      num_l("hello")\n      >> [0, 0, 1/3, 1/2, 2/5]\n    Args:\n      bools: SOp mapping a sequence to a sequence of booleans.\n    Returns:\n      frac_prevs: SOp mapping an input to a sequence, where every element\n        is the fraction of previous "True" tokens.\n    """\n    bools = rasp.numerical(bools)  # Turn to numerically-encoded.\n    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)  # Consider fraction of previous tokens.\n    return rasp.numerical(rasp.Aggregate(prevs, bools,\n                                         default=0)).named(\n        "frac_prevs")  # Produces an sop that averages the value of the s-op weighted by the selection matrix\n\n\n',
 'def make_count_less_freq(n: int) -> rasp.SOp:\n    """Returns how many tokens appear fewer than n times in the input.\n    The output sequence contains this count in each position.\n    Example usage:\n      count_less_freq = make_count_less_freq(2)\n      count_less_freq(["a", "a", "a", "b", "b", "c"])\n      >> [3, 3, 3, 3, 3, 3]\n      count_less_freq(["a", "a", "c", "b", "b", "c"])\n      >> [6, 6, 6, 6, 6, 6]\n    Args:\n      n: Integer to compare token frequences to.\n    """\n    hist = make_hist().named("hist")  # Returns the number of times each token occurs in the input.\n    select_less = rasp.Select(hist, hist,\n                              lambda x, y: x <= n).named(\n        "select_less")  # Judge whether the number of each token is less than \'n\'.\n    return rasp.SelectorWidth(select_less).named(\n        "count_less_freq")  # Add all the values which the number of token is less than \'n\'.\n\n\n',
 'def make_hist() -> rasp.SOp:\n    """Returns the number of times each token occurs in the input.\n     (As implemented in the RASP paper.)\n    Example usage:\n      hist = make_hist()\n      hist("abac")\n      >> [2, 1, 2, 1]\n    """\n    same_tok = rasp.Select(rasp.tokens, rasp.tokens,\n                           rasp.Comparison.EQ).named(\n        "same_tok")  # Match tokens and tokens one by one, and calculate that they are string equal.\n    return rasp.SelectorWidth(same_tok).named(\n        "hist")  # Count the equal number for each token if the same_tok value = 1.\n\n\n',
 'def make_count(sop, token):\n    """Returns the count of `token` in `sop`.\n    The output sequence contains this count in each position.\n    Example usage:\n      count = make_count(tokens, "a")\n      count(["a", "a", "a", "b", "b", "c"])\n      >> [3, 3, 3, 3, 3, 3]\n      count(["c", "a", "b", "c"])\n      >> [1, 1, 1, 1]\n    Args:\n      sop: Sop to count tokens in.\n      token: Token to count.\n    """\n    return rasp.SelectorWidth(rasp.Select(\n        sop, sop, lambda k, q: k == token))\\\n        .named(f"count_{token}")  # First determine which token in the sop are \'token\', and set the value =1, then add all the values while value = 1.\n\n\n',
 'def add(sop, number):\n    """Returns the value of sop added number\n    Example usage:\n      model = add(tokens, 2)\n      model([1, 4, 7, 2, 3])\n      >> [3, 6, 9, 4, 5]\n    """\n    return rasp.SequenceMap(lambda x, i: x + number, sop, rasp.indices).named(f"add_{number}")\n\n\n',
 'def atoi(sop):\n    """Converts all text to number, and uses 0 for strings of types other than numbers, It may be mixed with \'str\' or \'int\'.\n    Example usage:\n      model = atoi(tokens)\n      model([\'1\', \'4\', \'-\', \'2\', \'3\'])\n      >> [1, 4, 0, 2, 3]\n    """\n    return rasp.SequenceMap(lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices).named(\n        "atoi")  # Converts all text to number， and other token set 0.\n\n\n',
 'def Duplicate_detection(sop):\n    """If the token in the input is repeated, return it; otherwise, return 0\n      Example usage:\n      model = Duplicate_detection(tokens)\n      model([\'h\', \'e\', \'l, \'l\', \'o\'])\n      >> [0, 0, \'l\', \'l\', 0]\n    """\n    hist = make_hist()  # Count the number of times each token occurs in the input.\n    return rasp.SequenceMap(lambda x, i: i if x > 1 else 0, hist, rasp.tokens).named(\n        "Duplicate_detection")  # If the token in hist is repeated, return it; otherwise, return 0.\n\n\n',
 "def ralign(sop, default='-'):\n    c = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: '_', sop), rasp.Comparison.EQ), rasp.Map(lambda x: 1, sop))\n    return rasp.Aggregate(rasp.Select(rasp.indices + c, rasp.indices, rasp.Comparison.EQ), sop, default='-')\n\n\n",
 'def shift(sop, index):\n    """Shift all of the tokens in a sequence to the right by index positions.\n    Each token moves the index cell to the right, use \'_\' for the left moving part fill.\n    Example usage:\n      model = shift(sop, token="-")\n      model([\'h\', \'e\', \'l\', \'l\', \'o\'])\n      >> [\'_\', \'_\', \'h\', \'e\', \'l\']\n    Please note that meaningful tokens need to be placed on the far right.\n    Args:\n      sop: Sop to shift in.\n      index: Number of right moves.\n    """\n    idx = (rasp.indices + index).named("idx-1")  # Get the target indices.\n    selector = rasp.Select(idx, rasp.indices,\n                           rasp.Comparison.EQ).named("shift_selector")  # Use opp_idx to query indices, get the Select.\n    shift = rasp.Aggregate(selector, sop).named("shift")  # Aggregates the sops and selectors (converted from indexes).\n    return rasp.SequenceMap(lambda x, i: x if i >= index else "_", shift, rasp.indices).named(f"shift_{index}")\n\n\n',
 'def make_reverse(sop: rasp.SOp) -> rasp.SOp:\n    """Create an SOp that reverses a sequence, using length primitive.\n    Example usage:\n      reverse = make_reverse(rasp.tokens)\n      reverse("Hello")\n      >> [\'o\', \'l\', \'l\', \'e\', \'H\']\n    Args:\n      sop: an SOp\n    Returns:\n      reverse : SOp that reverses the input sequence.\n    """\n    opp_idx = (length - rasp.indices).named("opp_idx")  # Get the indices from back to front.\n    opp_idx = (opp_idx - 1).named("opp_idx-1")  # opp_idx - 1, so that the first digit of indices = 0.\n    reverse_selector = rasp.Select(rasp.indices, opp_idx,\n                                   rasp.Comparison.EQ).named(\n        "reverse_selector")  # Use opp_idx to query indices, get the Select.\n    return rasp.Aggregate(reverse_selector, sop).named(\n        "reverse")  # Aggregate the reverse_selector and sop, so that output the reverse sop\n\n\n',
 'def split(sop, token, index):\n    """using \'token\' as the separator string \'sop\', return the \'index\' th and then align right.\n    Example usage:\n      text = split(tokens, "+", 0)\n      text([4, 2, "+", 5, 6])\n      >> [0, 0, 0, 4, 2]\n      text = split(tokens, "-", 1)\n      text([8, 1, "-", 5, 7])\n      >> [0, 0, 0, 5, 7]\n    Args:\n      sop: Sop to count tokens in.\n      token: Token to count.\n      index: After split, token of index, such as when index = 0, text([42+56]) need return 42; and when index = 1, it need return 56.\n    """\n    target_position = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: token, sop), rasp.Comparison.EQ), rasp.indices) # Match the position of target token\n    if index == 0: # If need to match the front position.\n        out = rasp.Aggregate(rasp.Select(rasp.indices, rasp.indices - (length - target_position), rasp.Comparison.EQ),\n                             sop) # Move the sop on the left side of the token to the far right.\n        return rasp.SequenceMap(lambda x, i: x if i == 2 else "_", out, rasp.categorical(\n            rasp.SequenceMap(lambda x, i: 2 if x >= i else 0, rasp.indices, length - target_position))) # Use "_" to fill the empty position on the left.\n    else: # If need to match the finally number.\n        return rasp.SequenceMap(lambda x, i: x if i else "_", sop,\n                                rasp.SequenceMap(lambda x, i: 1 if x > i else 0, rasp.indices, target_position)).named(\n            f"shift") # Use "_" to fill the empty position on the left.\n\n\n',
 'def turn_number(sop) -> rasp.SOp:\n    """Convert the token in the form of categorical to the form of numerical.\n    Example usage:\n      text = turn_number(tokens)\n      text(["0", "0", "0", "5", "6"])\n      >> [56, 56, 56, 56, 56]\n    Args:\n      sop: Sop to turn numerical.\n    """\n    indice = rasp.Map(lambda x: \'0\'*x ,length-rasp.indices-1) # Like [10000, 1000, 100, 10, 1]\n    sop = rasp.SequenceMap(lambda x,y: float(str(x)+y),sop,indice) # Value alignment of each bit. Like [0, 0, 0, 50, 6]\n    sop = rasp.SequenceMap(lambda x,y: x * y,sop,length) # Before aggregation, multiply the length (because the result of aggregation is average).\n    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,rasp.Comparison.TRUE),rasp.numerical(rasp.Map(lambda x: x, sop)),default=0)) # Add each bit.\n    return out\n\n\n',
 'def parity(sop) -> rasp.SOp:\n    """Predict whether a bit-string has an even or odd number of ones in it. For example, the parity of the bitstring[0, 1, 1, 0, 1] is "odd" (or 1) as opposed to "even" (or 0), because there is an odd number of 1s in the bit-string.\n    In other words, The first step is to multiply the length of each token, then add all the tokens and aggregate them. Finally, use round to convert to an int number and calculate whether the remainder of dividing it by 2 is odd or even.\n    Example usage:\n      text = parity(tokens)\n      text([1, 1, 0, 1, 0])\n      >> [1, 1, 1, 1, 1]\n      text([0, 1, 0, 1, 0])\n      >> [0, 0, 0, 0, 0]\n    Args:\n      sop: Sop to turn numerical.\n    """\n    sop = rasp.SequenceMap(lambda x,y: x * y,sop,length) # Multiply the length of each token.\n    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,rasp.Comparison.TRUE),rasp.numerical(rasp.Map(lambda x: x, sop)),default=0)) # Add each bit.\n    out = rasp.Map(lambda x: 0 if x % 2 == 0 else 1,out) # Calculate whether the remainder of dividing it by 2 is odd or even.\n    return out\n\n',
 'def reverse(sop) -> rasp.SOp:\n    """Reverse the order of a string.\n    Example usage:\n      text = reverse(tokens)\n      text(["h", "e", "l", "l", "o"])\n      >> ["o", "l", "l", "e", "h"]\n    Args:\n      sop: Sop to turn numerical.\n    """\n    return make_reverse(sop).named(\n        "reverse")  # Get the indices from back to front, and use opp_idx - 1, so that the first digit of indices = 0.\n',
 'def make_pair_balance(sop: rasp.SOp, open_token: str,\n                      close_token: str) -> rasp.SOp:\n    """Return fraction of previous open tokens minus the fraction of close tokens.\n     (As implemented in the RASP paper.)\n    If the outputs are always non-negative and end in 0, that implies the input\n    has balanced parentheses.\n    Example usage:\n      num_l = make_pair_balance(rasp.tokens, "(", ")")\n      num_l("a()b(c))")\n      >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]\n    Args:\n      sop: Input SOp.\n      open_token: Token that counts positive.\n      close_token: Token that counts negative.\n    Returns:\n      pair_balance: SOp mapping an input to a sequence, where every element\n        is the fraction of previous open tokens minus previous close tokens.\n    """\n    bools_open = rasp.numerical(sop == open_token).named("bools_open")\n    opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")\n\n    bools_close = rasp.numerical(sop == close_token).named("bools_close")\n    closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")\n\n    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))\n    return pair_balance.named("pair_balance")\n\n\n',
 'def make_shuffle_dyck(pairs: List[str]) -> rasp.SOp:\n    """Returns 1 if a set of parentheses are balanced, 0 else.\n     (As implemented in the RASP paper.)\n    Example usage:\n      shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])\n      shuffle_dyck2("({)}")\n      >> [1, 1, 1, 1]\n      shuffle_dyck2("(){)}")\n      >> [0, 0, 0, 0, 0]\n    Args:\n      pairs: List of pairs of open and close tokens that each should be balanced.\n    """\n    assert len(pairs) >= 1\n\n    # Compute running balance of each type of parenthesis\n    balances = []\n    for pair in pairs:\n        assert len(pair) == 2\n        open_token, close_token = pair\n        balance = make_pair_balance(\n            rasp.tokens, open_token=open_token,\n            close_token=close_token).named(f"balance_{pair}")\n        balances.append(balance)\n\n    # Check if balances where negative anywhere -> parentheses not balanced\n    any_negative = balances[0] < 0\n    for balance in balances[1:]:\n        any_negative = any_negative | (balance < 0)\n\n    # Convert to numerical SOp\n    any_negative = rasp.numerical(rasp.Map(lambda x: x,\n                                           any_negative)).named("any_negative")\n\n    select_all = rasp.Select(rasp.indices, rasp.indices,\n                             rasp.Comparison.TRUE).named("select_all")\n    has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,\n                                            default=0)).named("has_neg")\n\n    # Check if all balances are 0 at the end -> closed all parentheses\n    all_zero = balances[0] == 0\n    for balance in balances[1:]:\n        all_zero = all_zero & (balance == 0)\n\n    select_last = rasp.Select(rasp.indices, length - 1,\n                              rasp.Comparison.EQ).named("select_last")\n    last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")\n\n    not_has_neg = (~has_neg).named("not_has_neg")\n    return (last_zero & not_has_neg).named("shuffle_dyck")\n\n\n',
 'def make_shuffle_dyck2() -> rasp.SOp:\n    return make_shuffle_dyck(pairs=["()", "{}"]).named("shuffle_dyck2")\n\n\n',
 'def shift_by(offset: int, sop: rasp.SOp) -> rasp.SOp:\n    """Returns the sop, shifted by `offset`, None-padded."""\n    select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,\n                                       lambda k, q: q == k + offset)\n    out = rasp.Aggregate(select_off_by_offset, sop, default=None)\n    return out.named(f"shift_by({offset})")\n\n\n',
 'def detect_pattern(sop: rasp.SOp, pattern: Sequence[rasp.Value]) -> rasp.SOp:\n    """Returns an SOp which is True at the final element of the pattern.\n    The first len(pattern) - 1 elements of the output SOp are None-padded.\n    detect_pattern(tokens, "abc")("abcabc") == [None, None, T, F, F, T]\n    Args:\n      sop: the SOp in which to look for patterns.\n      pattern: a sequence of values to look for.\n    Returns:\n      a sop which detects the pattern.\n    """\n\n    if len(pattern) < 1:\n        raise ValueError(f"Length of `pattern` must be at least 1. Got {pattern}")\n\n    # detectors[i] will be a boolean-valued SOp which is true at position j iff\n    # the i\'th (from the end) element of the pattern was detected at position j-i.\n    detectors = []\n    for i, element in enumerate(reversed(pattern)):\n        detector = sop == element\n        if i != 0:\n            detector = shift_by(i, detector)\n        detectors.append(detector)\n\n    # All that\'s left is to take the AND over all detectors.\n    pattern_detected = detectors.pop()\n    while detectors:\n        pattern_detected = pattern_detected & detectors.pop()\n\n    return pattern_detected.named(f"detect_pattern({pattern})")\n\n\n',
 'def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:\n    """Returns vals sorted by < relation on keys.\n    Only supports unique keys.\n    Example usage:\n      sort = make_sort(rasp.tokens, rasp.tokens)\n      sort([2, 4, 3, 1])\n      >> [1, 2, 3, 4]\n    Args:\n      vals: Values to sort.\n      keys: Keys for sorting.\n    """\n    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named(\n        "smaller")  # From the Keys for sorting, determine how many tokens are smaller.\n    target_pos = rasp.SelectorWidth(smaller).named(\n        "target_pos")  # Calculate how many tokens are less than its total number. Use this as the target position.\n    sel_new = rasp.Select(target_pos, rasp.indices,\n                          rasp.Comparison.EQ)  # Use the target position to query the original indices.\n    return rasp.Aggregate(sel_new, vals).named(\n        "sort")  # Aggregate the sel_new and vals, so that output the sorted value\n\n\n',
 'def make_sort(vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int,\n              min_key: float) -> rasp.SOp:\n    """Returns vals sorted by < relation on keys, which don\'t need to be unique.\n    The implementation differs from the RASP paper, as it avoids using\n    compositions of selectors to break ties. Instead, it uses the arguments\n    max_seq_len and min_key to ensure the keys are unique.\n    Note that this approach only works for numerical keys.\n    Example usage:\n      sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)\n      sort([2, 4, 3, 1])\n      >> [1, 2, 3, 4]\n      sort([2, 4, 1, 2])\n      >> [1, 2, 2, 4]\n    Args:\n      vals: Values to sort.\n      keys: Keys for sorting.\n      max_seq_len: Maximum sequence length (used to ensure keys are unique)\n      min_key: Minimum key value (used to ensure keys are unique)\n    Returns:\n      Output SOp of sort program.\n    """\n    keys = rasp.SequenceMap(lambda x, i: x + min_key * i / max_seq_len, keys,\n                            rasp.indices)  # We can recursively represent functions with more than two inputs using SequenceMaps. In order to avoid the repetition of two tokens with the same value, we add additional location information for them.\n    return make_sort_unique(vals, keys)\n\n\n']

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
            self.code_prompt.append(code)
        except:
            if sudo_save:
                self.code_prompt.append(code)
            else:
                exec(code)

    def __call__(self, instruct,vocab=['a','b','c'], example=[], LLM_name='text-davich-003', generation_number=10,
                 temperature=0.6, max_length=256, stop='\n\n', auto_save=True, return_torch=True,model_name='custom_model'):
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






