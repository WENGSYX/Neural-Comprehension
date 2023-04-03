"""RASP programs only using the subset of RASP supported by the compiler."""

from typing import List, Sequence

from tracr.rasp import rasp

""" 
Programs that work only under non-causal evaluation.

Comparison.EQ: lambda key, query: key == query,
Comparison.LT: lambda key, query: key < query,
Comparison.LEQ: lambda key, query: key <= query,
Comparison.GT: lambda key, query: key > query,
Comparison.GEQ: lambda key, query: key >= query,
Comparison.NEQ: lambda key, query: key != query,
Comparison.TRUE: lambda key, query: True,
Comparison.FALSE: lambda key, query: False,
"""


def make_length() -> rasp.SOp:
    """Creates the `length` SOp using selector width primitive.
    Example usage:
      length = make_length()
      length("abcdefg")
      >> [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
    Returns:
      length: SOp mapping an input to a sequence, where every element
        is the length of that sequence.
    """
    all_true_selector = rasp.Select(
        rasp.indices, rasp.tokens, rasp.Comparison.TRUE).named(
        "all_true_selector")  # Match tokens and tokens one by one, and calculate that they are completely equal.
    return rasp.SelectorWidth(all_true_selector).named(
        "length")  # which computes the number of elements in each row of a selector that with the value 1.

now_length = make_length()
def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    """Create an SOp that reverses a sequence, using length primitive.
    Example usage:
      reverse = make_reverse(rasp.tokens)
      reverse("Hello")
      >> ['o', 'l', 'l', 'e', 'H']
    Args:
      sop: an SOp
    Returns:
      reverse : SOp that reverses the input sequence.
    """
    opp_idx = (now_length - rasp.indices).named("opp_idx")  # Get the indices from back to front.
    opp_idx = (opp_idx - 1).named("opp_idx-1")  # opp_idx - 1, so that the first digit of indices = 0.
    reverse_selector = rasp.Select(rasp.indices, opp_idx,
                                   rasp.Comparison.EQ).named(
        "reverse_selector")  # Use opp_idx to query indices, get the Select.
    return rasp.Aggregate(reverse_selector, sop).named(
        "reverse")  # Aggregate the reverse_selector and sop, so that output the reverse sop


length = make_length()


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    """Create an SOp that reverses a sequence, using length primitive.
    Example usage:
      reverse = make_reverse(rasp.tokens)
      reverse("Hello")
      >> ['o', 'l', 'l', 'e', 'H']
    Args:
      sop: an SOp
    Returns:
      reverse : SOp that reverses the input sequence.
    """
    opp_idx = (length - rasp.indices).named("opp_idx")  # Get the indices from back to front.
    opp_idx = (opp_idx - 1).named("opp_idx-1")  # opp_idx - 1, so that the first digit of indices = 0.
    reverse_selector = rasp.Select(rasp.indices, opp_idx,
                                   rasp.Comparison.EQ).named(
        "reverse_selector")  # Use opp_idx to query indices, get the Select.
    return rasp.Aggregate(reverse_selector, sop).named(
        "reverse")  # Aggregate the reverse_selector and sop, so that output the reverse sop


def make_pair_balance(sop: rasp.SOp, open_token: str,
                      close_token: str) -> rasp.SOp:
    """Return fraction of previous open tokens minus the fraction of close tokens.
     (As implemented in the RASP paper.)
    If the outputs are always non-negative and end in 0, that implies the input
    has balanced parentheses.
    Example usage:
      num_l = make_pair_balance(rasp.tokens, "(", ")")
      num_l("a()b(c))")
      >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]
    Args:
      sop: Input SOp.
      open_token: Token that counts positive.
      close_token: Token that counts negative.
    Returns:
      pair_balance: SOp mapping an input to a sequence, where every element
        is the fraction of previous open tokens minus previous close tokens.
    """
    bools_open = rasp.numerical(sop == open_token).named("bools_open")
    opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")

    bools_close = rasp.numerical(sop == close_token).named("bools_close")
    closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance.named("pair_balance")


def make_shuffle_dyck(pairs: List[str]) -> rasp.SOp:
    """Returns 1 if a set of parentheses are balanced, 0 else.
     (As implemented in the RASP paper.)
    Example usage:
      shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])
      shuffle_dyck2("({)}")
      >> [1, 1, 1, 1]
      shuffle_dyck2("(){)}")
      >> [0, 0, 0, 0, 0]
    Args:
      pairs: List of pairs of open and close tokens that each should be balanced.
    """
    assert len(pairs) >= 1

    # Compute running balance of each type of parenthesis
    balances = []
    for pair in pairs:
        assert len(pair) == 2
        open_token, close_token = pair
        balance = make_pair_balance(
            rasp.tokens, open_token=open_token,
            close_token=close_token).named(f"balance_{pair}")
        balances.append(balance)

    # Check if balances where negative anywhere -> parentheses not balanced
    any_negative = balances[0] < 0
    for balance in balances[1:]:
        any_negative = any_negative | (balance < 0)

    # Convert to numerical SOp
    any_negative = rasp.numerical(rasp.Map(lambda x: x,
                                           any_negative)).named("any_negative")

    select_all = rasp.Select(rasp.indices, rasp.indices,
                             rasp.Comparison.TRUE).named("select_all")
    has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,
                                            default=0)).named("has_neg")

    # Check if all balances are 0 at the end -> closed all parentheses
    all_zero = balances[0] == 0
    for balance in balances[1:]:
        all_zero = all_zero & (balance == 0)

    select_last = rasp.Select(rasp.indices, length - 1,
                              rasp.Comparison.EQ).named("select_last")
    last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

    not_has_neg = (~has_neg).named("not_has_neg")
    return (last_zero & not_has_neg).named("shuffle_dyck")


def make_shuffle_dyck2() -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"]).named("shuffle_dyck2")


def shift_by(offset: int, sop: rasp.SOp) -> rasp.SOp:
    """Returns the sop, shifted by `offset`, None-padded."""
    select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                       lambda k, q: q == k + offset)
    out = rasp.Aggregate(select_off_by_offset, sop, default=None)
    return out.named(f"shift_by({offset})")


def detect_pattern(sop: rasp.SOp, pattern: Sequence[rasp.Value]) -> rasp.SOp:
    """Returns an SOp which is True at the final element of the pattern.
    The first len(pattern) - 1 elements of the output SOp are None-padded.
    detect_pattern(tokens, "abc")("abcabc") == [None, None, T, F, F, T]
    Args:
      sop: the SOp in which to look for patterns.
      pattern: a sequence of values to look for.
    Returns:
      a sop which detects the pattern.
    """

    if len(pattern) < 1:
        raise ValueError(f"Length of `pattern` must be at least 1. Got {pattern}")

    # detectors[i] will be a boolean-valued SOp which is true at position j iff
    # the i'th (from the end) element of the pattern was detected at position j-i.
    detectors = []
    for i, element in enumerate(reversed(pattern)):
        detector = sop == element
        if i != 0:
            detector = shift_by(i, detector)
        detectors.append(detector)

    # All that's left is to take the AND over all detectors.
    pattern_detected = detectors.pop()
    while detectors:
        pattern_detected = pattern_detected & detectors.pop()

    return pattern_detected.named(f"detect_pattern({pattern})")


def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    """Returns vals sorted by < relation on keys.
    Only supports unique keys.
    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
    Args:
      vals: Values to sort.
      keys: Keys for sorting.
    """
    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named(
        "smaller")  # From the Keys for sorting, determine how many tokens are smaller.
    target_pos = rasp.SelectorWidth(smaller).named(
        "target_pos")  # Calculate how many tokens are less than its total number. Use this as the target position.
    sel_new = rasp.Select(target_pos, rasp.indices,
                          rasp.Comparison.EQ)  # Use the target position to query the original indices.
    return rasp.Aggregate(sel_new, vals).named(
        "sort")  # Aggregate the sel_new and vals, so that output the sorted value


def make_sort(vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int,
              min_key: float) -> rasp.SOp:
    """Returns vals sorted by < relation on keys, which don't need to be unique.
    The implementation differs from the RASP paper, as it avoids using
    compositions of selectors to break ties. Instead, it uses the arguments
    max_seq_len and min_key to ensure the keys are unique.
    Note that this approach only works for numerical keys.
    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
      sort([2, 4, 1, 2])
      >> [1, 2, 2, 4]
    Args:
      vals: Values to sort.
      keys: Keys for sorting.
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
      min_key: Minimum key value (used to ensure keys are unique)
    Returns:
      Output SOp of sort program.
    """
    keys = rasp.SequenceMap(lambda x, i: x + min_key * i / max_seq_len, keys,
                            rasp.indices)  # We can recursively represent functions with more than two inputs using SequenceMaps. In order to avoid the repetition of two tokens with the same value, we add additional location information for them.
    return make_sort_unique(vals, keys)


def make_sort_freq(max_seq_len: int) -> rasp.SOp:
    """Returns tokens sorted by the frequency they appear in the input.
    Tokens the appear the same amount of times are output in the same order as in
    the input.
    Example usage:
      sort = make_sort_freq(rasp.tokens, rasp.tokens, 5)
      sort([2, 4, 2, 1])
      >> [2, 2, 4, 1]
    Args:
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
    """
    hist = -1 * make_hist().named("hist")
    return make_sort(
        rasp.tokens, hist, max_seq_len=max_seq_len, min_key=1).named("sort_freq")


### Programs that work under both causal and regular evaluation.


def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:
    """Count the fraction of previous tokens where a specific condition was True.
     (As implemented in the RASP paper.)
    Example usage:
      num_l = make_frac_prevs(rasp.tokens=="l")
      num_l("hello")
      >> [0, 0, 1/3, 1/2, 2/5]
    Args:
      bools: SOp mapping a sequence to a sequence of booleans.
    Returns:
      frac_prevs: SOp mapping an input to a sequence, where every element
        is the fraction of previous "True" tokens.
    """
    bools = rasp.numerical(bools)  # Turn to numerically-encoded.
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)  # Consider fraction of previous tokens.
    return rasp.numerical(rasp.Aggregate(prevs, bools,
                                         default=0)).named(
        "frac_prevs")  # Produces an sop that averages the value of the s-op weighted by the selection matrix


def make_count_less_freq(n: int) -> rasp.SOp:
    """Returns how many tokens appear fewer than n times in the input.
    The output sequence contains this count in each position.
    Example usage:
      count_less_freq = make_count_less_freq(2)
      count_less_freq(["a", "a", "a", "b", "b", "c"])
      >> [3, 3, 3, 3, 3, 3]
      count_less_freq(["a", "a", "c", "b", "b", "c"])
      >> [6, 6, 6, 6, 6, 6]
    Args:
      n: Integer to compare token frequences to.
    """
    hist = make_hist().named("hist")  # Returns the number of times each token occurs in the input.
    select_less = rasp.Select(hist, hist,
                              lambda x, y: x <= n).named(
        "select_less")  # Judge whether the number of each token is less than 'n'.
    return rasp.SelectorWidth(select_less).named(
        "count_less_freq")  # Add all the values which the number of token is less than 'n'.


def make_hist() -> rasp.SOp:
    """Returns the number of times each token occurs in the input.
     (As implemented in the RASP paper.)
    Example usage:
      hist = make_hist()
      hist("abac")
      >> [2, 1, 2, 1]
    """
    same_tok = rasp.Select(rasp.tokens, rasp.tokens,
                           rasp.Comparison.EQ).named(
        "same_tok")  # Match tokens and tokens one by one, and calculate that they are string equal.
    return rasp.SelectorWidth(same_tok).named(
        "hist")  # Count the equal number for each token if the same_tok value = 1.


def make_count(sop, token):
    """Returns the count of `token` in `sop`.
    The output sequence contains this count in each position.
    Example usage:
      count = make_count(tokens, "a")
      count(["a", "a", "a", "b", "b", "c"])
      >> [3, 3, 3, 3, 3, 3]
      count(["c", "a", "b", "c"])
      >> [1, 1, 1, 1]
    Args:
      sop: Sop to count tokens in.
      token: Token to count.
    """
    return rasp.SelectorWidth(rasp.Select(
        sop, sop, lambda k, q: k == token))\
        .named(f"count_{token}")  # First determine which token in the sop are 'token', and set the value =1, then add all the values while value = 1.


def add(sop, number):
    """Returns the value of sop added number
    Example usage:
      model = add(tokens, 2)
      model([1, 4, 7, 2, 3])
      >> [3, 6, 9, 4, 5]
    """
    return rasp.SequenceMap(lambda x, i: x + number, sop, rasp.indices).named(f"add_{number}")


def atoi(sop):
    """Converts all text to number, and uses 0 for strings of types other than numbers, It may be mixed with 'str' or 'int'.
    Example usage:
      model = atoi(tokens)
      model(['1', '4', '-', '2', '3'])
      >> [1, 4, 0, 2, 3]
    """
    return rasp.SequenceMap(lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices).named(
        "atoi")  # Converts all text to number， and other token set 0.


def Duplicate_detection(sop):
    """If the token in the input is repeated, return it; otherwise, return 0
      Example usage:
      model = Duplicate_detection(tokens)
      model(['h', 'e', 'l, 'l', 'o'])
      >> [0, 0, 'l', 'l', 0]
    """
    hist = make_hist()  # Count the number of times each token occurs in the input.
    return rasp.SequenceMap(lambda x, i: i if x > 1 else 0, hist, rasp.tokens).named(
        "Duplicate_detection")  # If the token in hist is repeated, return it; otherwise, return 0.


def ralign(sop, default='-'):
    c = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: '_', sop), rasp.Comparison.EQ), rasp.Map(lambda x: 1, sop))
    return rasp.Aggregate(rasp.Select(rasp.indices + c, rasp.indices, rasp.Comparison.EQ), sop, default='-')


def shift(sop, index):
    """Shift all of the tokens in a sequence to the right by index positions.
    Each token moves the index cell to the right, use '_' for the left moving part fill.
    Example usage:
      model = shift(sop, token="-")
      model(['h', 'e', 'l', 'l', 'o'])
      >> ['_', '_', 'h', 'e', 'l']
    Please note that meaningful tokens need to be placed on the far right.
    Args:
      sop: Sop to shift in.
      index: Number of right moves.
    """
    idx = (rasp.indices + index).named("idx-1")  # Get the target indices.
    selector = rasp.Select(idx, rasp.indices,
                           rasp.Comparison.EQ).named("shift_selector")  # Use opp_idx to query indices, get the Select.
    shift = rasp.Aggregate(selector, sop).named("shift")  # Aggregates the sops and selectors (converted from indexes).
    return rasp.SequenceMap(lambda x, i: x if i >= index else "_", shift, rasp.indices).named(f"shift_{index}")


def split(sop, token, index):
    """using 'token' as the separator string 'sop', return the 'index' th and then align right.
    Example usage:
      text = split(tokens, "+", 0)
      text([4, 2, "+", 5, 6])
      >> [0, 0, 0, 4, 2]
      text = split(tokens, "-", 1)
      text([8, 1, "-", 5, 7])
      >> [0, 0, 0, 5, 7]
    Args:
      sop: Sop to count tokens in.
      token: Token to count.
      index: After split, token of index, such as when index = 0, text([42+56]) need return 42; and when index = 1, it need return 56.
    """
    target_position = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: token, sop), rasp.Comparison.EQ), rasp.indices) # Match the position of target token
    if index == 0: # If need to match the front position.
        out = rasp.Aggregate(rasp.Select(rasp.indices, rasp.indices - (length - target_position), rasp.Comparison.EQ),
                             sop) # Move the sop on the left side of the token to the far right.
        return rasp.SequenceMap(lambda x, i: x if i == 2 else "_", out, rasp.categorical(
            rasp.SequenceMap(lambda x, i: 2 if x >= i else 0, rasp.indices, length - target_position))) # Use "_" to fill the empty position on the left.
    else: # If need to match the finally number.
        return rasp.SequenceMap(lambda x, i: x if i else "_", sop,
                                rasp.SequenceMap(lambda x, i: 1 if x > i else 0, rasp.indices, target_position)).named(
            f"shift") # Use "_" to fill the empty position on the left.


def turn_number(sop) -> rasp.SOp:
    """Convert the token in the form of categorical to the form of numerical.
    Example usage:
      text = turn_number(tokens)
      text(["0", "0", "0", "5", "6"])
      >> [56, 56, 56, 56, 56]
    Args:
      sop: Sop to turn numerical.
    """
    indice = rasp.Map(lambda x: '0'*x ,length-rasp.indices-1) # Like [10000, 1000, 100, 10, 1]
    sop = rasp.SequenceMap(lambda x,y: float(str(x)+y),sop,indice) # Value alignment of each bit. Like [0, 0, 0, 50, 6]
    sop = rasp.SequenceMap(lambda x,y: x * y,sop,length) # Before aggregation, multiply the length (because the result of aggregation is average).
    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,rasp.Comparison.TRUE),rasp.numerical(rasp.Map(lambda x: x, sop)),default=0)) # Add each bit.
    return out


def parity(sop) -> rasp.SOp:
    """Predict whether a bit-string has an even or odd number of ones in it. For example, the parity of the bitstring[0, 1, 1, 0, 1] is "odd" (or 1) as opposed to "even" (or 0), because there is an odd number of 1s in the bit-string.
    In other words, The first step is to multiply the length of each token, then add all the tokens and aggregate them. Finally, use round to convert to an int number and calculate whether the remainder of dividing it by 2 is odd or even.
    Example usage:
      text = parity(tokens)
      text([1, 1, 0, 1, 0])
      >> [1, 1, 1, 1, 1]
      text([0, 1, 0, 1, 0])
      >> [0, 0, 0, 0, 0]
    Args:
      sop: Sop to turn numerical.
    """
    sop = rasp.SequenceMap(lambda x,y: x * y,sop,length) # Multiply the length of each token.
    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,rasp.Comparison.TRUE),rasp.numerical(rasp.Map(lambda x: x, sop)),default=0)) # Add each bit.
    out = rasp.Map(lambda x: 0 if x % 2 == 0 else 1,out) # Calculate whether the remainder of dividing it by 2 is odd or even.
    return out

def reverse(sop) -> rasp.SOp:
    """Reverse the order of a string.
    Example usage:
      text = reverse(tokens)
      text(["h", "e", "l", "l", "o"])
      >> ["o", "l", "l", "e", "h"]
    Args:
      sop: Sop to turn numerical.
    """
    return make_reverse(sop).named(
        "reverse")  # Get the indices from back to front, and use opp_idx - 1, so that the first digit of indices = 0.
