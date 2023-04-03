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


