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


