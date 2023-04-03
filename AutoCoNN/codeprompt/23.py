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
