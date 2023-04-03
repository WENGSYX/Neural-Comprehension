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
