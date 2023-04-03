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


