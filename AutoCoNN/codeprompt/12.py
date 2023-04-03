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


