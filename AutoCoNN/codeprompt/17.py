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


