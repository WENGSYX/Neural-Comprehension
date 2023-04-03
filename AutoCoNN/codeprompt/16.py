def atoi(sop):
    """Converts all text to number, and uses 0 for strings of types other than numbers, It may be mixed with 'str' or 'int'.
    Example usage:
      model = atoi(tokens)
      model(['1', '4', '-', '2', '3'])
      >> [1, 4, 0, 2, 3]
    """
    return rasp.SequenceMap(lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices).named(
        "atoi")  # Converts all text to number， and other token set 0.


