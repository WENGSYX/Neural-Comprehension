def add(sop, number):
    """Returns the value of sop added number
    Example usage:
      model = add(tokens, 2)
      model([1, 4, 7, 2, 3])
      >> [3, 6, 9, 4, 5]
    """
    return rasp.SequenceMap(lambda x, i: x + number, sop, rasp.indices).named(f"add_{number}")


