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

