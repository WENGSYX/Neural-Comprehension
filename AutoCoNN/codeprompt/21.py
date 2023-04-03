def turn_number(sop) -> rasp.SOp:
    """Convert the token in the form of categorical to the form of numerical.
    Example usage:
      text = turn_number(tokens)
      text(["0", "0", "0", "5", "6"])
      >> [56, 56, 56, 56, 56]
    Args:
      sop: Sop to turn numerical.
    """
    indice = rasp.Map(lambda x: '0'*x ,length-rasp.indices-1) # Like [10000, 1000, 100, 10, 1]
    sop = rasp.SequenceMap(lambda x,y: float(str(x)+y),sop,indice) # Value alignment of each bit. Like [0, 0, 0, 50, 6]
    sop = rasp.SequenceMap(lambda x,y: x * y,sop,length) # Before aggregation, multiply the length (because the result of aggregation is average).
    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,rasp.Comparison.TRUE),rasp.numerical(rasp.Map(lambda x: x, sop)),default=0)) # Add each bit.
    return out


