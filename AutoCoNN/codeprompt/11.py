def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:
    """Count the fraction of previous tokens where a specific condition was True.
     (As implemented in the RASP paper.)
    Example usage:
      num_l = make_frac_prevs(rasp.tokens=="l")
      num_l("hello")
      >> [0, 0, 1/3, 1/2, 2/5]
    Args:
      bools: SOp mapping a sequence to a sequence of booleans.
    Returns:
      frac_prevs: SOp mapping an input to a sequence, where every element
        is the fraction of previous "True" tokens.
    """
    bools = rasp.numerical(bools)  # Turn to numerically-encoded.
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)  # Consider fraction of previous tokens.
    return rasp.numerical(rasp.Aggregate(prevs, bools,
                                         default=0)).named(
        "frac_prevs")  # Produces an sop that averages the value of the s-op weighted by the selection matrix


