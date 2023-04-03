def shift(sop, index):
    """Shift all of the tokens in a sequence to the right by index positions.
    Each token moves the index cell to the right, use '_' for the left moving part fill.
    Example usage:
      model = shift(sop, token="-")
      model(['h', 'e', 'l', 'l', 'o'])
      >> ['_', '_', 'h', 'e', 'l']
    Please note that meaningful tokens need to be placed on the far right.
    Args:
      sop: Sop to shift in.
      index: Number of right moves.
    """
    idx = (rasp.indices + index).named("idx-1")  # Get the target indices.
    selector = rasp.Select(idx, rasp.indices,
                           rasp.Comparison.EQ).named("shift_selector")  # Use opp_idx to query indices, get the Select.
    shift = rasp.Aggregate(selector, sop).named("shift")  # Aggregates the sops and selectors (converted from indexes).
    return rasp.SequenceMap(lambda x, i: x if i >= index else "_", shift, rasp.indices).named(f"shift_{index}")


