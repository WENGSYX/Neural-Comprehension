def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    """Returns vals sorted by < relation on keys.
    Only supports unique keys.
    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
    Args:
      vals: Values to sort.
      keys: Keys for sorting.
    """
    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named(
        "smaller")  # From the Keys for sorting, determine how many tokens are smaller.
    target_pos = rasp.SelectorWidth(smaller).named(
        "target_pos")  # Calculate how many tokens are less than its total number. Use this as the target position.
    sel_new = rasp.Select(target_pos, rasp.indices,
                          rasp.Comparison.EQ)  # Use the target position to query the original indices.
    return rasp.Aggregate(sel_new, vals).named(
        "sort")  # Aggregate the sel_new and vals, so that output the sorted value


