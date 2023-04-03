def make_shuffle_dyck(pairs: List[str]) -> rasp.SOp:
    """Returns 1 if a set of parentheses are balanced, 0 else.
     (As implemented in the RASP paper.)
    Example usage:
      shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])
      shuffle_dyck2("({)}")
      >> [1, 1, 1, 1]
      shuffle_dyck2("(){)}")
      >> [0, 0, 0, 0, 0]
    Args:
      pairs: List of pairs of open and close tokens that each should be balanced.
    """
    assert len(pairs) >= 1

    # Compute running balance of each type of parenthesis
    balances = []
    for pair in pairs:
        assert len(pair) == 2
        open_token, close_token = pair
        balance = make_pair_balance(
            rasp.tokens, open_token=open_token,
            close_token=close_token).named(f"balance_{pair}")
        balances.append(balance)

    # Check if balances where negative anywhere -> parentheses not balanced
    any_negative = balances[0] < 0
    for balance in balances[1:]:
        any_negative = any_negative | (balance < 0)

    # Convert to numerical SOp
    any_negative = rasp.numerical(rasp.Map(lambda x: x,
                                           any_negative)).named("any_negative")

    select_all = rasp.Select(rasp.indices, rasp.indices,
                             rasp.Comparison.TRUE).named("select_all")
    has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,
                                            default=0)).named("has_neg")

    # Check if all balances are 0 at the end -> closed all parentheses
    all_zero = balances[0] == 0
    for balance in balances[1:]:
        all_zero = all_zero & (balance == 0)

    select_last = rasp.Select(rasp.indices, length - 1,
                              rasp.Comparison.EQ).named("select_last")
    last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

    not_has_neg = (~has_neg).named("not_has_neg")
    return (last_zero & not_has_neg).named("shuffle_dyck")


