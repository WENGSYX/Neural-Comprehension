def make_pair_balance(sop: rasp.SOp, open_token: str,
                      close_token: str) -> rasp.SOp:
    """Return fraction of previous open tokens minus the fraction of close tokens.
     (As implemented in the RASP paper.)
    If the outputs are always non-negative and end in 0, that implies the input
    has balanced parentheses.
    Example usage:
      num_l = make_pair_balance(rasp.tokens, "(", ")")
      num_l("a()b(c))")
      >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]
    Args:
      sop: Input SOp.
      open_token: Token that counts positive.
      close_token: Token that counts negative.
    Returns:
      pair_balance: SOp mapping an input to a sequence, where every element
        is the fraction of previous open tokens minus previous close tokens.
    """
    bools_open = rasp.numerical(sop == open_token).named("bools_open")
    opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")

    bools_close = rasp.numerical(sop == close_token).named("bools_close")
    closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance.named("pair_balance")


