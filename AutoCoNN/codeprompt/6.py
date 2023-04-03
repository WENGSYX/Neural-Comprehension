def shift_by(offset: int, sop: rasp.SOp) -> rasp.SOp:
    """Returns the sop, shifted by `offset`, None-padded."""
    select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                       lambda k, q: q == k + offset)
    out = rasp.Aggregate(select_off_by_offset, sop, default=None)
    return out.named(f"shift_by({offset})")


