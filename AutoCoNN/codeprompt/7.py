def detect_pattern(sop: rasp.SOp, pattern: Sequence[rasp.Value]) -> rasp.SOp:
    """Returns an SOp which is True at the final element of the pattern.
    The first len(pattern) - 1 elements of the output SOp are None-padded.
    detect_pattern(tokens, "abc")("abcabc") == [None, None, T, F, F, T]
    Args:
      sop: the SOp in which to look for patterns.
      pattern: a sequence of values to look for.
    Returns:
      a sop which detects the pattern.
    """

    if len(pattern) < 1:
        raise ValueError(f"Length of `pattern` must be at least 1. Got {pattern}")

    # detectors[i] will be a boolean-valued SOp which is true at position j iff
    # the i'th (from the end) element of the pattern was detected at position j-i.
    detectors = []
    for i, element in enumerate(reversed(pattern)):
        detector = sop == element
        if i != 0:
            detector = shift_by(i, detector)
        detectors.append(detector)

    # All that's left is to take the AND over all detectors.
    pattern_detected = detectors.pop()
    while detectors:
        pattern_detected = pattern_detected & detectors.pop()

    return pattern_detected.named(f"detect_pattern({pattern})")


