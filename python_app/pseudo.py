def word_matches(a, b, max_start_diff=0.25, max_end_diff=0.35, min_iou=0.30):
    if a.word != b.word:
        return False

    if abs(a.start - b.start) > max_start_diff:
        return False

    if abs(a.end - b.end) > max_end_diff:
        return False

    if interval_iou(a.start, a.end, b.start, b.end) < min_iou:
        return False

    return True


def interval_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0

#  english normalization function
def normalize_word(w):
    return w.strip().lower().replace(" ", "")