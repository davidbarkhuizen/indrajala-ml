from collections.abc import Sequence


def argmax_first_occurrence(values: Sequence[float]) -> int:
    """
    The index of the largest value, ties to the first, as np.argmax: the pure-Python networks'
    classify_state.
    """
    return max(range(len(values)), key=lambda i: values[i])
