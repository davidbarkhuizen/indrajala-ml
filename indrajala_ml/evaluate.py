from random import uniform

from indrajala_ml.geometry import positive_region_bounding_box
from indrajala_ml.model.classifier_protocols import StateClassifier, TargetClassifier


def sample_class_balanced_states(
    classifier: TargetClassifier[float], count: int, max_attempts: int = 20_000
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    """
    (positive_states, negative_states): count states each that classifier puts in that class.

    Positive states are drawn from a tight box around the positive region when one is computable
    (geometry.positive_region_bounding_box): the region is often under 1% of input_bounds at
    cardinality 4 (as low as 0.02%), where rejection sampling over the whole box may never succeed
    within max_attempts. Otherwise (dimension != 2, not AND, unbounded) from input_bounds. Negative
    states are drawn from input_bounds.
    """

    positive_bounds = positive_region_bounding_box(classifier) or classifier.input_bounds

    def sample(category: float, bounds: list[tuple[float, float]]) -> list[tuple[float, ...]]:
        collected: list[tuple[float, ...]] = []
        attempts = 0
        while len(collected) < count:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"failed to sample {count} examples of class {category} within "
                    f"{max_attempts} attempts - the classifier's decision boundary likely "
                    "doesn't cross its input bounds, making this class unreachable"
                )
            attempts += 1
            state = tuple(uniform(*bound) for bound in bounds)
            if classifier.classify_state(state) == category:
                collected.append(state)
        return collected

    return sample(1.0, positive_bounds), sample(0.0, classifier.input_bounds)


def compare_on_random_point(
    reference: TargetClassifier[float], student: StateClassifier[float]
) -> tuple[tuple[float, ...], float, float]:

    state = tuple(uniform(*bounds) for bounds in reference.input_bounds)
    return state, reference.classify_state(state), student.classify_state(state)


def agreement_label(reference_category: float, student_category: float) -> str:
    """
    "agree" or "disagree", for demos reporting compare_on_random_point.
    """
    return "agree" if reference_category == student_category else "disagree"


def smoothed_series(values: list[float], window: int = 31) -> list[float]:

    # trailing moving average - only ever looks backward, so it stays a fair comparison
    # against the raw series at every point (no look-ahead)
    smoothed = []
    for i in range(len(values)):
        segment = values[max(0, i - window + 1) : i + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def class_balanced_disagreement_rate(
    reference: TargetClassifier[float],
    student: StateClassifier[object],
    per_class_sample_count: int = 10,
    max_attempts: int = 20_000,
) -> float:

    assert per_class_sample_count >= 1, f"per_class_sample_count must be at least 1; got {per_class_sample_count}"

    # an equal number of each class: uniform sampling would weight disagreement by each class's
    # area, and the positive class shrinks sharply as cardinality grows
    positive_states, negative_states = sample_class_balanced_states(reference, per_class_sample_count, max_attempts)

    positive_disagreements = sum(1 for state in positive_states if student.classify_state(state) != 1.0)
    negative_disagreements = sum(1 for state in negative_states if student.classify_state(state) != 0.0)

    return (positive_disagreements + negative_disagreements) / (2 * per_class_sample_count)
