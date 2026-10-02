from collections.abc import Callable
from random import Random

from indrajala_ml.model.networks.python.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.protocols.classifier_protocols import TargetClassifier
from indrajala_ml.pcg64 import Pcg64Generator, default_rng
from indrajala_ml.training.evaluate import sample_class_balanced_states


def random_alternating_training_data(
    size: int, classifier: TargetClassifier[float], max_attempts: int = 100_000, rng: Random | None = None
) -> list[tuple[tuple[float, ...], float]]:

    k: int = size // 2

    # positive states come from a tight box around the positive region when computable (see
    # evaluate.sample_class_balanced_states). The states and the shuffle draw from rng, OS
    # entropy if None (the RNG generators workplan, D6, D9)
    rng = Random() if rng is None else rng
    positive_states, negative_states = sample_class_balanced_states(classifier, k, max_attempts, rng)

    mixed = [(state, 1.0) for state in positive_states] + [(state, 0.0) for state in negative_states]
    rng.shuffle(mixed)
    return mixed


def reachable_reference_and_training_data(
    cardinality: int,
    dimension: int,
    bounds: list[tuple[float, float]],
    training_set_size: int,
    regeneration_attempts: int = 20,
    max_attempts: int = 20_000,
    is_valid: Callable[[LinearClassifierNetwork], bool] | None = None,
    rng: Pcg64Generator | None = None,
    data_rng: Random | None = None,
) -> tuple[LinearClassifierNetwork, list[tuple[tuple[float, ...], float]]]:

    # higher cardinality shrinks the positive region, so a random reference can make one class
    # unreachable: draw again. is_valid (cheap, e.g. "region must be bounded") runs before the
    # reachability sampling. Every reference draws from rng, the training data from data_rng,
    # each OS entropy if None
    rng = default_rng() if rng is None else rng
    data_rng = Random() if data_rng is None else data_rng
    for _ in range(regeneration_attempts):
        reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, rng=rng)
        if is_valid is not None and not is_valid(reference):
            continue
        try:
            return reference, random_alternating_training_data(
                training_set_size, reference, max_attempts=max_attempts, rng=data_rng
            )
        except RuntimeError:
            continue

    raise RuntimeError(f"no workable cardinality={cardinality} reference classifier found within these bounds")
