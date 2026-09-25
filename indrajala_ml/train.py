from random import shuffle
from typing import Callable

from indrajala_ml.evaluate import class_balanced_disagreement_rate, sample_class_balanced_states
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.prepared_dataset import PreparedDataset


def random_alternating_training_data(
    size: int, classifier: LinearClassifierNetwork, max_attempts: int = 100_000
) -> list[tuple[tuple[float, ...], float]]:

    k: int = size // 2

    # positive states come from a tight box around the positive region when computable (see
    # evaluate.sample_class_balanced_states)
    positive_states, negative_states = sample_class_balanced_states(classifier, k, max_attempts)

    mixed = [(state, 1.0) for state in positive_states] + [(state, 0.0) for state in negative_states]
    shuffle(mixed)
    return mixed


def reachable_reference_and_training_data(
    cardinality: int,
    dimension: int,
    bounds: list[tuple[float, float]],
    training_set_size: int,
    regeneration_attempts: int = 20,
    max_attempts: int = 20_000,
    is_valid: Callable[[LinearClassifierNetwork], bool] | None = None,
) -> tuple[LinearClassifierNetwork, list[tuple[tuple[float, ...], float]]]:

    # higher cardinality shrinks the positive region, so a random reference can make one class
    # unreachable: draw again. is_valid (cheap, e.g. "region must be bounded") runs before the
    # reachability sampling
    for _ in range(regeneration_attempts):
        reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)
        if is_valid is not None and not is_valid(reference):
            continue
        try:
            return reference, random_alternating_training_data(training_set_size, reference, max_attempts=max_attempts)
        except RuntimeError:
            continue

    raise RuntimeError(f"no workable cardinality={cardinality} reference classifier found within these bounds")


def _prepared_for(
    student, training_data: list[tuple[tuple[float, ...], float]] | PreparedDataset
) -> PreparedDataset | None:
    # the array networks train from one backend matrix (docs/optimizations/implemented.md),
    # prepared here once per run unless the caller already built one; every other student (the
    # pure-Python networks, the linear classifiers) keeps the tuple list
    if isinstance(training_data, PreparedDataset):
        assert hasattr(student, "learn_row"), f"{type(student).__name__} can't train from a PreparedDataset"
        return training_data
    if hasattr(student, "prepare_dataset"):
        return student.prepare_dataset(training_data)
    return None


def _training_accuracy(
    student: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, ...], float]],
    prepared: PreparedDataset | None = None,
) -> float:
    if prepared is not None:
        # classify_rows batches the forward passes (docs/optimizations/implemented.md)
        predictions = student.classify_rows(prepared)
        correct = sum(1 for predicted, category in zip(predictions, prepared.labels) if predicted == category)
        return correct / len(prepared)
    correct = sum(1 for state, category in training_data if student.classify_state(state) == category)
    return correct / len(training_data)


class TrainingDiagnostic:
    """
    Whether a training run's training-accuracy trajectory converged, plateaued or was still
    improving; nothing guarantees convergence.
    """

    def __init__(
        self, epoch_training_accuracies: list[float], best_epoch_index: int, best_training_accuracy: float
    ) -> None:
        self.epoch_training_accuracies = epoch_training_accuracies
        # best_epoch_index is -1 (rather than an index into epoch_training_accuracies) when
        # no epoch ever beat the untrained starting point's own accuracy
        self.best_epoch_index = best_epoch_index
        self.best_training_accuracy = best_training_accuracy

    @property
    def converged(self) -> bool:
        # every training example correctly classified
        return self.best_training_accuracy >= 1.0

    @property
    def plateaued(self) -> bool:
        # the best epoch wasn't the last one - later epochs never improved on it
        return not self.converged and self.best_epoch_index < len(self.epoch_training_accuracies) - 1

    @property
    def still_improving(self) -> bool:
        # the last epoch was still the best one seen, but training hasn't converged yet -
        # more epochs might help
        return not self.converged and not self.plateaued

    @property
    def status_label(self) -> str:
        # the label every demo prints after training
        return "converged" if self.converged else "plateaued" if self.plateaued else "still improving"


class ConvergenceSeries(list):
    """
    The list of (iteration, disagreement_rate) pairs train_linear_classifier_network returns, plus
    .diagnostic, a TrainingDiagnostic.
    """

    diagnostic: TrainingDiagnostic


def train_linear_classifier_network(
    student: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, ...], float]] | PreparedDataset,
    learning_rate: float | Callable[[int], float] = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork | None = None,
) -> ConvergenceSeries:
    """
    Trains student in place over training_data for the given number of epochs.

    training_data is a list of (state, category) tuples, or a PreparedDataset for an array network,
    which is prepared from the tuples once per run when not passed in (see _prepared_for).

    learning_rate is a float or a schedule from the iteration index to a rate (e.g.
    lr_schedule.linear_warmup), read once per learn() call. Iterations, not epochs, since a schedule
    targets per-step instability.

    Training accuracy can oscillate rather than settle, especially when the target isn't
    representable at student's cardinality/required_active, so student is left at the epoch end
    with the best training accuracy (a pocket snapshot), which is the last epoch when training
    converges. The returned ConvergenceSeries's .diagnostic says whether it converged, plateaued or
    was still improving.

    With reference_classifier, the series holds the disagreement rate against it before training and
    after every step, for plotting: the trajectory actually taken, not the pocketed student.
    """

    assert len(training_data) >= 1, "training_data must not be empty"

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

    prepared = _prepared_for(student, training_data)

    best_snapshot = student.snapshot()
    best_training_accuracy = _training_accuracy(student, training_data, prepared)
    best_epoch_index = -1  # -1: the untrained starting point was never beaten
    epoch_training_accuracies: list[float] = []

    # an example is a row index on the prepared path and a (state, category) tuple otherwise
    if prepared is not None:
        examples = range(len(prepared))
        learn_example = lambda lr, index: student.learn_row(lr, prepared, index)
    else:
        examples = training_data
        learn_example = lambda lr, datum: student.learn(lr, *datum)

    for epoch_index in range(epochs):
        for example in examples:
            current_lr = learning_rate(iterations) if callable(learning_rate) else learning_rate
            learn_example(current_lr, example)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

        training_accuracy = _training_accuracy(student, training_data, prepared)
        epoch_training_accuracies.append(training_accuracy)
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_epoch_index = epoch_index
            best_snapshot = student.snapshot()

    student.restore(best_snapshot)

    result = ConvergenceSeries(convergence)
    result.diagnostic = TrainingDiagnostic(epoch_training_accuracies, best_epoch_index, best_training_accuracy)
    return result


def _chunk_into_batches(data: list, batch_size: int) -> list[list]:
    # a final short batch is kept: learn_batch averages by len(batch)
    assert batch_size >= 1, f"batch_size must be at least 1; got {batch_size}"
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def train_backprop_network_mini_batch(
    student: BackpropClassifierNetwork,
    training_data: list[tuple[tuple[float, ...], float]] | PreparedDataset,
    batch_size: int,
    learning_rate: float | Callable[[int], float] = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork | None = None,
    reshuffle_each_epoch: bool = True,
) -> ConvergenceSeries:
    """
    train_linear_classifier_network with mini-batches, for gradient-based students (any network with
    learn_batch; the linear classifier's minimum-disturbance rule has none).

    learning_rate is a float or a schedule, read once per learn_batch against iterations, which here
    count batches.

    Reshuffles training_data every epoch by default (reshuffle_each_epoch=False keeps the batches
    fixed). A final short batch is kept. training_data may be a PreparedDataset.

    Otherwise as train_linear_classifier_network: the pocket snapshot of the best epoch, and the
    TrainingDiagnostic/ConvergenceSeries return.
    """

    assert len(training_data) >= 1, "training_data must not be empty"

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

    prepared = _prepared_for(student, training_data)

    best_snapshot = student.snapshot()
    best_training_accuracy = _training_accuracy(student, training_data, prepared)
    best_epoch_index = -1  # -1: the untrained starting point was never beaten
    epoch_training_accuracies: list[float] = []

    # the prepared path shuffles row indices in place of the tuples: shuffle draws depend only
    # on the list's length, so a seed gives the same permutation, and the same batches, either way
    if prepared is not None:
        examples = list(range(len(prepared)))
        learn_batch = lambda lr, indices: student.learn_batch_rows(lr, prepared, indices)
    else:
        examples = training_data
        learn_batch = student.learn_batch

    for epoch_index in range(epochs):
        epoch_data = list(examples)
        if reshuffle_each_epoch:
            shuffle(epoch_data)

        for batch in _chunk_into_batches(epoch_data, batch_size):
            current_lr = learning_rate(iterations) if callable(learning_rate) else learning_rate
            learn_batch(current_lr, batch)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

        training_accuracy = _training_accuracy(student, training_data, prepared)
        epoch_training_accuracies.append(training_accuracy)
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_epoch_index = epoch_index
            best_snapshot = student.snapshot()

    student.restore(best_snapshot)

    result = ConvergenceSeries(convergence)
    result.diagnostic = TrainingDiagnostic(epoch_training_accuracies, best_epoch_index, best_training_accuracy)
    return result
