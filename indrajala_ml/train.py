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

    # see evaluate.sample_class_balanced_states - draws positive-class states from a tight
    # box around classifier's own positive region when one is computable, rather than
    # rejection-sampling classifier.input_bounds in full, since that region can be a tiny,
    # near-unreachable fraction of input_bounds at higher cardinality
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

    # higher cardinality shrinks the reference's positive region (intersection of more
    # half-planes), so some random reference classifiers make one class unreachable within
    # these bounds - regenerate the reference rather than failing on one unlucky draw.
    # is_valid, when given, is checked before the (more expensive) reachability sampling
    # below, so a caller can reject a candidate on cheap criteria (e.g. "region must be
    # bounded") without paying for training-data generation on a rejected candidate
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
    # the array networks train from one backend matrix (candidate 1 in docs/optimizations.md),
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
        # classify_rows batches the forward passes (candidate 2 in docs/optimizations.md)
        predictions = student.classify_rows(prepared)
        correct = sum(1 for predicted, category in zip(predictions, prepared.labels) if predicted == category)
        return correct / len(prepared)
    correct = sum(1 for state, category in training_data if student.classify_state(state) == category)
    return correct / len(training_data)


class TrainingDiagnostic:
    """
    Summarizes how a train_linear_classifier_network() run's training-data accuracy
    trajectory behaved, since there's no guarantee it converges and eyeballing a chart is
    otherwise the only way to tell converged from plateaued from still improving.
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
        # the same "converged"/"plateaued"/"still improving" tri-state label every demo prints
        # after training - one place so the wording (and the tri-state logic itself) can't drift
        # between demos
        return "converged" if self.converged else "plateaued" if self.plateaued else "still improving"


class ConvergenceSeries(list):
    """
    Exactly the list of (iteration, disagreement_rate) pairs train_linear_classifier_network
    has always returned - every existing use (indexing, iterating, len(), list
    comprehensions) keeps working unchanged - plus a `.diagnostic` (a TrainingDiagnostic) for
    callers that want to know whether training converged, plateaued, or was still improving.
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

    training_data is a list of (state, category) tuples, or a PreparedDataset for an array
    network student. An array network trains from a PreparedDataset either way: one is prepared
    from the tuples once per run when not passed in (see _prepared_for).

    learning_rate is either a plain float (every existing caller) or a schedule function from
    the current iteration index to a rate (e.g. lr_schedule.linear_warmup) - resolved once per
    learn() call, using the same iterations counter this function already tracks for its
    convergence series. Iterations (not epoch_index) is the right step index: a schedule targets
    per-step instability, and batch_size already determines how many steps an epoch contains.

    There's no guarantee this converges - training accuracy can
    oscillate rather than settle, especially once the target isn't exactly representable at
    student's cardinality/required_active. So rather than leaving student wherever the last
    epoch happened to land, it's left at whichever epoch's end had the best training-data
    accuracy seen (a pocket-algorithm-style "keep the best, not the latest" snapshot) - a
    strict improvement when training does converge (the best epoch is then the last one, so
    this is a no-op), and a real difference when it doesn't. See the returned
    ConvergenceSeries's .diagnostic for whether this run converged, plateaued, or was still
    improving.

    The returned series itself holds the disagreement-rate series against
    reference_classifier, sampled before training and after every single learning step, when
    reference_classifier is given (used for plotting a convergence curve) - this reflects the
    raw, unrolled-back trajectory actually taken during training, not the final pocketed
    student.
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
    # a final undersized batch (len(data) doesn't evenly divide batch_size) is kept, not
    # dropped - learn_batch already averages by its own len(batch), so no training data goes
    # unused just because it didn't land on an exact batch boundary
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
    The mini-batch-shaped sibling of train_linear_classifier_network, for gradient-based
    students only: calls student.learn_batch, which LinearClassifierNetwork/AssociationNode's
    discrete minimum-disturbance update rule has no equivalent of, hence the separate function
    rather than a branch inside that one. Like that function, actually duck-typed
    across every learn_batch-supporting sibling (MultiClassBackpropClassifierNetwork included,
    not just BackpropClassifierNetwork itself), despite the type hint naming only the most
    common case - the same looseness train_linear_classifier_network's own hint already has.

    learning_rate accepts the same float-or-schedule-function widening as
    train_linear_classifier_network above, resolved once per learn_batch call against
    iterations - here counting batches, not examples (see this function's own "iterations"
    note below), the per-step index a schedule like linear_warmup ramps against.

    Reshuffles training_data at the start of every epoch by default (unlike
    train_linear_classifier_network's fixed per-example order across epochs) - standard
    mini-batch SGD practice, so the same batch composition doesn't recur identically every
    epoch; pass reshuffle_each_epoch=False for a fixed batch composition instead. The final
    batch of an epoch is kept even when batch_size doesn't evenly divide len(training_data),
    per _chunk_into_batches above.

    training_data takes a PreparedDataset too, as train_linear_classifier_network's does.

    Otherwise mirrors train_linear_classifier_network exactly: the same "keep the best epoch,
    not the latest" pocket snapshot (see that function's own docstring), and the same
    TrainingDiagnostic/ConvergenceSeries contract - "iterations" here counts batches (one
    learn_batch call each), the batch-shaped analogue of one learn() call there.
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
