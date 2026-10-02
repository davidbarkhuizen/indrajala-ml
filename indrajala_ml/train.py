from collections.abc import Callable, Sequence
from random import Random

from indrajala_ml.evaluate import class_balanced_disagreement_rate, sample_class_balanced_states
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.protocols.classifier_protocols import (
    BatchTrainableClassifier,
    Example,
    PreparedTrainableClassifier,
    StateClassifier,
    TargetClassifier,
    TrainableClassifier,
)
from indrajala_ml.model.specs.single_example import refuse_single_example_groups
from indrajala_ml.pcg64 import Pcg64Generator, default_rng
from indrajala_ml.prepared_dataset import PreparedDataset
from indrajala_ml.run_checkpoint import RunCheckpoint


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


def _prepared_for[L](
    student: TrainableClassifier[L], training_data: Sequence[Example[L]] | PreparedDataset
) -> PreparedDataset | None:
    # the array networks train from one backend matrix,
    # prepared here once per run unless the caller already built one; every other student (the
    # pure-Python networks, the linear classifiers) keeps the tuple list
    if isinstance(training_data, PreparedDataset):
        assert isinstance(student, PreparedTrainableClassifier), (
            f"{type(student).__name__} can't train from a PreparedDataset"
        )
        return training_data
    if isinstance(student, PreparedTrainableClassifier):
        return student.prepare_dataset(training_data)
    return None


def _training_accuracy[L](
    student: StateClassifier[L],
    training_data: Sequence[Example[L]] | PreparedDataset,
    prepared: PreparedDataset | None = None,
) -> float:
    if prepared is not None:
        # classify_rows batches the forward passes; only an
        # array network has a prepared dataset (_prepared_for)
        assert isinstance(student, PreparedTrainableClassifier)
        predictions = student.classify_rows(prepared)
        correct = sum(1 for predicted, category in zip(predictions, prepared.labels) if predicted == category)
        return correct / len(prepared)
    assert not isinstance(training_data, PreparedDataset)  # a PreparedDataset always has prepared
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


class ConvergenceSeries(list[tuple[int, float]]):
    """
    The list of (iteration, disagreement_rate) pairs train_linear_classifier_network returns, plus
    .diagnostic, a TrainingDiagnostic. train_backprop_network_mini_batch's also has .run_checkpoint,
    the run's state at its last epoch, to resume it from (run_checkpoint.py); the linear trainer's
    is None.
    """

    diagnostic: TrainingDiagnostic
    run_checkpoint: RunCheckpoint | None = None


def train_linear_classifier_network[L](
    student: TrainableClassifier[L],
    training_data: Sequence[Example[L]] | PreparedDataset,
    learning_rate: float | Callable[[int], float] = 0.25,
    epochs: int = 1,
    reference_classifier: TargetClassifier[float] | None = None,
    rng: Random | None = None,
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
    with the best training accuracy (a pocket checkpoint: the weights and the optimizer's state,
    model/persistence/checkpoint.py, so training on from there resumes from that epoch), which is the last
    epoch when training converges. The returned ConvergenceSeries's .diagnostic says whether it converged, plateaued or
    was still improving.

    With reference_classifier, the series holds the disagreement rate against it before training and
    after every step, for plotting: the trajectory actually taken, not the pocketed student. Its
    samples draw from rng, seeded from OS entropy if None (the RNG generators workplan, D6, D9).
    """

    assert len(training_data) >= 1, "training_data must not be empty"
    rng = Random() if rng is None else rng

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student, rng=rng)))

    prepared = _prepared_for(student, training_data)

    best_checkpoint = student.checkpoint()
    best_training_accuracy = _training_accuracy(student, training_data, prepared)
    best_epoch_index = -1  # -1: the untrained starting point was never beaten
    epoch_training_accuracies: list[float] = []

    def record_disagreement() -> None:
        if reference_classifier:
            convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student, rng=rng)))

    # one pass over the examples: row indices on the prepared path, (state, category) tuples
    # otherwise; each step's rate is read against the iterations before it
    def learn_epoch() -> None:
        nonlocal iterations
        if prepared is not None:
            assert isinstance(student, PreparedTrainableClassifier)  # _prepared_for's contract
            for index in range(len(prepared)):
                student.learn_row(_rate(learning_rate, iterations), prepared, index)
                iterations += 1
                record_disagreement()
        else:
            assert not isinstance(training_data, PreparedDataset)  # a PreparedDataset always has prepared
            for state, category in training_data:
                student.learn(_rate(learning_rate, iterations), state, category)
                iterations += 1
                record_disagreement()

    for epoch_index in range(epochs):
        learn_epoch()

        training_accuracy = _training_accuracy(student, training_data, prepared)
        epoch_training_accuracies.append(training_accuracy)
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_epoch_index = epoch_index
            best_checkpoint = student.checkpoint()

    student.restore_checkpoint(best_checkpoint)

    result = ConvergenceSeries(convergence)
    result.diagnostic = TrainingDiagnostic(epoch_training_accuracies, best_epoch_index, best_training_accuracy)
    return result


def _rate(learning_rate: float | Callable[[int], float], iterations: int) -> float:
    return learning_rate(iterations) if callable(learning_rate) else learning_rate


def _chunk_into_batches[T](data: list[T], batch_size: int, drop_single: bool = False) -> list[list[T]]:
    # a final short batch is kept: learn_batch averages by len(batch). drop_single drops it when
    # it has one example, which a network with batch norm refuses (the batch-norm workplan, D4)
    assert batch_size >= 1, f"batch_size must be at least 1; got {batch_size}"
    assert not drop_single or batch_size >= 2, f"batch norm needs batches of 2 or more; got batch_size={batch_size}"
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    if drop_single and len(batches[-1]) == 1:
        batches.pop()
    return batches


def _has_batch_norm(student: object) -> bool:
    # a network's batch_norm_index (array_network_base.py, backprop_network_base.py); a network
    # without one has no batch norm
    return getattr(student, "batch_norm_index", None) is not None


def _refuse_single_example_groups(student: object, examples: int, batch_size: int) -> None:
    # every batch is batch_size examples but the final short one: refuse before training a batch
    # size that would leave a ghost group of one example in either, which learn_batch refuses
    # (the batch-norm workplan, D6), rather than at the end of the first epoch
    specs = getattr(student, "layer_specs", None)
    if specs is None:
        return
    remainder = examples % batch_size
    sizes = [min(batch_size, examples)] + ([remainder] if remainder > 1 else [])
    for size in sizes:
        refuse_single_example_groups(specs, size)


def train_backprop_network_mini_batch[L](
    student: BatchTrainableClassifier[L],
    training_data: Sequence[Example[L]] | PreparedDataset,
    batch_size: int,
    learning_rate: float | Callable[[int], float] = 0.25,
    epochs: int = 1,
    reference_classifier: TargetClassifier[float] | None = None,
    reshuffle_each_epoch: bool = True,
    rng: Random | None = None,
    resume_from: RunCheckpoint | None = None,
) -> ConvergenceSeries:
    """
    train_linear_classifier_network with mini-batches, for gradient-based students (any network with
    learn_batch; the linear classifier's minimum-disturbance rule has none).

    learning_rate is a float or a schedule, read once per learn_batch against iterations, which here
    count batches.

    Reshuffles training_data every epoch by default (reshuffle_each_epoch=False keeps the batches
    fixed), drawing from rng, seeded from OS entropy if None (the RNG generators workplan, D6, D9);
    random.Random(s) gives the order random.seed(s) gave before. With reference_classifier, its
    disagreement samples draw from rng too. A final short batch is kept, except that a final batch
    of one is dropped for a network with batch norm, which can't train on one example (the
    batch-norm workplan, D4), and a batch size that leaves a batch norm's ghost group one example
    is refused before training (D6). training_data may be a PreparedDataset.

    Otherwise as train_linear_classifier_network: the pocket checkpoint of the best epoch, and the
    TrainingDiagnostic/ConvergenceSeries return.

    The result's .run_checkpoint is the run's state at the last epoch, taken before the pocket
    restores the best one (run_checkpoint.py). resume_from=, such a checkpoint, trains on from it:
    student takes its last epoch's checkpoint, rng its shuffle state, and the run its counters, its
    pocket and its series so far, so epochs counts the whole run, the epochs before included. The
    same training_data, batch_size, learning_rate, reference_classifier and reshuffle_each_epoch
    then take the steps the run would have taken had it never stopped, by bits.
    """

    assert len(training_data) >= 1, "training_data must not be empty"
    rng = Random() if rng is None else rng
    drop_single = _has_batch_norm(student)
    if drop_single:
        _refuse_single_example_groups(student, len(training_data), batch_size)

    prepared = _prepared_for(student, training_data)

    if resume_from is None:
        first_epoch = 0
        iterations = 0
        convergence: list[tuple[int, float]] = []
        if reference_classifier:
            convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student, rng=rng)))
        best_checkpoint = student.checkpoint()
        best_training_accuracy = _training_accuracy(student, training_data, prepared)
        best_epoch_index = -1  # -1: the untrained starting point was never beaten
        epoch_training_accuracies: list[float] = []
    else:
        assert epochs >= resume_from.epochs, f"epochs counts the whole run: {epochs} < {resume_from.epochs} done"
        first_epoch = resume_from.epochs
        iterations = resume_from.iterations
        convergence = list(resume_from.convergence)
        rng.setstate(resume_from.shuffle_state)
        student.restore_checkpoint(resume_from.network)
        best_checkpoint = resume_from.best
        best_training_accuracy = resume_from.best_training_accuracy
        best_epoch_index = resume_from.best_epoch_index
        epoch_training_accuracies = list(resume_from.epoch_training_accuracies)

    def record_disagreement() -> None:
        if reference_classifier:
            convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student, rng=rng)))

    # the prepared path shuffles row indices in place of the tuples: shuffle draws depend only
    # on the list's length, so a seed gives the same permutation, and the same batches, either way
    def epoch_order[T](examples: Sequence[T]) -> list[T]:
        epoch_data = list(examples)
        if reshuffle_each_epoch:
            rng.shuffle(epoch_data)
        return epoch_data

    def learn_epoch() -> None:
        nonlocal iterations
        if prepared is not None:
            assert isinstance(student, PreparedTrainableClassifier)  # _prepared_for's contract
            for indices in _chunk_into_batches(epoch_order(range(len(prepared))), batch_size, drop_single):
                student.learn_batch_rows(_rate(learning_rate, iterations), prepared, indices)
                iterations += 1
                record_disagreement()
        else:
            assert not isinstance(training_data, PreparedDataset)  # a PreparedDataset always has prepared
            for batch in _chunk_into_batches(epoch_order(training_data), batch_size, drop_single):
                student.learn_batch(_rate(learning_rate, iterations), batch)
                iterations += 1
                record_disagreement()

    for epoch_index in range(first_epoch, epochs):
        learn_epoch()

        training_accuracy = _training_accuracy(student, training_data, prepared)
        epoch_training_accuracies.append(training_accuracy)
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_epoch_index = epoch_index
            best_checkpoint = student.checkpoint()

    run_checkpoint = RunCheckpoint(
        network=student.checkpoint(),
        best=best_checkpoint,
        epochs=epochs,
        iterations=iterations,
        shuffle_state=rng.getstate(),
        best_epoch_index=best_epoch_index,
        best_training_accuracy=best_training_accuracy,
        epoch_training_accuracies=tuple(epoch_training_accuracies),
        convergence=tuple(convergence),
    )
    student.restore_checkpoint(best_checkpoint)

    result = ConvergenceSeries(convergence)
    result.diagnostic = TrainingDiagnostic(epoch_training_accuracies, best_epoch_index, best_training_accuracy)
    result.run_checkpoint = run_checkpoint
    return result
