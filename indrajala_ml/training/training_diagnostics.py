from indrajala_ml.training.run_checkpoint import RunCheckpoint


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
