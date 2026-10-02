# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, NoReturn, Self, cast

from indrajala_ml.model.array_layer_builder import build_array_layers
from indrajala_ml.model.array_protocols import (
    ArrayBackend,
    ArrayNetworkLayer,
    ArrayOptimizer,
    BackendArray,
    BiasFreeArrayLayer,
    ProjectionsArrayLayer,
    RunningStateLayer,
    TrainedArrayLayer,
    TrainingModeLayer,
    WeightedArrayLayer,
)
from indrajala_ml.model.bounds import validate_batch
from indrajala_ml.model.format2_persistence import Format2Persistence
from indrajala_ml.model.layer_protocols import GeneratorLayer
from indrajala_ml.model.layer_specs import (
    Dense,
    InputShape,
    LayerSpec,
    batch_norm_index,
    refuse_single_example_groups,
    refuse_single_example_network,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from indrajala_ml.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset


class ArrayNetworkBase[A: BackendArray](Format2Persistence[list[tuple[A, ...]], list[A]]):
    """
    What every array-backed network shares, numpy and Rust (NumpyArrayNetworkBase and
    RustArrayNetworkBase set the backend, A being its array type): layers built from layer specs
    (layer_specs.py, array_layer_builder.py), the forward pass, learn/learn_batch and their
    prepared-dataset forms, classify_rows, fan-in-aware randomize, snapshot/restore and
    checkpoint/restore_checkpoint, all over self.layers whatever their kinds.

    A sibling differs only in its layer specs (_hidden_spec/_output_spec), its update rule
    (_update_rule) and their hyperparameters. The forward and backward formulas live in the layer
    classes, the weight updates in the network's optimizer (optimizers.py), which holds the rule's
    state (momentum's velocities, Adam's m, v and t). What differs between the multiclass,
    single-output and conv networks (their constructor arguments, classify_state, targets,
    class_count, save/load) is in the shape mixins in array_network_shapes.py, listed before the
    backend's base.
    """

    # the array operations that differ between numpy and Rust (array_backend.py), set by the
    # backend's base
    backend: ArrayBackend[A]

    # the constructor keyword arguments a sibling stores under the same attribute names (e.g.
    # ("beta1", "beta2", "epsilon")), which its specs and rule read: a preset's format-2 file
    # records them with its arguments, and a legacy envelope beside its other fields
    hyperparameters: ClassVar[tuple[str, ...]] = ()

    # the format-2 file's (format2.py) shape, and a preset's constructor arguments beside its
    # hyperparameters, named as the attributes that hold them: set by the shape mixins. A
    # Sequential network records no preset (None).
    format2_shape: ClassVar[str]
    preset_arguments: ClassVar[tuple[str, ...] | None]

    def __init__(self, specs: Sequence[LayerSpec], input_shape: InputShape) -> None:
        self.input_shape = input_shape
        self.layer_specs = list(specs)
        self.layers = cast("list[ArrayNetworkLayer[A]]", build_array_layers(specs, input_shape, self.backend.name))
        self.output_layer = cast("WeightedArrayLayer[A]", self.layers[-1])
        # the dropout and batch-norm layers, which _set_training_mode switches
        self._training_mode_layers = [layer for layer in self.layers if isinstance(layer, TrainingModeLayer)]
        # the dropout layers, which draw their masks from the network's generator
        self._generator_layers = [layer for layer in self.layers if isinstance(layer, GeneratorLayer)]
        # OS entropy until randomized(seed=, rng=) or an assignment sets it (the RNG generators
        # workplan, D9)
        self.rng = self.backend.default_rng()
        # the index of the first batch-norm layer, if any: such a network refuses a one-example
        # training step (the batch-norm workplan, D4)
        self.batch_norm_index = batch_norm_index(specs)
        self.optimizer = self._new_optimizer()

    @property
    def rng(self) -> Any:
        """
        The generator this network owns (the RNG generators workplan, D8), the backend's
        default_rng: randomize() draws the weights from it and the dropout layers their masks, one
        stream between them.
        """
        return self._rng

    @rng.setter
    def rng(self, rng: Any) -> None:
        self._rng = rng
        for layer in self._generator_layers:
            layer.set_rng(rng)

    def _hidden_spec(self, size: int) -> Dense:
        # a dense hidden layer; the ReLU and dropout siblings return theirs
        return Dense(size)

    def _output_spec(self, size: int) -> Dense:
        # the output layer; the softmax and cross-entropy siblings return theirs
        return Dense(size, output=True)

    def _dense_specs(self, layer_sizes: Sequence[int], output_size: int) -> list[LayerSpec]:
        return [*(self._hidden_spec(size) for size in layer_sizes), self._output_spec(output_size)]

    def _update_rule(self) -> UpdateRule:
        # plain SGD; the momentum, Adam and L2 siblings return their rule, from their hyperparameters
        return SGD()

    def _new_optimizer(self) -> ArrayOptimizer[A]:
        return self.backend.optimizer(self._update_rule())

    def _forward(self, state: tuple[float, ...]) -> A:
        return self._forward_input(self.backend.vector(state))

    def _forward_input(self, x: A) -> A:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def classify_row(self, prepared: PreparedDataset, index: int) -> Any:
        # classify_state for row index of a prepared dataset; _classify_output is the shape
        # class's argmax or 0.5 threshold, shared with classify_state
        return self._classify_output(self._forward_input(self.backend.row(self._prepared_states(prepared), index)))

    def classify_rows(self, prepared: PreparedDataset) -> list[Any]:
        # classify_row for every row, as the trainers' accuracy pass needs it, through
        # forward_batch over chunks of rows. The
        # predictions are classify_row's, but only by construction on Rust, where a batched
        # forward row equals the single-example forward exactly: numpy's X @ W.T can differ from
        # W @ x in the last ULP, so an argmax between outputs an ULP apart could differ
        states = self._prepared_states(prepared)
        predictions: list[Any] = []
        for start in range(0, len(prepared), CLASSIFY_CHUNK_ROWS):
            X = self.backend.row_range(states, start, min(start + CLASSIFY_CHUNK_ROWS, len(prepared)))
            for layer in self.layers:
                X = layer.forward_batch(X)
            predictions.extend(self._classify_output_batch(X))
        return predictions

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> PreparedDataset:
        return PreparedDataset.from_rows(rows, self.backend.name)

    def _prepared_states(self, prepared: PreparedDataset) -> A:
        name = self.backend.name
        assert prepared.backend == name, f"a {name} network needs a {name} dataset; got {prepared.backend!r}"
        return cast(A, prepared.states)  # the assert: this backend's array type

    def _set_training_mode(self, training: bool) -> None:
        # dropout on for the forward pass of a learn call only (learn* brackets it)
        for layer in self._training_mode_layers:
            layer.set_training_mode(training)

    # the shape's hooks (array_network_shapes.py), over its label type: a float (single output) or
    # a class index (multiclass)

    def _target_array(self, category: Any) -> A:
        raise NotImplementedError

    def _target_batch_array(self, categories: Sequence[Any]) -> A:
        raise NotImplementedError

    def _classify_output(self, output: A) -> Any:
        raise NotImplementedError

    def _classify_output_batch(self, output_batch: A) -> list[Any]:
        # _classify_output for each row of a (batch, output_size) forward_batch output
        raise NotImplementedError

    # learn/learn_row and learn_batch/learn_batch_rows only differ in where the input array
    # comes from (a converted tuple, or a prepared dataset's rows); the training step itself is
    # _learn_input/_learn_batch_input, shared, so the two paths can't drift

    def learn(self, learning_rate: float, state: tuple[float, ...], category: Any) -> None:
        self._learn_input(learning_rate, self.backend.vector(state), category)

    def learn_row(self, learning_rate: float, prepared: PreparedDataset, index: int) -> None:
        self._learn_input(
            learning_rate, self.backend.row(self._prepared_states(prepared), index), prepared.labels[index]
        )

    def _refuse_single_example(self, batch_norm_index: int) -> NoReturn:
        refuse_single_example_network(self.layer_specs, batch_norm_index)

    def _learn_input(self, learning_rate: float, x: A, category: Any) -> None:
        if self.batch_norm_index is not None:
            self._refuse_single_example(self.batch_norm_index)
        activations = [x]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)
        finally:
            self._set_training_mode(False)

        target = self._target_array(category)
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        # step_single is accumulate_gradient then apply at batch_size=1, which the Rust optimizer
        # fuses into one call for plain SGD
        optimizer = self.optimizer
        optimizer.begin_step()
        for index, (layer, input_activation) in enumerate(zip(self.layers, activations)):
            optimizer.step_single(index, layer, input_activation, learning_rate)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], object]]) -> None:
        validate_batch(batch)
        X = self.backend.matrix([state for state, _target in batch])
        self._learn_batch_input(learning_rate, X, [category for _state, category in batch])

    def learn_batch_rows(self, learning_rate: float, prepared: PreparedDataset, indices: Sequence[int]) -> None:
        validate_batch(indices)
        X = self.backend.rows(self._prepared_states(prepared), indices)
        self._learn_batch_input(learning_rate, X, [prepared.labels[i] for i in indices])

    def _learn_batch_input(self, learning_rate: float, X: A, categories: Sequence[Any]) -> None:
        batch_size = len(categories)
        if self.batch_norm_index is not None:
            if batch_size == 1:
                self._refuse_single_example(self.batch_norm_index)
            refuse_single_example_groups(self.layer_specs, batch_size)

        activations = [X]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                X = layer.forward_batch(X)
                activations.append(X)
        finally:
            self._set_training_mode(False)

        target_batch = self._target_batch_array(categories)
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        optimizer = self.optimizer
        optimizer.begin_step()
        for index, (layer, input_activation_batch) in enumerate(zip(self.layers, activations)):
            layer.accumulate_gradient_batch(input_activation_batch)
            optimizer.apply(index, layer, learning_rate, batch_size)

    @classmethod
    def randomized(cls, *args: Any, seed: Any = None, rng: Any = None, **kwargs: Any) -> Self:
        # every sibling's randomized signature is its __init__ signature, so one pass-through
        # serves them all, positional or keyword, defaults included. seed (an int, a sequence of
        # ints or a SeedSequence) seeds the network's own generator, or rng is that generator; with
        # neither it is seeded from OS entropy
        assert seed is None or rng is None, "randomized takes seed or rng, not both"
        network = cls(*args, **kwargs)
        if rng is not None:
            network.rng = rng
        elif seed is not None:
            network.rng = network.backend.default_rng(seed)
        network.randomize()
        return network

    def randomize(self) -> None:
        # fan-in-aware (limit = 1/sqrt(fan_in)), drawn from the network's generator: from the same
        # seed, numpy and Rust draw the same weights. W then b per layer, in forward
        # order; a W is (rows, fan_in), a dense layer's (size, input_size) and a conv layer's
        # (channel_count, input_channels * kernel_size**2). A linear layer draws its W only, an
        # attention layer each projection's W then b in turn, and a pool, batch-norm, layer-norm,
        # patches, position or token-mean layer draws nothing.
        for layer in self.layers:
            if isinstance(layer, WeightedArrayLayer):
                rows, fan_in = layer.W.shape
                layer.W, layer.b = self.backend.random_layer(self.rng, rows, fan_in)
            elif isinstance(layer, BiasFreeArrayLayer):
                linear = cast("BiasFreeArrayLayer[A]", layer)
                rows, fan_in = linear.W.shape
                linear.W = self.backend.random_weights(self.rng, rows, fan_in)
            elif isinstance(layer, ProjectionsArrayLayer):
                projections = cast("ProjectionsArrayLayer[A]", layer)
                projections.set_parameters(
                    [
                        array
                        for rows, fan_in in projections.projection_shapes
                        for array in self.backend.random_layer(self.rng, rows, fan_in)
                    ]
                )

    @classmethod
    def _extra_init_kwargs(cls, state: dict[str, Any]) -> dict[str, Any]:
        # a legacy envelope's hyperparameters, saved beside its other fields, as constructor kwargs
        return {name: state[name] for name in cls.hyperparameters}

    def snapshot(self) -> list[tuple[A, ...]]:
        # per layer, its parameters then its running state: (W, b) for a dense or conv layer, (W,)
        # for a linear one, (gamma, beta, running_mean, running_var) for a batch-norm one, and ()
        # for a pool layer
        return [tuple(array.copy() for array in _layer_state(layer)) for layer in self.layers]

    def restore(self, snapshot: Sequence[tuple[Any, ...]]) -> None:
        # accepts this backend's arrays or nested lists (a loaded file, or a snapshot pickled
        # across a worker boundary)
        for layer, entry in zip(self.layers, snapshot):
            expected = len(_layer_state(layer))
            assert len(entry) == expected, f"a {type(layer).__name__} restores {expected} arrays; got {entry!r}"
            arrays = [self.backend.owned(values) for values in entry]
            if isinstance(layer, TrainedArrayLayer):
                count = len(layer.parameters())
                layer.set_parameters(arrays[:count])
                arrays = arrays[count:]
            if isinstance(layer, RunningStateLayer):
                cast("RunningStateLayer[A]", layer).set_running_state(arrays)

    @property
    def implementation(self) -> str:
        return self.backend.name

    @classmethod
    def _format2_implementation(cls) -> str:
        return cls.backend.name


def _layer_state[A: BackendArray](layer: ArrayNetworkLayer[A]) -> tuple[A, ...]:
    # what snapshot() keeps of a layer: its parameters, then its running state
    parameters = layer.parameters() if isinstance(layer, TrainedArrayLayer) else ()
    return parameters + (layer.running_state() if isinstance(layer, RunningStateLayer) else ())
