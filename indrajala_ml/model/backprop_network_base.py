from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Self, cast

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.layer_major import LayerMajorBatch
from indrajala_ml.model.layers.python.python_layer_builder import build_python_layers
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.optimizers.python_optimizer import PythonOptimizer, WeightSetState
from indrajala_ml.model.persistence.format2 import PYTHON
from indrajala_ml.model.persistence.format2_persistence import Format2Persistence
from indrajala_ml.model.protocols.layer_protocols import GeneratorLayer, TrainableLayer
from indrajala_ml.model.specs.bounds import validate_batch, validate_input_bounds
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.single_example import (
    batch_norm_index,
    refuse_single_example_groups,
    refuse_single_example_network,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from indrajala_ml.pcg64 import Pcg64Generator, default_rng


# LayerT, the hidden layers' type: dense layers, except in a network whose specs put conv and pool
# layers first (ConvMultiClassBackpropClassifierNetwork, the sequential networks)
class BackpropNetworkBase[LayerT: TrainableLayer = BackpropLayer](
    Format2Persistence[list[list[Any]], list[WeightSetState]]
):
    """
    What every pure-Python network shares: layers built from layer specs (layer_specs.py,
    python_layer_builder.py) behind a StateLayer, the forward pass, the optimizer (_update_rule,
    python_optimizer.py) applying the gradients, the hidden layers' backward pass, and
    snapshot/restore. The subclasses differ in the output layer's size, the
    predict_*/classify_state contract, and randomize().

    A sibling differs in its layer specs (_hidden_spec/_output_spec, as ArrayNetworkBase's), its
    update rule (_update_rule) and their hyperparameters.
    """

    # the constructor keyword arguments a sibling stores under the same attribute names (e.g.
    # ("momentum",)), which its specs and rule read, as ArrayNetworkBase's
    hyperparameters: ClassVar[tuple[str, ...]] = ()

    # the format-2 file's (format2.py) shape, and a preset's constructor arguments beside its
    # hyperparameters, named as the attributes that hold them. A Sequential network records no
    # preset (None).
    format2_shape: ClassVar[str]
    preset_arguments: ClassVar[tuple[str, ...] | None]

    implementation = PYTHON

    @classmethod
    def _format2_implementation(cls) -> str:
        return PYTHON

    def __init__(
        self,
        specs: Sequence[LayerSpec],
        input_shape: InputShape,
        input_bounds: list[tuple[float, float]],
    ) -> None:

        self.input_shape = input_shape
        self.layer_specs = list(specs)
        self.dimension = math.prod(input_shape)

        validate_input_bounds(self.dimension, input_bounds)
        self.input_bounds = input_bounds

        self.input_layer = StateLayer(self.dimension, input_bounds)

        # every layer with trained weights (and any pool layer), in forward order: the forward and
        # backward passes and snapshot/restore walk it
        self.trainable_layers: list[TrainableLayer] = build_python_layers(specs, input_shape, self.input_layer)
        # LayerT is what the specs build before the output layer: dense layers, unless conv and
        # pool layers come first
        self.hidden_layers = cast("list[LayerT]", self.trainable_layers[:-1])
        output_layer = self.trainable_layers[-1]
        assert isinstance(output_layer, BackpropLayer)  # validate_layer_specs: a Dense
        self.output_layer = output_layer
        # the index of the first batch-norm layer, if any: such a network trains layer-major
        # (layer_major.py, the batch-norm workplan's D3) and refuses a one-example training step (D4)
        self.batch_norm_index = batch_norm_index(specs)

        self.optimizer = PythonOptimizer(self._update_rule())
        # the dropout layers, which draw their masks from the network's generator
        self._generator_layers = [layer for layer in self.trainable_layers if isinstance(layer, GeneratorLayer)]
        # OS entropy until randomized(seed=, rng=) or an assignment sets it (the RNG generators
        # workplan, D9)
        self.rng = default_rng()

    @property
    def rng(self) -> Pcg64Generator:
        """
        The generator this network owns (the RNG generators workplan, D8, D5), the pure-Python port
        of numpy's default_rng: randomize() draws the weights from it and the dropout nodes their
        keep draws, one stream between them.
        """
        return self._rng

    @rng.setter
    def rng(self, rng: Pcg64Generator) -> None:
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

    @classmethod
    def randomized(cls, *args: Any, seed: Any = None, rng: Pcg64Generator | None = None, **kwargs: Any) -> Self:
        # every subclass's randomized signature is its __init__ signature; randomize() is per
        # subclass. seed (an int, a sequence of ints or any SeedSequence) seeds the network's own
        # generator, or rng is that generator, as ArrayNetworkBase.randomized; with neither it is
        # seeded from OS entropy
        assert seed is None or rng is None, "randomized takes seed or rng, not both"
        network = cls(*args, **kwargs)
        if rng is not None:
            network.rng = rng
        elif seed is not None:
            network.rng = default_rng(seed)
        network.randomize()
        return network

    # per subclass: the output layer's initialization, and the forward/backward passes over a
    # float target (single output) or a class index (multiclass)
    def randomize(self) -> None:
        raise NotImplementedError

    def _forward(self, state: tuple[float, ...]) -> Any:
        raise NotImplementedError

    def _backward(self, target: Any, /) -> None:
        raise NotImplementedError

    def _output_deltas(self, target: Any, /) -> None:
        # the output layer's deltas for one example's target, the first half of _backward
        raise NotImplementedError

    def update_state_layer(self, state: tuple[float, ...]) -> None:
        self.input_layer.update_state(state)

    def _forward_outputs(self, state: tuple[float, ...]) -> list[float]:
        self.update_state_layer(state)
        for layer in self.trainable_layers:
            layer.forward()
        return [node.value() for node in self.output_layer.nodes]

    def _backward_hidden_layers(self) -> None:
        for layer_index in reversed(range(len(self.hidden_layers))):
            next_layer = self.trainable_layers[layer_index + 1]
            self.hidden_layers[layer_index].compute_hidden_deltas(next_layer)

    def _set_training_mode(self, training: bool) -> None:
        # on for one training call only: the trainers classify with the same network between
        # learn steps. A no-op except for training-aware layers like DropoutLayer.
        for layer in self.trainable_layers:
            layer.set_training_mode(training)

    def _refuse_single_example(self) -> None:
        # learn's and a one-example learn_batch's check (D4)
        if self.batch_norm_index is not None:
            refuse_single_example_network(self.layer_specs, self.batch_norm_index)

    def learn(self, learning_rate: float, state: tuple[float, ...], category: Any) -> None:
        # one example's step; category is a float or a class index, as _learn_batch's targets.
        # Training mode for the forward pass only: a no-op but for dropout layers. A network with
        # batch norm refuses (D4).
        self._refuse_single_example()
        self._set_training_mode(True)
        try:
            self._forward(state)
        finally:
            self._set_training_mode(False)
        self._backward(category)
        self._apply_gradients(learning_rate)

    def _learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], Any]]) -> None:
        # forward, backward and accumulate per example, then one averaged update. A one-example
        # batch matches learn() bit for bit (tests/model/layers/python/test_gradient_accumulation.py). The target is
        # a float or a class index; _forward/_backward abstract over which. A network with batch
        # norm trains layer-major instead.
        validate_batch(batch)
        if self.batch_norm_index is not None:
            if len(batch) == 1:
                self._refuse_single_example()
            refuse_single_example_groups(self.layer_specs, len(batch))
            self._learn_batch_layer_major(learning_rate, batch)
            return
        self._set_training_mode(True)
        try:
            for state, target in batch:
                self._forward(state)
                self._backward(target)
                self._accumulate_gradients()
            self._apply_accumulated_gradients(learning_rate, len(batch))
        finally:
            self._set_training_mode(False)

    def _learn_batch_layer_major(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], Any]]) -> None:
        # the batch forward, then backward, layer by layer (layer_major.py), then one averaged update
        self._set_training_mode(True)
        try:
            pass_ = LayerMajorBatch(self.input_layer, self.trainable_layers, [state for state, _target in batch])
            pass_.forward()
            pass_.backward(lambda example: self._output_deltas(batch[example][1]))
            pass_.accumulate_gradients()
            self._apply_accumulated_gradients(learning_rate, len(batch))
        finally:
            self._set_training_mode(False)

    def _forward_batch_outputs(self, states: Sequence[tuple[float, ...]]) -> list[list[float]]:
        # the output rows of the forward pass learn_batch runs over states, in the training mode
        # the caller set: per example, or layer-major for a network with batch norm
        if self.batch_norm_index is None:
            return [self._forward_outputs(state) for state in states]
        pass_ = LayerMajorBatch(self.input_layer, self.trainable_layers, states)
        pass_.forward()
        return pass_.outputs()

    def _apply_gradients(self, learning_rate: float) -> None:
        optimizer = self.optimizer
        optimizer.begin_step()
        for index, layer in enumerate(self.trainable_layers):
            optimizer.step_single(index, layer, learning_rate)

    def _accumulate_gradients(self) -> None:
        for layer in self.trainable_layers:
            layer.accumulate_gradients()

    def _apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
        optimizer = self.optimizer
        optimizer.begin_step()
        for index, layer in enumerate(self.trainable_layers):
            optimizer.apply(index, layer, learning_rate, batch_size)

    def snapshot(self) -> list[list[Any]]:
        return [layer.snapshot_state() for layer in self.trainable_layers]

    def restore(self, snapshot: list[list[Any]]) -> None:
        for layer, layer_snapshot in zip(self.trainable_layers, snapshot):
            layer.restore_state(layer_snapshot)


def as_dense_layers(layers: Sequence[TrainableLayer]) -> list[BackpropLayer]:
    """
    layers, checked to be dense (BackpropLayer and its siblings), for what only a dense network
    does, such as BackpropClassifierNetwork's bounds-width randomize of its node weights.
    """
    dense = [layer for layer in layers if isinstance(layer, BackpropLayer)]
    assert len(dense) == len(layers), f"expected only dense layers; got {[type(layer).__name__ for layer in layers]}"
    return dense


def randomize_fan_in_aware(network: BackpropNetworkBase[Any]) -> None:
    """
    Fan-in-aware initialization, limit = 1/sqrt(fan_in) per layer, so a layer's weighted input sum
    doesn't saturate every sigmoid once fan-in reaches the tens or hundreds. On UCI digits: 99.5%
    training and 96.9% test accuracy. Used by MultiClassBackpropClassifierNetwork (and so the conv
    network), FanInAwareBackpropClassifierNetwork and the sequential networks.

    Every layer in forward order: a dense layer's nodes from its input layer's size, a conv
    layer's kernels from their receptive field, and a pool layer draws nothing, so adding pooling
    never shifts a later layer's draws.

    Unlike BackpropClassifierNetwork.randomize()'s bounds-width scaling (tuned for 1-2D geometric
    problems) it works at any dimension: the ensemble's 784-pixel MNIST sub-networks reach 6.6
    points higher test accuracy with it; bounds-width scaling leaves 83.5% of hidden activations
    saturated at initialization.
    """

    for layer in network.trainable_layers:
        layer.randomize_fan_in_aware(network.rng)
