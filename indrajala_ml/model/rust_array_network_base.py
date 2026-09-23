from __future__ import annotations

from typing import Sequence

import indrajala_math_rust as pa

from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class RustArrayNetworkBase:
    """
    The Rust-array-core-backed counterpart to ArrayNetworkBase - see that class's own docstring
    for the full rationale (this is the same collapse, one backend over). Not shared with
    ArrayNetworkBase itself: the two backends' array APIs differ throughout (np.array/np.argmax/
    np.zeros vs. pa.Array/pa.argmax/pa.Array.zeros), the same non-sharing ArrayLayer/
    RustArrayLayer already have today.
    """

    hidden_layer_cls: type = RustArrayLayer
    output_layer_cls: type = RustArrayLayer

    def __init__(self, layer_sizes: list[int], dimension: int, output_size: int) -> None:

        validate_layer_sizes(layer_sizes)

        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list[RustArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(self.hidden_layer_cls(size, previous_size))
            previous_size = size

        self.output_layer = self.output_layer_cls(output_size, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> "pa.Array":
        x = pa.Array(list(state))
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _set_training_mode(self, training: bool) -> None:
        # no-op for every sibling except dropout's own override
        pass

    def _target_array(self, category) -> "pa.Array":
        raise NotImplementedError

    def _target_batch_array(
        self, batch: Sequence[tuple[tuple[float, ...], object]], batch_size: int
    ) -> "pa.Array":
        raise NotImplementedError

    def learn(self, learning_rate: float, state: tuple[float, ...], category) -> None:
        activations = [pa.Array(list(state))]
        x = activations[0]
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

        # sgd_step is accumulate_gradient then apply_accumulated_gradient at batch_size=1, fused
        # into one call where the layer's update is plain SGD
        for layer, input_activation in zip(self.layers, activations):
            layer.sgd_step(input_activation, learning_rate)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], object]]) -> None:
        validate_batch(batch)
        batch_size = len(batch)

        activations = [pa.Array([list(state) for state, _target in batch])]
        X = activations[0]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                X = layer.forward_batch(X)
                activations.append(X)
        finally:
            self._set_training_mode(False)

        target_batch = self._target_batch_array(batch, batch_size)
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        # the same fan-in-aware scheme ArrayNetworkBase.randomize uses, drawn from
        # indrajala_math_rust.uniform instead of np.random.uniform - this can never be
        # seed-reproducible against the numpy sibling's own draws, since the two use unrelated
        # RNG implementations.
        previous_size = self.dimension
        for layer in self.layers:
            limit = 1.0 / (previous_size ** 0.5)
            layer.W = pa.uniform(-limit, limit, (layer.size, previous_size))
            layer.b = pa.uniform(-limit, limit, layer.size)
            previous_size = layer.size

    def snapshot(self) -> list[tuple["pa.Array", "pa.Array"]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple["pa.Array", "pa.Array"]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = W.copy()
            layer.b = b.copy()
