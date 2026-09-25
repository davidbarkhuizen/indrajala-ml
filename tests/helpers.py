import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.adam_layer import make_adam_layer_cls
from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape, ArraySingleOutputShape
from indrajala_ml.model.array_protocols import ArrayBackend, BackendArray, WeightedArrayLayer
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
    CrossEntropyOutputLayer,
)
from indrajala_ml.model.classifier_protocols import State
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)

# conftest's `backend` fixture: either array backend, NUMPY or RUST
Backend = ArrayBackend[Any]
# a backend's array constructor on nested lists: np.array (numpy) or pa.Array (Rust)
Wrap = Callable[[Any], Any]
# the array network a matching_* helper builds, of whichever class the caller passes
ArrayNetworkT = TypeVar("ArrayNetworkT", bound=ArrayNetworkBase[Any])
MultiClassT = TypeVar("MultiClassT", bound=ArrayMultiClassShape[Any])
SingleOutputT = TypeVar("SingleOutputT", bound=ArraySingleOutputShape[Any])


class _Snapshottable(Protocol):
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...


class _SavableMultiClass(Protocol):
    def save(self, path: str) -> None: ...

    def snapshot(self) -> Any: ...

    def classify_state(self, state: State) -> int: ...

    def predict_probabilities(self, state: State) -> list[float]: ...


SavableT = TypeVar("SavableT", bound=_SavableMultiClass)


def approx(expected: object, rel: float | None = None, abs: float | None = None) -> object:
    """pytest.approx, typed: pytest 8.1 leaves its parameters unannotated, which strict mode reports at every call."""
    return pytest.approx(expected, rel=rel, abs=abs)  # pyright: ignore[reportUnknownMemberType]


class FixedDownstream:
    """A stand-in next layer whose downstream gradient is fixed: what compute_hidden_delta*
    reads from the layer after it."""

    def __init__(self, gradient_batch: Any, gradient: Any) -> None:
        self.gradient_batch = gradient_batch
        self.gradient = gradient

    def downstream_batch(self) -> Any:
        return self.gradient_batch

    def downstream(self) -> Any:
        return self.gradient


def fixed_downstream(backend: Backend, gradient_batch: FloatArray) -> Any:
    """A FixedDownstream in backend's arrays; its gradient is gradient_batch's first row. Any: a
    layer types its next layer as one of its own backend's, which this stand-in isn't."""
    return FixedDownstream(backend.owned(gradient_batch.tolist()), backend.owned(gradient_batch[0].tolist()))


ClassT = TypeVar("ClassT")


def all_subclasses(cls: type[ClassT]) -> Iterator[type[ClassT]]:
    """
    Every subclass of cls in indrajala_ml, at any depth (only those of modules imported so far):
    not the test files' own, so a test's sibling classes don't join a walk over the real ones.
    """
    for subclass in cls.__subclasses__():
        if subclass.__module__.startswith("indrajala_ml."):
            yield subclass
        yield from all_subclasses(subclass)


def random_vector(rng: random.Random, n: int) -> list[float]:
    return [rng.uniform(-3.0, 3.0) for _ in range(n)]


def random_matrix(rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    return [random_vector(rng, cols) for _ in range(rows)]


def to_numpy(array: BackendArray) -> FloatArray:
    """Either backend's array as a numpy array, through .tolist()."""
    return np.array(array.tolist())


def rust_to_numpy(array: pa.Array) -> FloatArray:
    """A 1-D or 2-D pa.Array as a numpy array, read element by element through its indexing."""
    if len(array.shape) == 1:
        return np.array([array[i] for i in range(array.shape[0])])
    rows, cols = array.shape
    return np.array([[array[r, c] for c in range(cols)] for r in range(rows)])


def weighted(layer: object) -> WeightedArrayLayer[Any]:
    """An array network's layer, checked to have weights (a dense or conv layer, not a pool layer)."""
    assert isinstance(layer, WeightedArrayLayer), f"a {type(layer).__name__} has no weights"
    return cast("WeightedArrayLayer[Any]", layer)


def conv_layer(network: ConvMultiClassBackpropClassifierNetwork, index: int) -> ConvLayer:
    """network.conv_layers[index], checked to be a conv (not pool) layer."""
    layer = network.conv_layers[index]
    assert isinstance(layer, ConvLayer), f"conv_layers[{index}] is a {type(layer).__name__}"
    return layer


def conv_layers_only(network: ConvMultiClassBackpropClassifierNetwork) -> list[ConvLayer]:
    """network.conv_layers, checked to hold no pool layer."""
    layers = [layer for layer in network.conv_layers if isinstance(layer, ConvLayer)]
    assert len(layers) == len(network.conv_layers), "expected only conv layers"
    return layers


def assert_save_and_load_round_trip(
    network: SavableT, load_fn: Callable[[str], SavableT], tmp_path: Path, filename: str, states: Iterable[State]
) -> SavableT:
    """
    Saves network, reloads it with load_fn, and asserts the snapshot and predictions match
    exactly. Returns the loaded network for class-specific assertions.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.snapshot() == network.snapshot()
    for state in states:
        assert loaded.classify_state(state) == network.classify_state(state)
        assert loaded.predict_probabilities(state) == approx(network.predict_probabilities(state))

    return loaded


def assert_randomize_breaks_symmetry(network: BackpropNetworkBase[Any]) -> None:
    """
    Nodes in a layer must start with different weights: identical ones get identical gradients
    forever, collapsing the layer to one unit.
    """
    weight_sets = [tuple(node.input_node_weights) for node in network.hidden_layers[0].nodes]
    assert len(set(weight_sets)) == len(weight_sets)


def assert_snapshot_restore_round_trip(network: _Snapshottable, step: Callable[[], object], times: int = 5) -> None:
    """
    Snapshot, apply `step` `times` times (asserting the network moved), restore, and assert
    the snapshot is back to the original.
    """
    before = network.snapshot()

    for _ in range(times):
        step()

    assert network.snapshot() != before

    network.restore(before)

    assert network.snapshot() == before


def wire_fixed_single_hidden_node(network: BackpropNetworkBase[Any]) -> None:
    """
    The hand-derivation fixture of the *_backprop_model.py tests: hidden weight=0.5, bias=0.1,
    output weight=0.8, bias=-0.2.
    """
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([0.8])
    output_node.bias = -0.2


def set_random_node_weights(
    rng: random.Random, node_layer: BackpropLayer, input_size: int, limit: float
) -> tuple[list[list[float]], list[float]]:
    """
    Draws a (size, input_size) weight matrix, then a size-length bias vector, each from
    rng.uniform(-limit, limit), sets them on node_layer's nodes, and returns (weights, biases) so
    an array-backed counterpart can be given the identical values.
    """
    size = len(node_layer.nodes)
    weights = [[rng.uniform(-limit, limit) for _ in range(input_size)] for _ in range(size)]
    biases = [rng.uniform(-limit, limit) for _ in range(size)]
    for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
        node.update_input_weights(node_weights)
        node.bias = bias
    return weights, biases


def inject_matching_weights(
    rng: random.Random,
    node_network: BackpropNetworkBase[Any],
    array_network: ArrayNetworkBase[Any],
    wrap: Wrap,
    dimension: int,
) -> None:
    """
    Gives a per-node network and its dense array-backed sibling identical random weights, layer
    by layer from rng.uniform(-2, 2). `wrap` converts the nested lists into the array backend's
    own array type - np.array for the numpy sibling, pa.Array for the Rust one.
    """
    assert len(node_network.trainable_layers) == len(array_network.layers)
    previous_size = dimension
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        assert isinstance(node_layer, BackpropLayer)
        weights, biases = set_random_node_weights(rng, node_layer, previous_size, 2.0)
        weighted_layer = weighted(array_layer)
        weighted_layer.W = wrap(weights)
        weighted_layer.b = wrap(biases)
        previous_size = len(weights)


def matching_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
) -> tuple[MultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    A MultiClassBackpropClassifierNetwork and an array_network_cls network with identical
    injected weights (the backends' RNGs aren't comparable with Python's random, so randomize()
    isn't used).
    """
    node_network = MultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


def matching_single_output_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    bounds: float = 10.0,
) -> tuple[FanInAwareBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the single-output array networks: the reference is
    FanInAwareBackpropClassifierNetwork, whose initialization matches theirs.
    """
    node_network = FanInAwareBackpropClassifierNetwork(layer_sizes, dimension, [(-bounds, bounds)] * dimension)
    array_network = array_network_cls(layer_sizes, dimension)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


def matching_cross_entropy_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    bounds: float = 10.0,
) -> tuple[BinaryCrossEntropyBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the single-output cross-entropy networks, against
    BinaryCrossEntropyBackpropClassifierNetwork.
    """
    node_network = BinaryCrossEntropyBackpropClassifierNetwork(layer_sizes, dimension, [(-bounds, bounds)] * dimension)
    array_network = array_network_cls(layer_sizes, dimension)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class AdamMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass Adam network, the parity reference for the Adam array networks:
    MultiClassBackpropClassifierNetwork with make_adam_layer_cls hidden and output layers, as
    AdamBackpropClassifierNetwork builds the single-output one. There is no production per-node
    multiclass Adam network.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> None:
        layer_cls = make_adam_layer_cls(beta1, beta2, epsilon)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_adam_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    bounds: float = 10.0,
) -> tuple[AdamMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the Adam networks: an
    AdamMultiClassBackpropClassifierNetwork and array_network_cls with the same beta1/beta2/epsilon
    and identical injected weights.
    """
    node_network = AdamMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, beta1, beta2, epsilon
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, beta1, beta2, epsilon)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class L2MultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass L2 network, the parity reference for the L2 array networks:
    MultiClassBackpropClassifierNetwork with make_l2_layer_cls hidden and output layers.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        l2_lambda: float,
    ) -> None:
        layer_cls = make_l2_layer_cls(l2_lambda)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_l2_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    l2_lambda: float,
    bounds: float = 10.0,
) -> tuple[L2MultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the L2 networks, with l2_lambda.
    """
    node_network = L2MultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, l2_lambda
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, l2_lambda)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class MomentumMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass momentum network, the parity reference for the momentum array
    networks: MultiClassBackpropClassifierNetwork with make_momentum_layer_cls hidden and output
    layers.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        momentum: float,
    ) -> None:
        layer_cls = make_momentum_layer_cls(momentum)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_momentum_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    momentum: float,
    bounds: float = 10.0,
) -> tuple[MomentumMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the momentum networks, with momentum.
    """
    node_network = MomentumMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, momentum
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, momentum)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class ReLUMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass ReLU network, the parity reference for the ReLU array networks:
    ReLULayer hidden layers and a sigmoid output layer.
    """

    hidden_layer_cls = ReLULayer


def matching_relu_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
) -> tuple[ReLUMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the ReLU networks.
    """
    node_network = ReLUMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class DropoutMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass dropout network, the reference for the dropout array networks:
    make_dropout_layer_cls hidden layers and a sigmoid output layer. It is a reference at eval
    only, where dropout does nothing: training draws masks from unrelated RNGs.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        drop_probability: float,
    ) -> None:
        self.hidden_layer_cls = make_dropout_layer_cls(drop_probability)
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_dropout_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    drop_probability: float,
    bounds: float = 10.0,
) -> tuple[DropoutMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the dropout networks, with drop_probability; for
    eval-mode comparisons only (see DropoutMultiClassBackpropClassifierNetwork).
    """
    node_network = DropoutMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, drop_probability
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, drop_probability)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


def matching_softmax_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
) -> tuple[SoftmaxMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the softmax networks, against the production
    SoftmaxMultiClassBackpropClassifierNetwork.
    """
    node_network = SoftmaxMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


class CrossEntropyMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node multiclass cross-entropy network, the parity reference for the multiclass
    cross-entropy array networks: sigmoid hidden layers and a CrossEntropyOutputLayer output.
    """

    output_layer_cls = CrossEntropyOutputLayer


def matching_cross_entropy_multiclass_array_backprop_networks(
    rng: random.Random,
    array_network_cls: Callable[..., ArrayNetworkT],
    wrap: Wrap,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
) -> tuple[CrossEntropyMultiClassBackpropClassifierNetwork, ArrayNetworkT]:
    """
    matching_array_backprop_networks for the multiclass cross-entropy networks.
    """
    node_network = CrossEntropyMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    inject_matching_weights(rng, node_network, array_network, wrap, dimension)
    return node_network, array_network


def matching_conv_array_backprop_networks(
    rng: random.Random,
    input_height: int,
    input_width: int,
    conv_specs: Sequence[ConvSpec | PoolSpec],
    dense_layer_sizes: list[int],
    class_count: int,
    array_network_cls: type[ConvVectorizedMultiClassBackpropClassifierNetwork]
    | type[ConvRustArrayMultiClassBackpropClassifierNetwork] = ConvVectorizedMultiClassBackpropClassifierNetwork,
    wrap: Wrap = np.array,
) -> tuple[
    ConvMultiClassBackpropClassifierNetwork,
    ConvVectorizedMultiClassBackpropClassifierNetwork | ConvRustArrayMultiClassBackpropClassifierNetwork,
]:
    """
    A ConvMultiClassBackpropClassifierNetwork and an array conv network (array_network_cls,
    numpy or Rust, with `wrap` its backend's array constructor) with identical injected weights,
    conv kernels included (a conv W's row c is kernel c's weights). Pool layers have none.
    """
    node_network = ConvMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    array_network = array_network_cls(input_height, input_width, conv_specs, dense_layer_sizes, class_count)

    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if isinstance(node_layer, ConvLayer):
            assert isinstance(array_layer, (ConvArrayLayer, ConvRustArrayLayer))
            for kernel in node_layer.kernels:
                kernel.weights = [rng.uniform(-1.0, 1.0) for _ in range(array_layer.fan_in)]
                kernel.bias = rng.uniform(-0.5, 0.5)
            array_layer.W = wrap([kernel.weights for kernel in node_layer.kernels])
            array_layer.b = wrap([kernel.bias for kernel in node_layer.kernels])
        elif isinstance(node_layer, BackpropLayer):  # a pool layer has no weights
            assert isinstance(array_layer, (ArrayLayer, RustArrayLayer))
            for node in node_layer.nodes:
                node.update_input_weights([rng.uniform(-1.0, 1.0) for _ in range(array_layer.input_size)])
                node.bias = rng.uniform(-1.0, 1.0)
            array_layer.W = wrap([list(node.input_node_weights) for node in node_layer.nodes])
            array_layer.b = wrap([node.bias for node in node_layer.nodes])

    return node_network, array_network


def copy_conv_network_weights_into_array_network(
    node_network: ConvMultiClassBackpropClassifierNetwork,
    array_network: ConvVectorizedMultiClassBackpropClassifierNetwork,
) -> None:
    """Copies a ConvMultiClassBackpropClassifierNetwork's weights (e.g. after its own seeded
    randomize()) into a same-shaped ConvVectorizedMultiClassBackpropClassifierNetwork."""
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if isinstance(node_layer, ConvLayer):
            weighted_layer = weighted(array_layer)
            weighted_layer.W = np.array([kernel.weights for kernel in node_layer.kernels])
            weighted_layer.b = np.array([kernel.bias for kernel in node_layer.kernels])
        elif isinstance(node_layer, BackpropLayer):  # a pool layer has no weights
            weighted_layer = weighted(array_layer)
            weighted_layer.W = np.array([node.input_node_weights for node in node_layer.nodes])
            weighted_layer.b = np.array([node.bias for node in node_layer.nodes])


def assert_conv_array_network_weights_match(
    node_network: ConvMultiClassBackpropClassifierNetwork,
    array_network: ArrayNetworkBase[Any],
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """The conv counterpart of assert_array_network_weights_match, for either backend: conv
    layers compare per kernel, pool layers have nothing to compare."""
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if isinstance(node_layer, ConvLayer):
            expected_W = np.array([kernel.weights for kernel in node_layer.kernels])
            expected_b = np.array([kernel.bias for kernel in node_layer.kernels])
        elif isinstance(node_layer, BackpropLayer):
            expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
            expected_b = np.array([node.bias for node in node_layer.nodes])
        else:  # a pool layer has nothing to compare
            continue
        weighted_layer = weighted(array_layer)
        np.testing.assert_allclose(weighted_layer.W.tolist(), expected_W, rtol=rtol, atol=atol)
        np.testing.assert_allclose(weighted_layer.b.tolist(), expected_b, rtol=rtol, atol=atol)


def matching_conv_numpy_rust_networks(
    rng: random.Random,
    input_height: int,
    input_width: int,
    conv_specs: Sequence[ConvSpec | PoolSpec],
    dense_layer_sizes: list[int],
    class_count: int,
) -> tuple[ConvVectorizedMultiClassBackpropClassifierNetwork, ConvRustArrayMultiClassBackpropClassifierNetwork]:
    """
    A ConvVectorizedMultiClassBackpropClassifierNetwork and a
    ConvRustArrayMultiClassBackpropClassifierNetwork with identical injected weights - drawn from
    rng into the numpy network, then restored into the Rust one from its snapshot (the two
    backends' RNGs aren't comparable, so neither network's own randomize() is used).
    """
    numpy_network = ConvVectorizedMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    rust_network = ConvRustArrayMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    for layer in numpy_network.layers:
        if isinstance(layer, WeightedArrayLayer):  # every layer but a pool layer
            rows, cols = layer.W.shape
            layer.W = np.array([[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)])
            layer.b = np.array([rng.uniform(-0.5, 0.5) for _ in range(layer.b.shape[0])])
    rust_network.restore(numpy_network.snapshot())
    return numpy_network, rust_network


def assert_array_network_weights_match(
    node_network: BackpropNetworkBase[Any],
    array_network: ArrayNetworkBase[Any],
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """
    Compares a per-node network's weights with an array network's, through .tolist(), which
    numpy arrays and pa.Array both support.
    """
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        assert isinstance(node_layer, BackpropLayer)
        expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
        expected_b = np.array([node.bias for node in node_layer.nodes])
        weighted_layer = weighted(array_layer)
        assert np.allclose(weighted_layer.W.tolist(), expected_W, rtol=rtol, atol=atol)
        assert np.allclose(weighted_layer.b.tolist(), expected_b, rtol=rtol, atol=atol)


def assert_array_network_snapshot_restore_round_trip(
    array_network_cls: Any, layer_sizes: list[int], dimension: int, class_count: int, *hyperparameters: float
) -> None:
    """
    A randomized array network's snapshot restored into a fresh one gives the same weights,
    compared through .tolist(). hyperparameters are the constructor's further arguments (e.g.
    momentum).
    """
    network = array_network_cls.randomized(layer_sizes, dimension, class_count, *hyperparameters)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension, class_count, *hyperparameters)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_single_output_array_network_snapshot_restore_round_trip(
    array_network_cls: Any, layer_sizes: list[int], dimension: int
) -> None:
    """
    The single-output analogue of assert_array_network_snapshot_restore_round_trip above, for
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork (no class_count argument).
    """
    network = array_network_cls.randomized(layer_sizes, dimension)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_single_output_array_network_save_load_round_trip(
    network: SingleOutputT, load_fn: Callable[[str], SingleOutputT], tmp_path: Path, filename: str, state: State
) -> SingleOutputT:
    """
    assert_array_network_save_load_round_trip below for single-output networks
    (predict_probability, no class_count).
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.predict_probability(state) == approx(network.predict_probability(state))

    return loaded


def assert_array_network_save_load_round_trip(
    network: MultiClassT, load_fn: Callable[[str], MultiClassT], tmp_path: Path, filename: str, state: State
) -> MultiClassT:
    """
    assert_save_and_load_round_trip for array networks: array == is elementwise, so this
    compares layer_sizes, dimension, class_count and predict_probabilities instead of snapshots.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.predict_probabilities(state) == approx(network.predict_probabilities(state))

    return loaded


def classifier_with_bounded_square_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly the square [-1, 1] x [-1, 1]: x > -1, x < 1, y > -1, y < 1
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([0.0, -1.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def classifier_with_tiny_bounded_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region [-0.1, 0.1]^2: 0.01% of square_bounds(10.0), for tight-box sampling
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 0.1), ([-1.0, 0.0], 0.1), ([0.0, 1.0], 0.1), ([0.0, -1.0], 0.1)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def unreachable_class_classifier(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # the decision boundary never crosses these bounds, so one class can't be sampled: for the
    # guard that stops the sampling loop hanging
    classifier = LinearClassifierNetwork(1, 2, bounds)
    node = classifier.hidden_layer.nodes[0]
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0
    return classifier


def network_with_hidden_thresholds(
    dimension: int,
    bounds: list[tuple[float, float]],
    thresholds: list[float],
    required_active: int | None = None,
) -> LinearClassifierNetwork:

    # zero input weights make each node's z() equal to its threshold alone, regardless of
    # state - so the thresholds directly pick each node's active/inactive status
    network = LinearClassifierNetwork(len(thresholds), dimension, bounds, required_active)
    for node, threshold in zip(network.hidden_layer.nodes, thresholds):
        node.update_input_weights([0.0] * dimension)
        node.threshold = threshold
    return network
