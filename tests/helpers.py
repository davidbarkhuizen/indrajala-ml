import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape, ArraySingleOutputShape
from indrajala_ml.model.array_protocols import ArrayBackend, BackendArray, WeightedArrayLayer
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import BackpropNetworkBase
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
from indrajala_ml.model.layer_specs import InputShape, LayerSpec
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.python_optimizer import PythonOptimizer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from indrajala_ml.pcg64 import default_rng

# conftest's `backend` fixture: either array backend, NUMPY or RUST
Backend = ArrayBackend[Any]
# conftest's `implementation` fixture: an array backend's name, or "python" for the pure-Python networks
Implementation = Literal["numpy", "rust", "python"]
# a backend's array constructor on nested lists: np.array (numpy) or pa.Array (Rust)
Wrap = Callable[[Any], Any]


class _Snapshottable(Protocol):
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...


class _SavableMultiClass(Protocol):
    def save(self, path: str) -> None: ...

    def snapshot(self) -> Any: ...

    def classify_state(self, state: State) -> int: ...

    def predict_probabilities(self, state: State) -> list[float]: ...


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


def all_subclasses[ClassT](cls: type[ClassT]) -> Iterator[type[ClassT]]:
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


def bits(value: Any) -> Any:
    """
    Every float in value as float.hex, through either backend's arrays, lists, tuples and dicts:
    two values' bits are equal exactly when their floats are, bit for bit (0.0 and -0.0 differ,
    NaNs match), and pytest shows a readable diff. Tuples become lists and dict keys strings, as in
    a JSON file of bits (tests/saved_model_fixtures.py's).
    """
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return bits(to_list())
    if isinstance(value, list | tuple):
        return [bits(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    if isinstance(value, dict):
        return {str(key): bits(item) for key, item in cast("dict[Any, Any]", value).items()}
    if isinstance(value, float):
        return value.hex()
    return value


def randomized(
    implementation: Implementation,
    input_shape: InputShape,
    specs: list[LayerSpec],
    rule: UpdateRule | None = None,
    seed: int = 3,
    shape: Literal["multiclass", "single_output"] = "multiclass",
) -> Any:
    """A network of specs on implementation, of shape, randomized from seed."""
    rule = SGD() if rule is None else rule
    if implementation == "python":
        cls = (
            SequentialMultiClassBackpropClassifierNetwork
            if shape == "multiclass"
            else SequentialBackpropClassifierNetwork
        )
        built: Any = cls(input_shape, specs, rule)
        built.rng = default_rng(seed)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        built = SequentialArrayNetwork(input_shape, specs, rule, shape=shape, backend=backend)
        built.rng = backend.default_rng(seed)
    built.randomize()
    return built


def split[StateT](batch: Sequence[tuple[StateT, int]]) -> tuple[list[StateT], list[int]]:
    """A batch of (state, label) pairs as its states and its labels."""
    return [state for state, _ in batch], [label for _, label in batch]


def max_relative_gap(expected: Sequence[Sequence[Any]], actual: Sequence[Sequence[Any]]) -> float:
    """The largest |actual - expected| / |expected| over two snapshots' arrays, either backend's."""
    gap = 0.0
    for expected_entry, actual_entry in zip(expected, actual, strict=True):
        for expected_array, actual_array in zip(expected_entry, actual_entry, strict=True):
            want = to_numpy(expected_array)
            gap = max(gap, float((np.abs(to_numpy(actual_array) - want) / np.abs(want)).max()))
    return gap


def exp_by_math(values: FloatArray) -> FloatArray:
    """np.exp as math.exp, elementwise, so a scalar transcription computes the same bits."""
    return np.vectorize(math.exp, otypes=[np.float64])(values)


def exp_by_crate(values: FloatArray) -> FloatArray:
    """np.exp as the crate's exp (Rust's f64::exp), elementwise."""
    return np.array(pa.exp(pa.Array(values.reshape(-1).tolist())).tolist()).reshape(values.shape)


def sigmoid_by(exp: Callable[[FloatArray], FloatArray]) -> Callable[[FloatArray], FloatArray]:
    """array_layer.sigmoid, 1/(1+e^-z), with exp in place of np.exp."""

    def sigmoid(z: FloatArray) -> FloatArray:
        return 1.0 / (1.0 + exp(-z))

    return sigmoid


def patching(module: ModuleType, name: str, replacement: object) -> Callable[[pytest.MonkeyPatch], None]:
    """
    A fixture that sets module's name to replacement for one test: how a test swaps a numpy layer's
    exp (or the sigmoid on it) for math.exp or the crate's. exp isn't correctly rounded: np.exp
    picks its implementation by CPU, and can differ from math.exp and Rust's f64::exp in the last
    bit, which every later value then carries. Everything but exp is then compared by bits.
    """

    @pytest.fixture
    def fixture(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, name, replacement)

    return fixture


class WeightSets:
    """
    A stand-in pure-Python layer over loose BackpropNodes or ConvKernels, so LayerOptimizer can
    step them as a network steps a layer's.
    """

    def __init__(self, *weight_sets: Any) -> None:
        self._weight_sets = list(weight_sets)

    def weight_sets(self) -> list[Any]:
        return self._weight_sets


class LayerOptimizer:
    """
    One layer updated as a network updates it: by the pure-Python optimizer (python_optimizer.py)
    for a pure-Python layer or WeightSets, else by its backend's optimizer (optimizers.py), with
    the layer at index 0 and one begin_step per update, so a layer-level test can drive an update
    rule on its own.
    """

    def __init__(self, layer: Any, rule: UpdateRule | None = None) -> None:
        # rule None: SGD
        self.layer = layer
        if hasattr(layer, "weight_sets"):
            self.optimizer: Any = PythonOptimizer(rule or SGD())
        else:
            self.optimizer = (RUST if isinstance(layer.W, pa.Array) else NUMPY).optimizer(rule or SGD())

    def apply(self, learning_rate: float, batch_size: int) -> None:
        self.optimizer.begin_step()
        self.optimizer.apply(0, self.layer, learning_rate, batch_size)

    def step_single(self, input_activation: Any, learning_rate: float) -> None:
        # an array layer's single-example step
        self.optimizer.begin_step()
        self.optimizer.step_single(0, self.layer, input_activation, learning_rate)

    def apply_single(self, *nodes: Any) -> Callable[[float], None]:
        # a pure-Python single-example step over nodes: each accumulates its delta, then one apply
        # at batch_size=1, as BackpropNetworkBase._apply_gradients steps a layer
        def step(learning_rate: float) -> None:
            for node in nodes:
                node.accumulate_gradient()
            self.apply(learning_rate, 1)

        return step

    @property
    def state(self) -> list[Any]:
        # the rule's state for the layer: an array layer's momentum [velocity_W, velocity_b] or
        # Adam [m_W, v_W, m_b, v_b]; per pure-Python weight set, its (weight lists, bias values)
        return self.optimizer._state[0]


def array_layer_like(layer_cls: Callable[[int, int], Any], layer: BackpropLayer, backend: Backend) -> Any:
    """A layer_cls array layer on backend with layer's weights and biases: a dense pure-Python layer
    (a BackpropLayer, ReLU, softmax or cross-entropy one) as its array counterpart."""
    array_layer = layer_cls(layer.size, len(layer.input_layer.nodes))
    snapshot = layer.snapshot_state()
    array_layer.W = backend.owned([weights for weights, _bias in snapshot])
    array_layer.b = backend.owned([bias for _weights, bias in snapshot])
    return array_layer


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


def assert_save_and_load_round_trip[SavableT: _SavableMultiClass](
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


REFERENCE_CLS = {
    "multiclass": SequentialMultiClassBackpropClassifierNetwork,
    "single_output": SequentialBackpropClassifierNetwork,
}


def reference_network(
    input_shape: InputShape,
    specs: Sequence[LayerSpec],
    rule: UpdateRule,
    shape: Literal["multiclass", "single_output"] = "multiclass",
) -> Any:
    """
    The pure-Python network of specs and rule (sequential_backprop_network.py): the parity
    reference for an array network built from the same specs and rule, preset or sequential.
    """
    return REFERENCE_CLS[shape](input_shape, specs, rule)


def dense_reference(
    rng: random.Random,
    array_network: ArrayNetworkBase[Any],
    specs: Sequence[LayerSpec],
    rule: UpdateRule,
    wrap: Wrap,
    dimension: int,
    shape: Literal["multiclass", "single_output"] = "multiclass",
) -> BackpropNetworkBase[Any]:
    """
    reference_network for array_network, a dense network of specs and rule, over a flat input,
    with identical injected weights (inject_matching_weights; the backends' RNGs aren't comparable
    with Python's random, so randomize() isn't used).
    """
    reference = reference_network((dimension,), specs, rule, shape)
    inject_matching_weights(rng, reference, array_network, wrap, dimension)
    return reference


def conv_reference(
    rng: random.Random,
    array_network: ArrayNetworkBase[Any],
    input_shape: InputShape,
    specs: Sequence[LayerSpec],
    rule: UpdateRule,
    wrap: Wrap,
) -> SequentialMultiClassBackpropClassifierNetwork:
    """
    reference_network for array_network, a conv network (numpy or Rust, with `wrap` its backend's
    array constructor) of specs and rule, with identical injected weights, conv kernels included (a
    conv W's row c is kernel c's weights). Pool layers have none.
    """
    reference = reference_network(input_shape, specs, rule)
    for node_layer, array_layer in zip(reference.trainable_layers, array_network.layers):
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
    return reference


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
    node_network: BackpropNetworkBase[Any],
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
    rng into the numpy network, then restored into the Rust one from its snapshot.
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
    array_network_cls: Any, layer_sizes: list[int], dimension: int, **hyperparameters: float
) -> None:
    """
    The single-output analogue of assert_array_network_snapshot_restore_round_trip above, for
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork and their presets (no
    class_count argument; the hyperparameters are keyword-only).
    """
    network = array_network_cls.randomized(layer_sizes, dimension, **hyperparameters)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension, **hyperparameters)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_single_output_array_network_save_load_round_trip[SingleOutputT: ArraySingleOutputShape[Any]](
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


def assert_array_network_save_load_round_trip[MultiClassT: ArrayMultiClassShape[Any]](
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
