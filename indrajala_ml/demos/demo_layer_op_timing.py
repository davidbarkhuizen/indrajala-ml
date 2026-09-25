# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer

CALLS = 300
LOOPS = 5
BATCH_SIZES = (1, 32, 512)
SEED = 0
LEARNING_RATE = 1e-6  # small, so repeated apply calls barely move W between timed calls

# (label, size, input_size): the dense layer after a ConvSpec(3, 8) layer on 28x28 input, and the
# dense production network 784 -> 30 -> 10
DENSE_SHAPES = [
    ("conv tail 32 x 5408", 32, 5408),
    ("dense 30 x 784", 30, 784),
    ("dense 10 x 30", 10, 30),
]
# (label, side): a ConvSpec(3, 8) layer on MNIST 28x28 and on UCI digits 8x8
CONV_SHAPES = [
    ("conv 28x28, 8 ch", 28),
    ("conv 8x8, 8 ch", 8),
]
CONV_KERNEL_SIZE = 3
CONV_CHANNELS = 8
# (label, side): PoolSpec(2) on those conv layers' 8-channel outputs, as in conv-pool-conv
POOL_SHAPES = [
    ("pool 26x26, 8 ch", 26),
    ("pool 6x6, 8 ch", 6),
]
POOL_SIZE = 2

BACKENDS = ("numpy", "rust")
DENSE_LAYERS = {"numpy": ArrayLayer, "rust": RustArrayLayer}
CONV_LAYERS = {"numpy": ConvArrayLayer, "rust": ConvRustArrayLayer}
POOL_LAYERS = {"numpy": MaxPoolArrayLayer, "rust": MaxPoolRustArrayLayer}


@dataclass(frozen=True)
class Case:
    """One table row: build(backend) returns a zero-argument call to time on that backend."""

    shape: str
    op: str
    batch: int | None  # None for a single-example op
    build: Callable[[str], Callable[[], object]]


@dataclass(frozen=True)
class Row:
    shape: str
    op: str
    batch: int | None
    calls: int
    numpy_us: float
    rust_us: float

    @property
    def ratio(self) -> float:
        return self.rust_us / self.numpy_us


# The cases pick their backend by name at runtime, so a backend's arrays and layers are typed Any
# here, the one place the name is resolved.


def _backend_array(backend: str, values: FloatArray) -> Any:
    return values.copy() if backend == "numpy" else pa.Array(values.tolist())


def _dense_layer(backend: str, size: int, input_size: int, rng: np.random.Generator) -> Any:
    layer = DENSE_LAYERS[backend](size, input_size)
    layer.W = _backend_array(backend, rng.uniform(-0.3, 0.3, size=(size, input_size)))
    layer.b = _backend_array(backend, rng.uniform(-0.3, 0.3, size=size))
    return layer


def dense_cases(label: str, size: int, input_size: int, batch_sizes: Sequence[int]) -> list[Case]:
    """
    Every ArrayLayer method at one (size, input_size) shape. Each build draws the same values from
    a fresh seeded generator, so both backends time identical inputs. 'hidden_delta' times the
    layer *below* this one (size input_size) calling compute_hidden_delta with this layer as
    next_layer, since that is where this layer's W is read. 'sgd step' is this layer's share of
    one single-example learn() step, sgd_step: fused on Rust, accumulate_gradient then
    apply_accumulated_gradient on numpy.
    """

    def single(op: str) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            rng = np.random.default_rng(SEED)
            layer = _dense_layer(backend, size, input_size, rng)
            x = _backend_array(backend, rng.uniform(0.0, 1.0, size=input_size))
            layer.forward(x)
            layer.delta = _backend_array(backend, rng.uniform(-0.1, 0.1, size=size))
            if op == "forward":
                return lambda: layer.forward(x)
            if op == "downstream":
                return layer.downstream
            if op == "hidden_delta":
                below = _dense_layer(backend, input_size, 1, rng)
                below.a = x
                return lambda: below.compute_hidden_delta(layer)
            if op == "accumulate_gradient":
                return lambda: layer.accumulate_gradient(x)
            if op == "apply_accumulated_gradient":
                return lambda: layer.apply_accumulated_gradient(LEARNING_RATE, 1)
            return lambda: layer.sgd_step(x, LEARNING_RATE)

        return build

    def batched(op: str, batch: int) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            rng = np.random.default_rng(SEED)
            layer = _dense_layer(backend, size, input_size, rng)
            X = _backend_array(backend, rng.uniform(0.0, 1.0, size=(batch, input_size)))
            layer.forward_batch(X)
            layer.delta_batch = _backend_array(backend, rng.uniform(-0.1, 0.1, size=(batch, size)))
            if op == "forward_batch":
                return lambda: layer.forward_batch(X)
            if op == "downstream_batch":
                return layer.downstream_batch
            if op == "hidden_delta_batch":
                below = _dense_layer(backend, input_size, 1, rng)
                below.A = X
                return lambda: below.compute_hidden_delta_batch(layer)
            return lambda: layer.accumulate_gradient_batch(X)

        return build

    single_ops = [
        "forward",
        "downstream",
        "hidden_delta",
        "accumulate_gradient",
        "apply_accumulated_gradient",
        "sgd step",
    ]
    batch_ops = ["forward_batch", "downstream_batch", "hidden_delta_batch", "accumulate_gradient_batch"]
    cases = [Case(label, op, None, single(op)) for op in single_ops]
    cases += [Case(label, op, batch, batched(op, batch)) for op in batch_ops for batch in batch_sizes]
    return cases


def conv_cases(label: str, side: int, batch_sizes: Sequence[int]) -> list[Case]:
    """
    ConvSpec(3, 8) on one side x side input channel: forward, downstream and accumulate_gradient,
    single-example and batched. The single-example ops include each backend's N = 1 wrapping.
    """

    def build_layer(backend: str, batch: int) -> tuple[Any, FloatArray, FloatArray]:
        rng = np.random.default_rng(SEED)
        layer: Any = CONV_LAYERS[backend](side, side, 1, CONV_KERNEL_SIZE, CONV_CHANNELS)
        layer.W = _backend_array(backend, rng.uniform(-0.3, 0.3, size=(layer.channel_count, layer.fan_in)))
        channel_count: int = layer.channel_count
        layer.b = _backend_array(backend, rng.uniform(-0.3, 0.3, size=channel_count))
        X = rng.uniform(0.0, 1.0, size=(batch, layer.input_size))
        delta = rng.uniform(-0.1, 0.1, size=(batch, layer.size))
        return layer, X, delta

    def single(op: str) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            layer, X, delta = build_layer(backend, 1)
            x = _backend_array(backend, X[0])
            layer.forward(x)
            layer.delta = _backend_array(backend, delta[0])
            if op == "forward":
                return lambda: layer.forward(x)
            if op == "downstream":
                return layer.downstream
            return lambda: layer.accumulate_gradient(x)

        return build

    def batched(op: str, batch: int) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            layer, X, delta = build_layer(backend, batch)
            X = _backend_array(backend, X)
            layer.forward_batch(X)
            layer.delta_batch = _backend_array(backend, delta)
            if op == "forward_batch":
                return lambda: layer.forward_batch(X)
            if op == "downstream_batch":
                return layer.downstream_batch
            return lambda: layer.accumulate_gradient_batch(X)

        return build

    cases = [Case(label, op, None, single(op)) for op in ["forward", "downstream", "accumulate_gradient"]]
    cases += [
        Case(label, op, batch, batched(op, batch))
        for op in ["forward_batch", "downstream_batch", "accumulate_gradient_batch"]
        for batch in batch_sizes
    ]
    return cases


def pool_cases(label: str, side: int, batch_sizes: Sequence[int]) -> list[Case]:
    """
    PoolSpec(2) on a side x side, 8-channel input: forward and downstream, single-example and
    batched. The input is ReLU-like (about half exact zeros), as after a conv layer, so windows
    tie as they do in training.
    """

    def build_layer(backend: str, batch: int) -> tuple[Any, FloatArray, FloatArray]:
        rng = np.random.default_rng(SEED)
        layer: Any = POOL_LAYERS[backend](side, side, CONV_CHANNELS, POOL_SIZE)
        X = np.maximum(rng.uniform(-1.0, 1.0, size=(batch, layer.input_size)), 0.0)
        delta = rng.uniform(-0.1, 0.1, size=(batch, layer.size))
        return layer, X, delta

    def single(op: str) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            layer, X, delta = build_layer(backend, 1)
            x = _backend_array(backend, X[0])
            layer.forward(x)
            layer.delta = _backend_array(backend, delta[0])
            return (lambda: layer.forward(x)) if op == "forward" else layer.downstream

        return build

    def batched(op: str, batch: int) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            layer, X, delta = build_layer(backend, batch)
            X = _backend_array(backend, X)
            layer.forward_batch(X)
            layer.delta_batch = _backend_array(backend, delta)
            return (lambda: layer.forward_batch(X)) if op == "forward_batch" else layer.downstream_batch

        return build

    cases = [Case(label, op, None, single(op)) for op in ["forward", "downstream"]]
    cases += [
        Case(label, op, batch, batched(op, batch))
        for op in ["forward_batch", "downstream_batch"]
        for batch in batch_sizes
    ]
    return cases


def all_cases(batch_sizes: Sequence[int] = BATCH_SIZES) -> list[Case]:
    cases: list[Case] = []
    for label, size, input_size in DENSE_SHAPES:
        cases += dense_cases(label, size, input_size, batch_sizes)
    for label, side in CONV_SHAPES:
        cases += conv_cases(label, side, batch_sizes)
    for label, side in POOL_SHAPES:
        cases += pool_cases(label, side, batch_sizes)
    return cases


def calls_per_loop(batch: int | None, calls: int) -> int:
    # a batch call does about `batch` examples' work, so larger batches get proportionally fewer
    # calls per loop (never fewer than 10), keeping every row's loop at a similar wall-clock cost
    if batch is None or batch <= 1:
        return calls
    return max(10, calls // batch)


def time_case(case: Case, calls: int, loops: int) -> Row:
    """
    Median over `loops` of the mean microseconds per call across a loop of `calls` calls, the
    backends interleaved loop by loop so drift in machine load hits both alike. One untimed
    warm-up call per backend first.
    """
    fns = {backend: case.build(backend) for backend in BACKENDS}
    for fn in fns.values():
        fn()
    count = calls_per_loop(case.batch, calls)
    per_call: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    for _loop in range(loops):
        for backend, fn in fns.items():
            start = time.perf_counter()
            for _call in range(count):
                fn()
            per_call[backend].append((time.perf_counter() - start) / count * 1e6)
    return Row(
        case.shape,
        case.op,
        case.batch,
        count,
        statistics.median(per_call["numpy"]),
        statistics.median(per_call["rust"]),
    )


def format_row(row: Row) -> str:
    batch = "-" if row.batch is None else str(row.batch)
    return (
        f"{row.shape:<20} {row.op:<28} {batch:>5} {row.calls:>5} "
        f"{row.numpy_us:>11.1f} {row.rust_us:>11.1f} {row.ratio:>6.2f}"
    )


HEADER = f"{'shape':<20} {'op':<28} {'batch':>5} {'calls':>5} {'numpy us':>11} {'rust us':>11} {'ratio':>6}"


def main() -> None:

    print(
        "Per-op layer timing, numpy vs Rust. Each cell is the median over "
        f"{LOOPS} loops of microseconds per call; a loop is {CALLS} calls for a single-example op "
        f"and {CALLS} // batch (at least 10) for a batch op, the backends interleaved loop by loop. "
        "Ratio is Rust / numpy (below 1 means Rust is faster). Pure Python is never timed."
    )
    print()
    print(HEADER)
    shape = None
    for case in all_cases():
        if case.shape != shape and shape is not None:
            print()
        shape = case.shape
        print(format_row(time_case(case, CALLS, LOOPS)), flush=True)


if __name__ == "__main__":
    main()
