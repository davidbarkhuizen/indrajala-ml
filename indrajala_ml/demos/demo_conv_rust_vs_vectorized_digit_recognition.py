import cProfile
import pstats
import random
import statistics

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.demos.timing import timed_call, timed_train
from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_backprop_network_mini_batch

CLASS_COUNT = 10
DENSE_LAYER_SIZES = [32]
LEARNING_RATE = 0.5
BATCH_SIZE = 32
REPEATS = 5
SEED = 0

ARCHITECTURES: dict[str, list[ConvSpec | PoolSpec]] = {
    "conv": [ConvSpec(3, 8)],
    "conv-pool-conv": [ConvSpec(3, 8), PoolSpec(2), ConvSpec(3, 8)],
    "conv-conv-stride2": [ConvSpec(3, 8), ConvSpec(3, 8, stride=2)],
}
TRAINERS = ["single-example", f"mini-batch ({BATCH_SIZE})"]
BACKENDS = {
    "numpy": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": ConvRustArrayMultiClassBackpropClassifierNetwork,
}

MNIST_TRAIN_PATH = "data/mnist/mnist-train.bin"
MNIST_TEST_PATH = "data/mnist/mnist-test.bin"
MNIST_TRAIN_LIMIT = 2000
MNIST_TEST_LIMIT = 500


def load_datasets() -> dict[str, tuple[int, list, list, int]]:
    """name -> (side, train_data, test_data, epochs)."""
    train_uci, test_uci = split_train_test(load_digits_dataset(), test_fraction=0.2, seed=SEED)
    train_mnist = load_mnist_dataset(MNIST_TRAIN_PATH, limit=MNIST_TRAIN_LIMIT)
    test_mnist = load_mnist_dataset(MNIST_TEST_PATH, limit=MNIST_TEST_LIMIT)
    return {
        "UCI digits 8x8": (8, train_uci, test_uci, 2),
        f"MNIST 28x28 ({MNIST_TRAIN_LIMIT} train)": (28, train_mnist, test_mnist, 1),
    }


def initial_snapshot(side: int, conv_specs: list) -> list[tuple]:
    # drawn once, by the numpy network, and restored into both backends - identical starting
    # weights, since the two backends' RNGs aren't comparable
    np.random.seed(SEED)
    return ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        side, side, conv_specs, DENSE_LAYER_SIZES, CLASS_COUNT
    ).snapshot()


def train_once(backend: str, trainer: str, side: int, conv_specs: list, snapshot, train_data, epochs: int):
    """(network, ConvergenceSeries, elapsed seconds) for one timed training run from snapshot."""
    network = BACKENDS[backend](side, side, conv_specs, DENSE_LAYER_SIZES, CLASS_COUNT)
    network.restore(snapshot)
    random.seed(SEED)  # the mini-batch trainer's shuffle order, identical for every run
    if trainer == TRAINERS[0]:
        result, elapsed = timed_train(network, train_data, learning_rate=LEARNING_RATE, epochs=epochs)
    else:
        result, elapsed = timed_call(
            train_backprop_network_mini_batch,
            network,
            train_data,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=epochs,
        )
    return network, result, elapsed


def compare(side: int, conv_specs: list, trainer: str, train_data, test_data, epochs: int, repeats: int) -> dict:
    """Times both backends from identical initial weights, interleaving their runs so any drift
    in machine load hits both alike; reports median seconds and each backend's accuracy."""
    snapshot = initial_snapshot(side, conv_specs)
    elapsed: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    outcome: dict[str, tuple] = {}
    for _repeat in range(repeats):
        for backend in BACKENDS:
            network, result, seconds = train_once(backend, trainer, side, conv_specs, snapshot, train_data, epochs)
            elapsed[backend].append(seconds)
            outcome[backend] = (network, result.diagnostic.best_training_accuracy, accuracy(network, test_data))

    numpy_network, rust_network = outcome["numpy"][0], outcome["rust"][0]
    # the control for "agree": numpy against itself from the same weights with one moved by 1
    # ULP. Long single-example runs are chaotically sensitive to rounding, so the backends'
    # different summation orders end at different networks, as the nudge does. Rust agreeing
    # with numpy about as often as the control is the parity evidence; the step-by-step parity
    # tests pin per-step agreement (within ~1e-15).
    perturbed_network, _result, _seconds = train_once(
        "numpy", trainer, side, conv_specs, _nudged_by_one_ulp(snapshot), train_data, epochs
    )
    agreement = _agreement(numpy_network, rust_network, test_data)
    medians = {backend: statistics.median(times) for backend, times in elapsed.items()}
    return {
        "median": medians,
        "ratio": medians["rust"] / medians["numpy"],
        "train_accuracy": {backend: outcome[backend][1] for backend in BACKENDS},
        "test_accuracy": {backend: outcome[backend][2] for backend in BACKENDS},
        "prediction_agreement": agreement,
        "control_agreement": _agreement(numpy_network, perturbed_network, test_data),
    }


def _agreement(first, second, test_data) -> float:
    return statistics.mean(first.classify_state(state) == second.classify_state(state) for state, _label in test_data)


def _nudged_by_one_ulp(snapshot: list[tuple]) -> list[tuple]:
    W, b = snapshot[0]
    W = W.copy()
    W[0, 0] = np.nextafter(W[0, 0], np.inf)
    return [(W, b), *snapshot[1:]]


def wrapping_cost(side: int, rows: list, repeats: int) -> dict[str, tuple[float, float]]:
    """
    backend -> (single-example microseconds per example, batch-of-one microseconds per example)
    for one ConvSpec(3, 8) layer's forward + downstream + accumulate_gradient. Both paths run
    the same batch op on the same example; the only difference is the single-example path's
    N = 1 wrapping (a reshape in Rust, a newaxis view and [0] in numpy), so the gap is what that
    wrapping costs. Inputs and deltas are converted to each backend's arrays before timing.
    """
    spec = ARCHITECTURES["conv"][0]
    results = {}
    for backend, network_cls in BACKENDS.items():
        layer = network_cls(side, side, [spec], DENSE_LAYER_SIZES, CLASS_COUNT).conv_layers[0]
        rng = np.random.default_rng(SEED)
        layer.W = _backend_array(backend, rng.uniform(-0.3, 0.3, size=(layer.channel_count, layer.fan_in)))
        deltas = rng.uniform(-1.0, 1.0, size=(len(rows), layer.size))
        states = np.array([state for state, _label in rows])
        single = [(_backend_array(backend, x), _backend_array(backend, d)) for x, d in zip(states, deltas)]
        batched = [
            (_backend_array(backend, x[None, :]), _backend_array(backend, d[None, :])) for x, d in zip(states, deltas)
        ]

        def run_single():
            for x, delta in single:
                layer.forward(x)
                layer.delta = delta
                layer.downstream()
                layer.accumulate_gradient(x)

        def run_batched():
            for X, delta_batch in batched:
                layer.forward_batch(X)
                layer.delta_batch = delta_batch
                layer.downstream_batch()
                layer.accumulate_gradient_batch(X)

        single_seconds, batched_seconds = [], []
        for _repeat in range(repeats):
            single_seconds.append(timed_call(run_single)[1])
            batched_seconds.append(timed_call(run_batched)[1])
        per_example = 1e6 / len(rows)
        results[backend] = (
            statistics.median(single_seconds) * per_example,
            statistics.median(batched_seconds) * per_example,
        )
    return results


def _backend_array(backend: str, values: np.ndarray):
    return values.copy() if backend == "numpy" else pa.Array(values.tolist())


def rust_op_breakdown(side: int, conv_specs: list, trainer: str, train_data, epochs: int) -> tuple[float, list]:
    """
    cProfile over one Rust training run: (total profiled seconds, [(op name, seconds, calls)])
    for every indrajala_math_rust call, largest first. cProfile's own per-call overhead inflates
    the Python side, so these are shares of a profiled run, not of the timed runs above.
    """
    snapshot = initial_snapshot(side, conv_specs)
    profiler = cProfile.Profile()
    profiler.enable()
    train_once("rust", trainer, side, conv_specs, snapshot, train_data, epochs)
    profiler.disable()

    stats = pstats.Stats(profiler)
    ops = []
    for (_file, _line, name), (_calls, total_calls, own_seconds, _cumulative, _callers) in stats.stats.items():
        op = _rust_op_name(name)
        if op is not None:
            ops.append((op, own_seconds, total_calls))
    return stats.total_tt, sorted(ops, key=lambda op: op[1], reverse=True)


def _rust_op_name(profiler_name: str) -> str | None:
    # "<built-in method indrajala_math_rust.indrajala_math_rust.conv_forward_batch>" -> the function;
    # "<method 'reshape' of 'builtins.Array' objects>" -> "Array.reshape"; anything else -> None
    if profiler_name.startswith("<built-in method indrajala_math_rust."):
        return profiler_name.rstrip(">").split(".")[-1]
    if profiler_name.startswith("<method '") and "'builtins.Array'" in profiler_name:
        return "Array." + profiler_name.split("'")[1]
    return None


def main() -> None:

    print(
        "numpy vs Rust conv networks, timed from identical initial weights. Each cell is the "
        f"median of {REPEATS} interleaved runs; ratio is Rust / numpy (below 1 means Rust is "
        "faster). Test accuracy is the final network's. 'agree' is how often the numpy and Rust "
        "networks classify a test row the same; '1-ulp ctrl' is the same for numpy against numpy "
        "started with one weight nudged by 1 ULP - long single-example runs are chaotically "
        "sensitive to rounding, so this control, not 100%, is the bar for 'agree'. The mini-batch "
        "runs barely train at this learning rate and epoch budget, so their accuracy match says "
        "little. Pure Python is never timed."
    )
    print()

    datasets = load_datasets()
    for name, (side, train_data, test_data, epochs) in datasets.items():
        print(f"== {name}: {len(train_data)} train / {len(test_data)} test, {epochs} epoch(s)")
        print(
            f"{'architecture':<18} {'trainer':<16} {'numpy s':>8} {'rust s':>8} {'ratio':>6}  "
            f"{'test acc numpy/rust':>20}  {'agree':>6}  {'1-ulp ctrl':>10}"
        )
        for architecture, conv_specs in ARCHITECTURES.items():
            for trainer in TRAINERS:
                row = compare(side, conv_specs, trainer, train_data, test_data, epochs, REPEATS)
                median = row["median"]
                print(
                    f"{architecture:<18} {trainer:<16} {median['numpy']:>8.2f} {median['rust']:>8.2f} "
                    f"{row['ratio']:>6.2f}  "
                    f"{row['test_accuracy']['numpy']:>9.4f}/{row['test_accuracy']['rust']:<10.4f}  "
                    f"{row['prediction_agreement']:>6.1%}  {row['control_agreement']:>10.1%}"
                )
        print()

    print("== N = 1 wrapping cost: one ConvSpec(3, 8) layer, forward + downstream + accumulate_gradient")
    print(f"{'dataset':<28} {'backend':<6} {'single us':>10} {'batch-of-1 us':>14} {'overhead':>9}")
    for name, (side, train_data, _test_data, _epochs) in datasets.items():
        for backend, (single_us, batched_us) in wrapping_cost(side, train_data[:500], REPEATS).items():
            print(f"{name:<28} {backend:<6} {single_us:>10.1f} {batched_us:>14.1f} {single_us / batched_us - 1:>9.1%}")
    print()

    side, train_data, _test_data, epochs = datasets[f"MNIST 28x28 ({MNIST_TRAIN_LIMIT} train)"]
    for trainer in TRAINERS:
        total, ops = rust_op_breakdown(side, ARCHITECTURES["conv-pool-conv"], trainer, train_data, epochs)
        print(f"== Rust time by op (cProfile, MNIST conv-pool-conv, {trainer}; {total:.1f}s profiled)")
        for op, seconds, calls in ops[:10]:
            print(f"  {op:<36} {seconds:>7.2f}s {seconds / total:>6.1%}  {calls:>8} calls")
        print()


if __name__ == "__main__":
    main()
