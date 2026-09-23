import numpy as np

from indrajala_ml.demos import demo_conv_rust_vs_vectorized_digit_recognition as demo
from indrajala_ml.demos.registry import DEMOS
from indrajala_ml.digits_data import load_digits_dataset


def test_is_registered():

    assert demo.__name__ in {info.module for info in DEMOS}


def test_both_backends_start_from_the_same_weights_and_reach_the_same_accuracy():

    # timings aren't asserted - only that the comparison is like for like: identical initial
    # weights, and (up to the backends' dot-product grouping) identical training outcomes
    rows = load_digits_dataset()
    train_data, test_data = rows[:80], rows[80:120]
    for architecture, conv_specs in demo.ARCHITECTURES.items():
        for trainer in demo.TRAINERS:
            row = demo.compare(8, conv_specs, trainer, train_data, test_data, epochs=1, repeats=1)
            assert row["train_accuracy"]["numpy"] == row["train_accuracy"]["rust"], (architecture, trainer)
            assert row["test_accuracy"]["numpy"] == row["test_accuracy"]["rust"], (architecture, trainer)
            assert row["prediction_agreement"] == 1.0
            assert 0.0 <= row["control_agreement"] <= 1.0
            assert row["median"]["numpy"] > 0.0 and row["median"]["rust"] > 0.0


def test_wrapping_cost_and_op_breakdown_run_on_both_backends():

    rows = load_digits_dataset()[:20]
    costs = demo.wrapping_cost(8, rows, repeats=1)
    assert set(costs) == {"numpy", "rust"}
    assert all(single > 0.0 and batched > 0.0 for single, batched in costs.values())

    total, ops = demo.rust_op_breakdown(8, demo.ARCHITECTURES["conv-pool-conv"], demo.TRAINERS[1], rows, epochs=1)
    names = {name for name, _seconds, _calls in ops}
    assert {"conv_forward_batch", "conv_downstream_batch", "max_pool_forward_batch"} <= names
    assert 0.0 < sum(seconds for _name, seconds, _calls in ops) <= total


def test_rust_op_names_are_parsed_from_profiler_entries():

    assert demo._rust_op_name("<built-in method indrajala_math_rust.indrajala_math_rust.conv_forward_batch>") == (
        "conv_forward_batch"
    )
    assert demo._rust_op_name("<method 'reshape' of 'builtins.Array' objects>") == "Array.reshape"
    assert demo._rust_op_name("<built-in method builtins.len>") is None


def test_the_one_ulp_control_moves_exactly_one_weight_by_one_ulp():

    snapshot = demo.initial_snapshot(8, demo.ARCHITECTURES["conv"])
    nudged = demo._nudged_by_one_ulp(snapshot)
    W, W_nudged = snapshot[0][0], nudged[0][0]
    assert W_nudged[0, 0] == np.nextafter(W[0, 0], np.inf) and W_nudged[0, 0] != W[0, 0]
    W_nudged[0, 0] = W[0, 0]
    np.testing.assert_array_equal(W_nudged, W)
    assert all(a is b for a, b in zip(snapshot[1:], nudged[1:]))
