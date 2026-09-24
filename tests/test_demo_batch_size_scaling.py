from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.demos import demo_batch_size_scaling as demo
from indrajala_ml.demos.registry import DEMOS
from indrajala_ml.mnist_data import load_mnist_dataset


def test_is_registered():

    assert demo.__name__ in {info.module for info in DEMOS}


def test_runs_on_a_small_subset_with_the_scaled_rate():

    # accuracy isn't pinned - only that every cell runs, reports every epoch, and uses the
    # linear scaling rule's rate
    train_data = load_mnist_dataset(bss.TRAIN_PATH, limit=300)
    test_data = load_mnist_dataset(bss.TEST_PATH, limit=100)

    results = demo.run(train_data, test_data, batch_sizes=[32, 64], seeds=[0, 1], epochs=2)

    assert set(results) == {(32, 0.0), (32, 1.0), (64, 0.0), (64, 1.0)}
    for (batch_size, _warmup), runs in results.items():
        assert len(runs) == 2
        assert all(len(run["test_accuracies"]) == 2 for run in runs)
        assert runs[0]["steps"] == 2 * -(-300 // batch_size)

    rows = demo.format_rows(results, len(train_data))
    assert len(rows) == 4
    assert rows[0].split()[:3] == ["32", "0.25", "0"]
    assert rows[1].split()[:3] == ["32", "0.25", "10"]  # ceil(1 epoch * 300 / 32)
    assert rows[2].split()[:2] == ["64", "0.5"]


def test_scaled_rate_is_base_rate_times_batch_over_32():

    for batch_size in demo.BATCH_SIZES:
        assert bss.scaled_learning_rate(demo.BASE_RATE, batch_size) == demo.BASE_RATE * batch_size / 32
