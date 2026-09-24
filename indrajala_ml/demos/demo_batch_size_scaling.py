from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.mnist_data import load_mnist_dataset

# a reduced version of the batch-size-scaling study's sweep (indrajala_ml/batch_size_scaling.py):
# momentum 0.9, whose batch-32 rate the baseline sweep picked, with the 1-epoch warmup under which
# the rule held furthest there
BATCH_SIZES = [32, 128, 512, 1024]
BASE_RATE = 0.25
MOMENTUM = 0.9
WARMUP_EPOCHS = [0.0, 1.0]
SEEDS = [0, 1, 2]
EPOCHS = 3


def run(train_data: list, test_data: list, batch_sizes: list[int], seeds: list[int], epochs: int) -> dict:
    """{(batch_size, warmup_epochs): [train_and_evaluate result per seed]}, Rust, runs in series."""
    results = {}
    for batch_size in batch_sizes:
        rate = bss.scaled_learning_rate(BASE_RATE, batch_size)
        for warmup in WARMUP_EPOCHS:
            results[(batch_size, warmup)] = [
                bss.train_and_evaluate("rust", train_data, test_data, batch_size, rate, warmup, MOMENTUM, epochs, seed)
                for seed in seeds
            ]
    return results


def format_rows(results: dict, train_size: int) -> list[str]:
    rows = []
    for (batch_size, warmup), runs in results.items():
        finals = [run["test_accuracies"][-1] for run in runs]
        step_seconds = sorted(seconds for run in runs for seconds in run["step_seconds"])
        rows.append(
            f"{batch_size:>5} {bss.scaled_learning_rate(BASE_RATE, batch_size):>6g} "
            f"{bss.warmup_steps(warmup, train_size, batch_size):>7} {runs[0]['steps']:>6} "
            f"{sum(finals) / len(finals):>9.2%} {min(finals):>7.2%} {max(finals):>7.2%} "
            f"{step_seconds[len(step_seconds) // 2]:>12.2f}"
        )
    return rows


def main() -> None:

    print(
        "The linear learning-rate scaling rule (Goyal et al. 2017) on full MNIST, dense 784 -> 30 -> 10, "
        f"Rust. The rate at batch B is {BASE_RATE} * B / 32, momentum {MOMENTUM}, with no warmup and with "
        f"a linear warmup over the first epoch. {len(SEEDS)} seeds, {EPOCHS} epochs each; every batch size "
        "sees the same data per epoch, so larger batches take fewer steps. Test accuracy after the last "
        "epoch, and the median seconds per epoch spent in the learn_batch calls. Without warmup the scaled "
        "rate diverges from batch 128 up; with it, the rule holds to about batch 512. Takes a few minutes."
    )
    print()

    print(f"loading {bss.TRAIN_PATH} / {bss.TEST_PATH}...")
    train_data = load_mnist_dataset(bss.TRAIN_PATH)
    test_data = load_mnist_dataset(bss.TEST_PATH)
    print(f"loaded {len(train_data)} train / {len(test_data)} test examples")
    print()

    print(f"{'B':>5} {'rate':>6} {'warmup':>7} {'steps':>6} {'mean acc':>9} {'min':>7} {'max':>7} {'step s/epoch':>12}")
    for batch_size in BATCH_SIZES:
        results = run(train_data, test_data, [batch_size], SEEDS, EPOCHS)
        for row in format_rows(results, len(train_data)):
            print(row, flush=True)
    print()
    print("warmup is in steps (0: none). min and max are over seeds.")
    print()


if __name__ == "__main__":
    main()
