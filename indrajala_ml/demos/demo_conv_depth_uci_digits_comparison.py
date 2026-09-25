import random
import statistics
from typing import Any

from indrajala_ml.benchmark_sweep import run_parameter_sweep
from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.classifier_protocols import Example
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network

SIDE = 8
CLASS_COUNT = 10
DENSE_LAYER_SIZES = [32]
LEARNING_RATE = 0.5
EPOCHS = 20
SEEDS = list(range(8))

# every config shares the same dense tail, learning rate, epoch count, and (per seed) the same
# train/test split - only the convolutional front end differs. conv2-wide's second layer is
# widened to 16 channels so its total parameter count roughly matches conv1's (conv2's
# shrinking 4x4 output otherwise halves the dense layer's fan-in, confounding depth with size).
# conv1-pool and conv1-stride2 both downsample conv1's 6x6 output to 3x3, one by max pooling and
# one by striding the convolution itself - pooling's own contribution, at equal output size
CONV_CONFIGS: dict[str, list[ConvSpec | PoolSpec]] = {
    "conv1": [ConvSpec(kernel_size=3, channel_count=8)],
    "conv2": [ConvSpec(kernel_size=3, channel_count=8), ConvSpec(kernel_size=3, channel_count=8)],
    "conv2-wide": [ConvSpec(kernel_size=3, channel_count=8), ConvSpec(kernel_size=3, channel_count=16)],
    "conv1-pool": [ConvSpec(kernel_size=3, channel_count=8), PoolSpec(pool_size=2)],
    "conv1-stride2": [ConvSpec(kernel_size=3, channel_count=8, stride=2)],
}
CONFIGS = ["dense"] + list(CONV_CONFIGS)


def _build(config: str) -> MultiClassBackpropClassifierNetwork[Any]:
    if config == "dense":
        return MultiClassBackpropClassifierNetwork.randomized(
            DENSE_LAYER_SIZES, SIDE * SIDE, [(0.0, 1.0)] * (SIDE * SIDE), CLASS_COUNT
        )
    return ConvMultiClassBackpropClassifierNetwork.randomized(
        SIDE, SIDE, CONV_CONFIGS[config], DENSE_LAYER_SIZES, CLASS_COUNT
    )


def parameter_count(network: MultiClassBackpropClassifierNetwork[Any]) -> int:
    return sum(len(weights) + 1 for layer in network.snapshot() for weights, _bias in layer)


def run_one(dataset: list[Example[int]], config: str, seed: int) -> dict[str, float]:
    # module-level, for run_parameter_sweep's picklability requirement; the seed drives both
    # the split and the weight initialization, and is shared across configs, so results pair up
    # seed-by-seed
    train_data, test_data = split_train_test(dataset, test_fraction=0.2, seed=seed)
    random.seed(seed)
    student = _build(config)

    # accuracy only, no wall-clock: the pure-Python implementation is never used for timing
    # (see the README's Models section)
    result = train_linear_classifier_network(student, train_data, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    return {"seed": seed, "train": result.diagnostic.best_training_accuracy, "test": accuracy(student, test_data)}


def _mean_stdev(values: list[float]) -> str:
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main() -> None:

    print(
        "Does stacking a second convolutional layer help on 8x8 UCI digits? Two valid 3x3 convs "
        "take 8x8 -> 6x6 -> 4x4, and the second layer's receptive field (5x5) already covers most "
        "of the image, so it's a genuinely open question - a null is a real possibility. "
        f"{len(CONFIGS)} configs x {len(SEEDS)} seeds, each seed its own 80/20 split and init, "
        "shared across configs so differences pair up seed by seed. Takes roughly an hour on 8 "
        "logical cores."
    )
    print()

    dataset = load_digits_dataset()
    results = run_parameter_sweep(CONFIGS, SEEDS, run_one, shared_context=dataset)
    by_seed = {config: {r["seed"]: r for r in runs} for config, runs in results.items()}

    print("| config | params | best train accuracy | test accuracy |")
    print("|---|---|---|---|")
    for config in CONFIGS:
        runs = results[config]
        print(
            f"| {config} | {parameter_count(_build(config))} "
            f"| {_mean_stdev([r['train'] for r in runs])} "
            f"| {_mean_stdev([r['test'] for r in runs])} |"
        )
    print()

    print("Paired test-accuracy differences (same seed, same split), vs conv1:")
    for config in CONFIGS:
        if config == "conv1":
            continue
        diffs = [by_seed[config][seed]["test"] - by_seed["conv1"][seed]["test"] for seed in SEEDS]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        print(f"  {config} - conv1: {_mean_stdev(diffs)}  ({wins} better / {losses} worse of {len(SEEDS)})")


if __name__ == "__main__":
    main()
