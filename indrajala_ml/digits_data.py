from indrajala_ml.dataset_utils import split_train_test  # re-exported - see that module's docstring

__all__ = ["load_digits_dataset", "split_train_test"]


def load_digits_dataset(path: str = "data/digits/digits.csv") -> list[tuple[tuple[float, ...], int]]:
    """
    Parses data/digits/digits.csv - 1797 lines, 65 comma-separated integers each (64 pixel
    values 0-16, then a label 0-9), no header, extracted once (offline) from the UCI ML
    hand-written digits dataset. Normalizes each pixel to [0.0, 1.0] (divide by 16.0, the
    known max) - StateLayer/BackpropNode have no built-in normalization, and unnormalized
    0-16 inputs would defeat MultiClassBackpropClassifierNetwork.randomize()'s fan-in-aware
    weight scaling, which assumes roughly unit-scale inputs.
    """

    dataset: list[tuple[tuple[float, ...], int]] = []

    with open(path) as f:
        for line in f:
            values = [int(x) for x in line.strip().split(",")]
            assert len(values) == 65, f"expected 64 pixels + 1 label per line; got {len(values)} values"
            pixels = tuple(value / 16.0 for value in values[:64])
            label = values[64]
            dataset.append((pixels, label))

    return dataset
