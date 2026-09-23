from indrajala_ml.dataset_utils import split_train_test  # re-exported - see that module's docstring

__all__ = ["load_iris_dataset", "split_train_test"]

# Fisher's Iris dataset's own known per-feature (sepal length, sepal width, petal length, petal
# width) min/max, in cm - computed once, directly, from sklearn.datasets.load_iris() (checked,
# not assumed), the same way digits_data.py's /16.0 is UCI digits' own known pixel max. Used to
# min-max normalize each feature to [0.0, 1.0] - unnormalized cm-scale values (up to 7.9) would
# defeat MultiClassBackpropClassifierNetwork.randomize()'s fan-in-aware weight scaling, which
# assumes roughly unit-scale inputs, exactly the reason digits_data.py normalizes its own pixels.
_FEATURE_MIN = (4.3, 2.0, 1.0, 0.1)
_FEATURE_MAX = (7.9, 4.4, 6.9, 2.5)


def load_iris_dataset(path: str = "data/iris/iris.csv") -> list[tuple[tuple[float, ...], int]]:
    """
    Parses data/iris/iris.csv - 150 lines, 5 comma-separated values each (4 real-valued
    measurements in cm - sepal length/width, petal length/width - then a label 0/1/2 for
    setosa/versicolor/virginica), no header, extracted once (offline) from
    sklearn.datasets.load_iris() in the source's own row order (50 setosa, 50 versicolor, 50
    virginica - not pre-shuffled; split_train_test does that at load time). Normalizes each
    feature to [0.0, 1.0] via min-max scaling against the dataset's own known min/max (see
    _FEATURE_MIN/_FEATURE_MAX above) - the same normalization convention
    digits_data.load_digits_dataset/mnist_data.load_mnist_dataset already establish, just
    min-max instead of divide-by-a-shared-max since each of Iris's 4 features has its own
    distinct scale, unlike a shared 0-16 pixel range.
    """

    dataset: list[tuple[tuple[float, ...], int]] = []

    with open(path) as f:
        for line in f:
            values = line.strip().split(",")
            assert len(values) == 5, f"expected 4 measurements + 1 label per line; got {len(values)} values"
            raw_features = tuple(float(v) for v in values[:4])
            normalized = tuple(
                (raw - lo) / (hi - lo) for raw, lo, hi in zip(raw_features, _FEATURE_MIN, _FEATURE_MAX)
            )
            label = int(values[4])
            dataset.append((normalized, label))

    return dataset
