from indrajala_ml.dataset_utils import split_train_test  # re-exported - see that module's docstring

__all__ = ["load_iris_dataset", "split_train_test"]

# Iris's per-feature (sepal length, sepal width, petal length, petal width) min and max in cm,
# computed from sklearn.datasets.load_iris(), for min-max normalization to [0.0, 1.0]: values up
# to 7.9 would defeat fan-in-aware initialization, which assumes unit-scale inputs
_FEATURE_MIN = (4.3, 2.0, 1.0, 0.1)
_FEATURE_MAX = (7.9, 4.4, 6.9, 2.5)


def load_iris_dataset(path: str = "data/iris/iris.csv") -> list[tuple[tuple[float, ...], int]]:
    """
    Parses data/iris/iris.csv: 150 lines of 4 measurements in cm (sepal length/width, petal
    length/width) and a label 0/1/2 (setosa/versicolor/virginica), no header, extracted offline from
    sklearn.datasets.load_iris() in its order (50 of each class, unshuffled). Each feature is
    min-max normalized to [0.0, 1.0], since each has its own scale.
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
