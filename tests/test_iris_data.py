from indrajala_ml.iris_data import load_iris_dataset
from tests.helpers import approx

# split_train_test's own tests live in test_dataset_utils.py - see test_digits_data.py's own
# note on why it isn't re-tested per dataset here.


def test_load_iris_dataset_returns_the_full_bundled_dataset():

    dataset = load_iris_dataset()

    assert len(dataset) == 150
    assert all(len(state) == 4 for state, _ in dataset)
    assert all(0.0 <= value <= 1.0 for state, _ in dataset for value in state)
    assert all(label in (0, 1, 2) for _, label in dataset)

    # Fisher's Iris is exactly class-balanced: 50 setosa, 50 versicolor, 50 virginica
    labels = [label for _, label in dataset]
    assert labels.count(0) == 50
    assert labels.count(1) == 50
    assert labels.count(2) == 50


def test_load_iris_dataset_normalizes_against_the_dataset_own_min_max():

    dataset = load_iris_dataset()

    # the bounds are the dataset's own min/max, so every feature reaches both 0.0 and 1.0
    for dimension in range(4):
        values = [state[dimension] for state, _ in dataset]
        assert min(values) == approx(0.0, abs=1e-9)
        assert max(values) == approx(1.0, abs=1e-9)


def test_load_iris_dataset_decodes_the_well_known_first_row_correctly():

    dataset = load_iris_dataset()

    # Iris's well-known first row: sepal 5.1/3.5, petal 1.4/0.2, setosa
    state, label = dataset[0]
    assert label == 0
    assert state == approx((0.2222222222222222, 0.625, 0.06779661016949151, 0.041666666666666664))
