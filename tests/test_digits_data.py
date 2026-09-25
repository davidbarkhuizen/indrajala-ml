from indrajala_ml.digits_data import load_digits_dataset

# split_train_test's own tests live in test_dataset_utils.py - it's a shared utility
# (indrajala_ml/dataset_utils.py), re-exported here for callers' convenience, not
# digits-specific behavior worth re-testing per dataset.


def test_load_digits_dataset_returns_the_full_bundled_dataset():

    dataset = load_digits_dataset()

    assert len(dataset) == 1797
    assert all(len(state) == 64 for state, _ in dataset)
    assert all(0.0 <= value <= 1.0 for state, _ in dataset for value in state)
    assert all(0 <= label <= 9 for _, label in dataset)

    # not every state is the same, and not every pixel is at the normalized extreme - a
    # sanity check that parsing/normalization actually did something, not just returned zeros
    assert len({label for _, label in dataset}) == 10
    assert any(value not in (0.0, 1.0) for state, _ in dataset for value in state)
