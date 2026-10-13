import random

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.networks.python.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.training.evaluate import class_balanced_disagreement_rate
from indrajala_ml.training.train import train_linear_classifier_network
from indrajala_ml.training.training_data import random_alternating_training_data


def test_training_of_cardinality_two_linear_classifier_reduces_disagreement():

    # the training data and the disagreement samples draw from one random.Random
    samples = random.Random(0)

    cardinality: int = 2
    dimension: int = 2
    l: float = 10.0
    bounds = square_bounds(l, dimension)

    reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, seed=1)
    training_data = random_alternating_training_data(400, reference, rng=samples)

    student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, seed=101)

    disagreement_before = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500, rng=samples)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)
    disagreement_after = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500, rng=samples)

    assert disagreement_after < disagreement_before
    assert disagreement_after < 0.15
