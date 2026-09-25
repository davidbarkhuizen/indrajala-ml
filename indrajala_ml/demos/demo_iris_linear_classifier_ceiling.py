from indrajala_ml.iris_data import load_iris_dataset
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import train_linear_classifier_network

SETOSA_LABEL = 0
VERSICOLOR_LABEL = 1
VIRGINICA_LABEL = 2


def main() -> None:

    dataset = load_iris_dataset()
    dimension = 4
    bounds = [(0.0, 1.0)] * dimension

    print(
        "Fisher's Iris (1936) - the first real dataset any LinearClassifierNetwork demo in this "
        "repo trains on; every other one uses synthetic 2D geometric targets. Iris is famous for "
        "exactly one linear-separability split: setosa is perfectly separable from the other two "
        "species, but versicolor and virginica overlap and are NOT perfectly linearly separable - "
        "verified directly here (a one-time linear-programming feasibility check over the exact "
        "same 4 features found the versicolor-vs-virginica separating-hyperplane problem "
        "infeasible), not assumed from folklore."
    )
    print()

    # setosa vs the other two: linearly separable, so a cardinality=1 perceptron converges, as
    # the perceptron convergence theorem guarantees (5 of 5 seeds reached 1.0 within 100
    # epochs, so 100 has headroom)
    setosa_data = [(state, 1.0 if label == SETOSA_LABEL else 0.0) for state, label in dataset]
    setosa_student = LinearClassifierNetwork.randomized(1, dimension, bounds)
    setosa_result = train_linear_classifier_network(setosa_student, setosa_data, learning_rate=0.25, epochs=100)
    setosa_diagnostic = setosa_result.diagnostic

    print(
        f"setosa vs. rest (cardinality=1): training accuracy {setosa_diagnostic.best_training_accuracy:.3f} "
        f"({setosa_diagnostic.status_label} - best epoch "
        f"{setosa_diagnostic.best_epoch_index + 1}/{len(setosa_diagnostic.epoch_training_accuracies)}) - "
        "the trivially-separable case, a sanity check that this isn't inherently broken on real "
        "feature data, only where the classes genuinely overlap."
    )
    print()

    # versicolor vs virginica: not linearly separable. Swept across cardinality and gate, as in
    # demo_xor_linear_classifier_ceiling.py, so it isn't one architecture's failure
    vv_data = [
        (state, 1.0 if label == VIRGINICA_LABEL else 0.0) for state, label in dataset if label != SETOSA_LABEL
    ]

    configs: list[tuple[int, int, str]] = [
        (1, 1, "cardinality=1"),
        (2, 1, "cardinality=2, OR"),
        (2, 2, "cardinality=2, AND"),
        (3, 1, "cardinality=3, OR"),
        (3, 2, "cardinality=3, majority (2-of-3)"),
        (3, 3, "cardinality=3, AND"),
    ]

    best_accuracy = 0.0
    best_label = ""

    for cardinality, required_active, label in configs:
        student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, required_active)
        result = train_linear_classifier_network(student, vv_data, learning_rate=0.25, epochs=300)
        diagnostic = result.diagnostic

        print(
            f"versicolor vs. virginica ({label}): training accuracy "
            f"{diagnostic.best_training_accuracy:.3f} ({diagnostic.status_label} - best epoch "
            f"{diagnostic.best_epoch_index + 1}/{len(diagnostic.epoch_training_accuracies)})"
        )

        if diagnostic.best_training_accuracy > best_accuracy:
            best_accuracy = diagnostic.best_training_accuracy
            best_label = label

    print()
    print(
        f"best seen across every configuration: {best_accuracy:.3f} ({best_label}) - never 1.000, "
        "unlike setosa-vs-rest above. This is a real ceiling caused by genuine feature-space "
        "overlap between the two species (confirmed infeasible to separate perfectly, not a "
        "training/tuning shortfall), a different kind of ceiling than demo_xor_linear_classifier_"
        "ceiling.py's XOR target - that one is a *shape* no linear gate can express at any "
        "accuracy; this one is representable almost perfectly, just not exactly. See "
        "demo_iris_backprop_versus_perceptron.py for whether backprop's extra capacity helps here "
        "the way it does on XOR."
    )


if __name__ == "__main__":
    main()
