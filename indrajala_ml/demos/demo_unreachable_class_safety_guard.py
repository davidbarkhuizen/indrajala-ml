from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.networks.python.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import random_alternating_training_data


def main() -> None:

    dimension = 2
    bounds = square_bounds(10.0)

    print("generating training data from a normal, randomly initialised classifier...")
    for attempt in range(1, 21):
        reachable = LinearClassifierNetwork.randomized(1, dimension, bounds)
        try:
            training_data = random_alternating_training_data(200, reachable, max_attempts=10_000)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("no workable classifier found - this should be exceedingly rare")

    print(f"  succeeded on attempt {attempt}: sampled {len(training_data)} class-balanced examples")
    if attempt > 1:
        print(f"  (needed {attempt} attempts - even a normal randomize() can occasionally make one class unreachable)")

    print()
    print("generating training data from a deliberately unreachable-class classifier...")
    print("  (tiny weights + a large threshold mean one class never fires within these bounds)")
    unreachable = LinearClassifierNetwork(1, dimension, bounds)
    node = unreachable.hidden_layer.nodes[0]
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0

    try:
        random_alternating_training_data(200, unreachable, max_attempts=10_000)
        print("  unexpectedly succeeded")
    except RuntimeError as error:
        print(f"  raised as expected: {error}")


if __name__ == "__main__":
    main()
