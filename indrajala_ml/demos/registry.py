from dataclasses import dataclass


@dataclass(frozen=True)
class DemoInfo:
    module: str
    title: str
    summary: str
    description: str


DEMOS: list[DemoInfo] = [
    DemoInfo(
        module="indrajala_ml.demos.demo_minimum_disturbance_training",
        title="Minimum-disturbance training",
        summary="Trains one cardinality=4 linear classifier; plots its convergence and decision boundary.",
        description=(
            "Trains a LinearClassifierNetwork (cardinality=4, four hyperplanes ANDed together) against "
            "a random reference classifier, showcasing the minimum-disturbance multi-unit learning rule. "
            "Pops up three windows: the convergence curve on a linear scale, the same curve on a log "
            "scale (disagreement tends to drop roughly exponentially), and the decision-boundary chart. "
            "The windows stay open until you close them. Also prints the trained student's classification "
            "of a fresh point never seen during training, alongside the reference's own classification."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_linear_classifier_cardinality_sweep",
        title="Linear-classifier cardinality sweep",
        summary="Trains linear classifiers at cardinality 1-4 and overlays their convergence curves.",
        description=(
            "Trains independent reference/student pairs at cardinality = 1, 2, 3, 4 and overlays their "
            "disagreement-rate convergence curves on two charts (linear-scale and log-scale), plus "
            "printing each cardinality's before/after disagreement and a prediction-agreement check. "
            "Since a higher cardinality shrinks the reference's positive region, some randomly generated "
            "references make one class unreachable within the bounds - the demo regenerates the reference "
            "rather than failing the whole sweep on one unlucky draw."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_unreachable_class_safety_guard",
        title="Unreachable-class safety guard",
        summary="Headless demo of the guard that stops training-data generation from hanging forever.",
        description=(
            "A headless, console-only demo of random_alternating_training_data's safety guard. First "
            "generates training data from a normal, randomly initialised classifier, then deliberately "
            "constructs a classifier with tiny weights and a large threshold whose decision boundary "
            "never crosses its bounds - and shows the resulting RuntimeError being raised and caught "
            "instead of the data generator hanging forever."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_xor_linear_classifier_ceiling",
        title="XOR: linear-classifier ceiling",
        summary="Shows no linear-classifier gate (AND/OR/majority, cardinality 1-4) can learn XOR.",
        description=(
            "Trains students at cardinality = 1..4, swept across AND, OR, and majority required_active "
            "gates, against an XOR-style target (two diagonally opposite quadrants) that is deliberately "
            "not representable by any LinearClassifierNetwork of that shape. Reports each configuration's "
            "training diagnostic (converged / plateaued / still improving) and a summary tally showing "
            "none of them get close to converging, then plots the training data against the "
            "best-performing student's hyperplanes to show the mismatch visually."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_xor_backprop_convergence",
        title="XOR: backprop convergence",
        summary="Trains a backprop network on the same XOR target - and shows it converges past the ceiling.",
        description=(
            "The direct counterpart to the linear-classifier-ceiling demo: same XOR target, same training "
            "data generation, but training a BackpropClassifierNetwork instead of a LinearClassifierNetwork. "
            "Where every linear configuration plateaus well short of convergence, this one reaches a "
            "training accuracy well past that ceiling, because its output layer is trained too. Plots the "
            "training data over a probability heatmap instead of decision lines."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_iris_linear_classifier_ceiling",
        title="Iris: linear-classifier ceiling",
        summary="Headless - real Fisher's Iris data: perceptron solves setosa-vs-rest, hits a real ceiling on versicolor-vs-virginica.",
        description=(
            "Headless, console-only - the first demo in this repo to train a LinearClassifierNetwork on "
            "real data instead of a synthetic 2D geometric target. Fisher's Iris (1936) has one famous "
            "linear-separability split: setosa is perfectly separable from the other two species (a "
            "cardinality=1 perceptron reliably converges to 1.000), but versicolor and virginica overlap "
            "and are NOT perfectly linearly separable - confirmed directly via a one-time linear-"
            "programming feasibility check, not assumed. Sweeps cardinality/gate configurations on "
            "versicolor-vs-virginica the same way demo_xor_linear_classifier_ceiling.py does, and none "
            "reach 1.000 - a different kind of ceiling than XOR's inexpressible shape: this one is caused "
            "by genuine feature-space overlap, and is representable almost perfectly, just not exactly."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_iris_backprop_versus_perceptron",
        title="Iris: backprop vs. perceptron on a real ceiling",
        summary="Headless - unlike XOR, backprop doesn't clear the versicolor/virginica ceiling; it does generalize slightly better.",
        description=(
            "Headless, console-only - the direct counterpart to the Iris linear-classifier-ceiling demo, "
            "asking whether BackpropClassifierNetwork clears that ceiling the way it clears XOR's. Repeats "
            "an 80/20 train/test split 10 times (a single 20-example test set is too small and noisy to "
            "trust alone) and averages perceptron vs. backprop training AND held-out accuracy. Measured "
            "result: a null on training accuracy (both plateau at roughly the same ceiling, since extra "
            "capacity doesn't help when the ceiling is caused by real data ambiguity, not an inexpressible "
            "shape), but backprop's held-out test accuracy is measurably higher on average - reported as "
            "measured, not rounded up to match the more dramatic XOR story."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_backprop_stripes_architecture_sweep",
        title="Backprop stripes architecture sweep",
        summary="Compares backprop architectures ([4],[8],[4,4],[8,8]) at matched node budgets.",
        description=(
            "Mirrors the cardinality-sweep demo's pattern (train several configurations against the same "
            "target, overlay smoothed convergence curves) but compares BackpropClassifierNetwork "
            "architectures - [4], [8], [4, 4], [8, 8] - at matched total node budgets. The target is a "
            "4-band alternating-stripes pattern, harder than the XOR demos' 2-region target, so the "
            "architectures actually separate instead of all converging easily. Unseeded, like every sweep "
            "demo here: which architecture wins varies noticeably between runs."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_backprop_circular_boundary",
        title="Backprop circular boundary",
        summary="Trains a backprop network on a circular target to show a genuinely curved boundary.",
        description=(
            "Every other demo's target boundary is built from straight edges, since a LinearClassifierNetwork's "
            "positive region is always a polygon. This target is a disk, a genuinely curved boundary. Trains "
            "a single-hidden-layer BackpropClassifierNetwork (reaches ~0.99 training accuracy) and plots the "
            "learned probability heatmap against the training data, visibly showing a smooth, rounded decision "
            "boundary rather than a faceted polygon."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_backprop_linear_parity_check",
        title="Backprop vs linear parity check",
        summary="Trains both model types on an easy, linearly-separable target as a parity check.",
        description=(
            "Every other backprop demo picks a target no LinearClassifierNetwork can represent well (XOR, "
            "stripes, a circle); this one is the opposite check - a single half-plane (cardinality=1), squarely "
            "within LinearClassifierNetwork's own representational sweet spot. Trains a LinearClassifierNetwork "
            "and a BackpropClassifierNetwork on the exact same reference and training data, then overlays the "
            "reference's boundary, the linear student's boundary, and the backprop student's probability heatmap "
            "- a parity check showing backprop learns just as well here."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_uci_digit_recognition",
        title="UCI digit recognition",
        summary="Trains a multi-class backprop network on the bundled 8x8 UCI digits dataset.",
        description=(
            "Classic-style handwritten digit recognition using MultiClassBackpropClassifierNetwork. Loads the "
            "bundled UCI ML hand-written digits dataset (8x8 pixel images, 10 classes, 1797 samples), splits "
            "80/20 into train/test, and trains a single-hidden-layer (32-node) network. Saves the trained model "
            "to disk so it can be reloaded without retraining, then plots a training-accuracy-by-epoch curve, a "
            "confusion matrix, and a grid of sample test predictions."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_vectorized_uci_digit_recognition",
        title="Vectorized UCI digit recognition (numpy)",
        summary="Trains pure-Python and numpy-array-backed networks on UCI digits; compares accuracy and wall-clock.",
        description=(
            "Vectorized vs pure-Python validation. Trains MultiClassBackpropClassifierNetwork and its "
            "numpy-array-backed sibling VectorizedMultiClassBackpropClassifierNetwork on the same UCI digits "
            "split, at the same seed and hyperparameters, reporting each one's measured training/test "
            "accuracy and wall-clock training time side by side. Then plots both training-accuracy-by-epoch "
            "curves overlaid, a confusion matrix, and sample test predictions for the vectorized network."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_rust_vs_vectorized_uci_digit_recognition",
        title="Rust vs vectorized UCI digit recognition",
        summary="Trains pure-Python, numpy, and Rust-array-core-backed networks on UCI digits side by side.",
        description=(
            "Rust vs numpy validation. Extends demo_vectorized_uci_digit_recognition.py with a third "
            "network: RustArrayMultiClassBackpropClassifierNetwork, the Rust-array-core-backed sibling "
            "that replaces numpy as the production backend (numpy stays on permanently as the benchmark "
            "comparison). Trains all three at the same seed/hyperparameters and reports each one's "
            "measured accuracy and wall-clock training time. Then plots all three "
            "training-accuracy-by-epoch curves overlaid, a confusion matrix, and sample test "
            "predictions for the Rust network."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_uci_digit_capture",
        title="UCI digit capture",
        summary="Interactive mouse-painted digit capture, classified live by the UCI digit-recognition model.",
        description=(
            "An interactive companion to the UCI digit-recognition demo. Loads the model that demo trains and "
            "saves (run that demo first if it hasn't been trained yet), then opens two canvases: a 32x32 grid "
            "you paint with the mouse, and an 8x8 preview showing the result of genuinely reproducing the "
            "bundled training data's own preprocessing on what you drew. Every stroke updates the preview and "
            "reclassifies live, showing the predicted digit and the model's confidence. Needs a display and "
            "mouse input."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_mnist_ensemble_recognition",
        title="MNIST ensemble recognition",
        summary="Trains a 10-network ensemble on the full MNIST dataset via parallel multiprocessing.",
        description=(
            "The same shape as the UCI digit-recognition demo, but on the real, full-scale MNIST dataset "
            "(28x28 pixel images, 60000 train / 10000 test), using EnsembleBackpropClassifierNetwork: 10 "
            "completely independent BackpropClassifierNetworks, one per digit, trained as 10 parallel "
            "multiprocessing jobs with a memory-aware worker count. Takes tens of minutes to train at full "
            "scale; saves the trained model to disk so it can be reloaded without retraining."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_vectorized_mnist_recognition",
        title="Vectorized MNIST recognition (numpy)",
        summary="One real epoch on full MNIST, pure-Python vs. numpy-array-backed - accuracy and wall-clock.",
        description=(
            "Vectorized vs pure-Python validation at real MNIST scale. Headless, console-only - trains "
            "MultiClassBackpropClassifierNetwork and VectorizedMultiClassBackpropClassifierNetwork for one "
            "real epoch each over the full 60000-example MNIST training set (same 30-node-hidden-layer "
            "architecture as the ~12.5-minutes/epoch pure-Python baseline), then reports each one's test "
            "accuracy and wall-clock epoch time side by side. Also times "
            "load_mnist_dataset_as_array's bulk-decode against load_mnist_dataset's tuple-per-example decode "
            "on the same file. Takes on the order of 15-20 minutes to run (mostly the pure-Python epoch)."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_rust_vs_vectorized_mnist_recognition",
        title="Rust vs vectorized MNIST recognition",
        summary="One real epoch on full MNIST, pure-Python vs. numpy vs. Rust-array-core-backed.",
        description=(
            "Rust vs numpy validation at real MNIST scale. Headless, console-only - extends "
            "demo_vectorized_mnist_recognition.py with a third network: "
            "RustArrayMultiClassBackpropClassifierNetwork, the Rust-array-core-backed sibling that "
            "replaces numpy as the production backend. Trains all three for one real epoch each "
            "over the full 60000-example MNIST training set (same 30-node-hidden-layer "
            "architecture), then reports each one's test accuracy and wall-clock epoch time side "
            "by side. Takes on the order of 15-20 minutes to run (mostly the pure-Python epoch)."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_mnist_ensemble_capture",
        title="MNIST ensemble capture",
        summary="Interactive mouse-painted digit capture, classified live by the MNIST ensemble model.",
        description=(
            "An interactive companion to the MNIST-ensemble-recognition demo, the same overall interaction as "
            "the UCI digit-capture demo but against MNIST's own reference preprocessing. Loads the model that "
            "demo trains and saves (run that demo first if it hasn't been trained yet), then opens a 64x64 "
            "paint canvas and a 28x28 preview showing MNIST's own crop/scale/center-of-mass preprocessing "
            "applied to what you drew. Every stroke updates the preview and reclassifies live. Needs a display "
            "and mouse input."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_conv_depth_uci_digits_comparison",
        title="Conv depth comparison on UCI digits",
        summary="Headless - one vs two stacked conv layers (plus a dense baseline) on UCI digits, across paired seeds.",
        description=(
            "Headless, console-only. Trains a dense-only baseline, one conv layer, two stacked conv "
            "layers, and a parameter-matched two-conv-layer variant on UCI digits, all with the same "
            "dense tail, learning rate and epochs, over 8 seeds (each seed fixes the split and init, "
            "shared across configs). Prints mean/stdev training and held-out accuracy per config and "
            "paired per-seed test-accuracy differences against the single conv layer. Pure Python, so "
            "accuracy only - no timing. Takes roughly 40+ minutes on 8 logical cores."
        ),
    ),
    DemoInfo(
        module="indrajala_ml.demos.demo_backprop_variant_comparison",
        title="Backprop variant comparison",
        summary="Reproduces this repo's documented model/loss/init comparisons, runnably, side by side.",
        description=(
            "Three A/B comparisons, each trained fresh and plotted: multi-class loss function (one-vs-rest "
            "MSE vs softmax cross-entropy, on the full UCI digits set), binary loss function (quadratic vs "
            "binary cross-entropy, on the XOR target, at both an untuned and a retuned learning rate), and "
            "weight-init scheme (this codebase's production fan-in-aware default vs Xavier/Glorot, on UCI "
            "digits). Prints each comparison's measured accuracy and plots a training-accuracy-by-epoch "
            "chart per section. Takes about 4 minutes to train all seven variants."
        ),
    ),
]
