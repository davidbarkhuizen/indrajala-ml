from indrajala_ml.demos.capture_app import CaptureConfig, run_capture_demo
from indrajala_ml.mnist_capture import (
    CANVAS_SIZE,
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    paint_brush_stroke,
    preprocess_capture,
)
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork

MODEL_PATH = "data/mnist/trained_model.json"


def _grid_to_state(grid: list[list[float]]) -> tuple[float, ...]:
    return tuple(value for row in grid for value in row)


# Captures a digit as MNIST's source data was preprocessed (mnist_capture.preprocess_capture):
# a binary CAPTURE_GRID_SIZE-square bitmap painted with a CAPTURE_BRUSH_RADIUS brush is cropped
# to its bounding box, scaled to fit a 20px box and placed by center of mass in a 28x28 field.
# The UI is capture_app.py's.
CONFIG = CaptureConfig(
    capture_grid_size=CAPTURE_GRID_SIZE,
    capture_tile_size=8,
    preview_grid_size=CANVAS_SIZE,
    preview_tile_size=10,
    brush_radius=CAPTURE_BRUSH_RADIUS,
    paint_brush_stroke=paint_brush_stroke,
    preprocess=preprocess_capture,
    to_state=_grid_to_state,
)


def main() -> None:
    run_capture_demo(
        title="MNIST capture",
        model_path=MODEL_PATH,
        load_classifier=EnsembleBackpropClassifierNetwork.load,
        train_demo_title="MNIST ensemble recognition",
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
