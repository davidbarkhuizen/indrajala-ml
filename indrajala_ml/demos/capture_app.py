import sys
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from indrajala_ml.digit_capture import intensity_to_color, pixel_to_tile

OFF_COLOR = "#000000"
ON_COLOR = "#ffffff"


class StateClassifier(Protocol):
    """
    What a capture demo classifies with (the multiclass network or the ensemble).
    """

    def classify_state(self, state: tuple[float, ...]) -> int: ...
    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]: ...


@dataclass(frozen=True)
class CaptureConfig:
    """
    What differs between the UCI-digits and MNIST capture demos: grid and tile sizes, the brush, and
    the capture-grid -> model-input pipeline.
    """

    capture_grid_size: int
    capture_tile_size: int
    preview_grid_size: int
    preview_tile_size: int
    brush_radius: int
    paint_brush_stroke: Callable[[list[list[float]], int, int], list[list[float]]]
    preprocess: Callable[[list[list[float]]], list[list[float]]]
    to_state: Callable[[list[list[float]]], tuple[float, ...]]


class CaptureApp:
    """
    The capture-and-classify UI of both capture demos: paint a binary capture grid by mouse (with a
    brush, since a one-cell line is fainter than any training stroke), preview it preprocessed into
    the model's input shape, and classify it on every stroke. The demos differ only in their
    CaptureConfig and classifier.
    """

    def __init__(self, root: tk.Tk, classifier: StateClassifier, config: CaptureConfig) -> None:

        self.classifier = classifier
        self.config = config

        grid_size = config.capture_grid_size
        self.capture_grid: list[list[float]] = [[0.0] * grid_size for _ in range(grid_size)]
        self.capture_tile_ids: list[list[int]] = [[0] * grid_size for _ in range(grid_size)]
        self.preview_tile_ids: list[list[int]] = [
            [0] * config.preview_grid_size for _ in range(config.preview_grid_size)
        ]

        capture_canvas_size = grid_size * config.capture_tile_size
        self.capture_canvas = tk.Canvas(
            root, width=capture_canvas_size, height=capture_canvas_size, bg=OFF_COLOR, highlightthickness=0
        )
        self.capture_canvas.grid(row=0, column=0, padx=10, pady=10)

        for row in range(grid_size):
            for col in range(grid_size):
                x0, y0 = col * config.capture_tile_size, row * config.capture_tile_size
                x1, y1 = x0 + config.capture_tile_size, y0 + config.capture_tile_size
                self.capture_tile_ids[row][col] = self.capture_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=OFF_COLOR, outline=""
                )

        self.capture_canvas.bind("<Button-1>", self._handle_paint_event)
        self.capture_canvas.bind("<B1-Motion>", self._handle_paint_event)

        preview_canvas_size = config.preview_grid_size * config.preview_tile_size
        self.preview_canvas = tk.Canvas(
            root, width=preview_canvas_size, height=preview_canvas_size, bg=OFF_COLOR, highlightthickness=0
        )
        self.preview_canvas.grid(row=0, column=1, padx=10, pady=10)

        for row in range(config.preview_grid_size):
            for col in range(config.preview_grid_size):
                x0, y0 = col * config.preview_tile_size, row * config.preview_tile_size
                x1, y1 = x0 + config.preview_tile_size, y0 + config.preview_tile_size
                self.preview_tile_ids[row][col] = self.preview_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=OFF_COLOR, outline="gray20"
                )

        self.prediction_label = tk.Label(root, text="draw a digit", font=("TkDefaultFont", 14))
        self.prediction_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        clear_button = tk.Button(root, text="Clear", command=self.clear)
        clear_button.grid(row=2, column=0, columnspan=2, pady=(0, 10))

    def _handle_paint_event(self, event: tk.Event) -> None:
        row, col = pixel_to_tile(event.x, event.y, self.config.capture_tile_size)
        if 0 <= row < self.config.capture_grid_size and 0 <= col < self.config.capture_grid_size:
            self._paint_capture_tile(row, col)

    def _paint_capture_tile(self, row: int, col: int) -> None:
        self.capture_grid = self.config.paint_brush_stroke(self.capture_grid, row, col)

        # only the brush's own bounding box could have changed - cheap enough to just
        # re-render every tile in it, clipped to the grid, rather than diffing
        radius = self.config.brush_radius
        grid_size = self.config.capture_grid_size
        for brush_row in range(max(0, row - radius), min(grid_size, row + radius + 1)):
            for brush_col in range(max(0, col - radius), min(grid_size, col + radius + 1)):
                self.capture_canvas.itemconfig(self.capture_tile_ids[brush_row][brush_col], fill=ON_COLOR)

        self._update_preview_and_classify()

    def clear(self) -> None:
        grid_size = self.config.capture_grid_size
        for row in range(grid_size):
            for col in range(grid_size):
                self.capture_grid[row][col] = 0.0
                self.capture_canvas.itemconfig(self.capture_tile_ids[row][col], fill=OFF_COLOR)
        self._update_preview_and_classify()
        self.prediction_label.config(text="draw a digit")

    def _update_preview_and_classify(self) -> None:
        target_grid = self.config.preprocess(self.capture_grid)

        for row in range(self.config.preview_grid_size):
            for col in range(self.config.preview_grid_size):
                color = intensity_to_color(target_grid[row][col])
                self.preview_canvas.itemconfig(self.preview_tile_ids[row][col], fill=color)

        if sum(sum(row) for row in target_grid) == 0.0:
            self.prediction_label.config(text="draw a digit")
            return

        state = self.config.to_state(target_grid)
        predicted = self.classifier.classify_state(state)
        probabilities = self.classifier.predict_probabilities(state)
        confidence = probabilities[predicted]
        self.prediction_label.config(text=f"predicted: {predicted}  (confidence {confidence:.2f})")


def run_capture_demo(
    *,
    title: str,
    model_path: str,
    load_classifier: Callable[[str], StateClassifier],
    train_demo_title: str,
    config: CaptureConfig,
) -> None:
    """
    A capture demo's main(): loads the classifier its recognition demo saved (or prints a hint and
    exits if there is none) and runs the capture UI.
    """

    try:
        classifier = load_classifier(model_path)
    except FileNotFoundError:
        print(
            f"no trained model found at {model_path} - run `./cli demo` and choose "
            f"'{train_demo_title}' first to train and save one."
        )
        sys.exit(1)

    root = tk.Tk()
    root.title(title)
    CaptureApp(root, classifier, config)
    root.mainloop()
