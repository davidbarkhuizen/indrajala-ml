from indrajala_ml.capture_common import stamp_brush

GRID_SIZE = 8

# the UCI digits' own preprocessing (sklearn.datasets.load_digits()'s description): "32x32
# bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted
# in each block", giving data/digits/digits.csv's 8x8 grid of 0-16. Capturing at 32x32 and
# counting blocks gives a painted digit the training examples' soft edges.
CAPTURE_GRID_SIZE = 32
BLOCK_SIZE = CAPTURE_GRID_SIZE // GRID_SIZE
MAX_BLOCK_VALUE = BLOCK_SIZE * BLOCK_SIZE  # 16 - matches digits_data.py's own /16.0 normalization


def tile_grid_to_state(grid: list[list[float]]) -> tuple[float, ...]:
    """
    Flattens an 8x8 tile grid row-major into the 64-value [0.0, 1.0] state load_digits_dataset
    produces (row 0 -> indices 0-7, ...), as sklearn's load_digits().data flattens its images. Kept
    out of the tkinter capture demo so it can be tested without a display.
    """

    assert len(grid) == GRID_SIZE, f"grid must have {GRID_SIZE} rows; got {len(grid)}"
    assert all(len(row) == GRID_SIZE for row in grid), f"every row must have {GRID_SIZE} columns"

    return tuple(value for row in grid for value in row)


def pixel_to_tile(x: int, y: int, tile_size: int) -> tuple[int, int]:
    """
    The (row, col) tile a canvas pixel falls in. Doesn't clamp: a drag just outside the canvas can
    give an out-of-range tile, which the caller ignores (demo_uci_digit_capture.py's
    _handle_paint_event).
    """

    return y // tile_size, x // tile_size


def downsample_to_target_grid(capture_grid: list[list[float]]) -> list[list[float]]:
    """
    The UCI digits' preprocessing: a CAPTURE_GRID_SIZE-square binary bitmap is cut into BLOCK_SIZE
    blocks, each block's on pixels counted (out of MAX_BLOCK_VALUE, 16) and divided by 16, as
    load_digits_dataset normalizes the training data. A stroke covering part of a block gives
    partial intensity, the soft edges real examples have.
    """

    assert len(capture_grid) == CAPTURE_GRID_SIZE, f"capture_grid must have {CAPTURE_GRID_SIZE} rows"
    assert all(len(row) == CAPTURE_GRID_SIZE for row in capture_grid), (
        f"every row must have {CAPTURE_GRID_SIZE} columns"
    )

    target_grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    for target_row in range(GRID_SIZE):
        for target_col in range(GRID_SIZE):
            on_pixel_count = sum(
                capture_grid[target_row * BLOCK_SIZE + block_row][target_col * BLOCK_SIZE + block_col]
                for block_row in range(BLOCK_SIZE)
                for block_col in range(BLOCK_SIZE)
            )
            target_grid[target_row][target_col] = on_pixel_count / MAX_BLOCK_VALUE

    return target_grid


CAPTURE_BRUSH_RADIUS = 2


def paint_brush_stroke(
    grid: list[list[float]], row: int, col: int, radius: int = CAPTURE_BRUSH_RADIUS
) -> list[list[float]]:
    """
    A brush stroke on the capture grid. A one-cell stroke block-counts down to about 25% intensity
    at best, while real examples are near full (14-16 of 16) along their strokes; a classifier then
    calls almost everything "7".

    radius=2 (a 5x5 stamp) was picked by testing: it reaches near-full block intensity along a
    stroke while leaving a "0"'s hole open. At radius 3 and 4 the hole filled in, and a hand-drawn
    "0" classified correctly at 2 was read as "4". The stamping is capture_common.stamp_brush.
    """

    assert len(grid) == CAPTURE_GRID_SIZE and all(len(r) == CAPTURE_GRID_SIZE for r in grid), (
        "grid must be CAPTURE_GRID_SIZE x CAPTURE_GRID_SIZE"
    )

    return stamp_brush(grid, row, col, radius)


def intensity_to_color(intensity: float) -> str:
    """
    A [0.0, 1.0] intensity as a grayscale hex color, 0.0 -> "#000000", 1.0 -> "#ffffff".
    """

    assert 0.0 <= intensity <= 1.0, f"intensity must be in [0.0, 1.0]; got {intensity}"

    level = round(intensity * 255)
    return f"#{level:02x}{level:02x}{level:02x}"
