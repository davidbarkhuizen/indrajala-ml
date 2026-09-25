import math

from indrajala_ml.capture_common import stamp_brush

TARGET_MAX_DIMENSION = 20
CANVAS_SIZE = 28

# the capture tool's painting resolution, above TARGET_MAX_DIMENSION so scale_to_fit has room
# to work
CAPTURE_GRID_SIZE = 64

# tuned against a trained model, as digit_capture's radius was: a one-cell stroke is far
# thinner than a real digit's once scaled into the 20px box
CAPTURE_BRUSH_RADIUS = 4


def paint_brush_stroke(
    grid: list[list[float]], row: int, col: int, radius: int = CAPTURE_BRUSH_RADIUS
) -> list[list[float]]:
    """
    A copy of grid with a square brush stamped at (row, col) (capture_common.stamp_brush), after
    this grid's size checks.
    """

    assert len(grid) == CAPTURE_GRID_SIZE and all(len(r) == CAPTURE_GRID_SIZE for r in grid), (
        "grid must be CAPTURE_GRID_SIZE x CAPTURE_GRID_SIZE"
    )

    return stamp_brush(grid, row, col, radius)


def bounding_box(grid: list[list[float]]) -> tuple[int, int, int, int] | None:
    """
    (min_row, min_col, max_row, max_col) of every positive pixel, or None for an empty grid: MNIST's
    first preprocessing step crops to it.
    """

    min_row = min_col = max_row = max_col = None

    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            if value > 0.0:
                min_row = row if min_row is None else min(min_row, row)
                max_row = row if max_row is None else max(max_row, row)
                min_col = col if min_col is None else min(min_col, col)
                max_col = col if max_col is None else max(max_col, col)

    if min_row is None:
        return None

    return min_row, min_col, max_row, max_col


def crop(grid: list[list[float]], box: tuple[int, int, int, int]) -> list[list[float]]:
    min_row, min_col, max_row, max_col = box
    return [row[min_col : max_col + 1] for row in grid[min_row : max_row + 1]]


def resize_area_weighted(source: list[list[float]], target_height: int, target_width: int) -> list[list[float]]:
    """
    Area-weighted resampling: each output pixel is the average of the source pixels it overlaps,
    weighted by overlap area (MNIST's "anti-aliasing"). Unlike block counting
    (digit_capture.downsample_to_target_grid), it handles any size ratio, which a cropped box needs.
    """

    assert target_height >= 1 and target_width >= 1, "target dimensions must be at least 1"

    source_height = len(source)
    source_width = len(source[0]) if source_height else 0
    assert source_height >= 1 and source_width >= 1, "source must not be empty"

    row_scale = source_height / target_height
    col_scale = source_width / target_width

    result = [[0.0] * target_width for _ in range(target_height)]

    for out_row in range(target_height):
        row_start = out_row * row_scale
        row_end = row_start + row_scale
        for out_col in range(target_width):
            col_start = out_col * col_scale
            col_end = col_start + col_scale

            total = 0.0
            for source_row in range(int(row_start), min(source_height, math.ceil(row_end))):
                row_overlap = min(row_end, source_row + 1) - max(row_start, source_row)
                if row_overlap <= 0.0:
                    continue
                for source_col in range(int(col_start), min(source_width, math.ceil(col_end))):
                    col_overlap = min(col_end, source_col + 1) - max(col_start, source_col)
                    if col_overlap <= 0.0:
                        continue
                    total += source[source_row][source_col] * row_overlap * col_overlap

            result[out_row][out_col] = total / (row_scale * col_scale)

    return result


def scale_to_fit(source: list[list[float]], max_dimension: int = TARGET_MAX_DIMENSION) -> list[list[float]]:
    """
    Resizes, keeping the aspect ratio, so the longer side is max_dimension: MNIST's "size normalized
    to fit in a 20x20 pixel box while preserving aspect ratio". A narrow "1" stays narrow.
    """

    source_height = len(source)
    source_width = len(source[0]) if source_height else 0
    assert source_height >= 1 and source_width >= 1, "source must not be empty"

    if source_height >= source_width:
        target_height = max_dimension
        target_width = max(1, round(source_width * max_dimension / source_height))
    else:
        target_width = max_dimension
        target_height = max(1, round(source_height * max_dimension / source_width))

    resized = resize_area_weighted(source, target_height, target_width)

    # the area-weighted average of [0.0, 1.0] values can round just past 1.0
    # (1.0000000000000002), which intensity_to_color rejects; clamp
    return [[min(1.0, max(0.0, value)) for value in row] for row in resized]


def center_of_mass(grid: list[list[float]]) -> tuple[float, float]:
    """
    The intensity-weighted centroid (row, col): MNIST centers on this, not the bounding box's
    center.
    """

    total = 0.0
    row_sum = 0.0
    col_sum = 0.0

    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            total += value
            row_sum += value * (row + 0.5)
            col_sum += value * (col + 0.5)

    height = len(grid)
    width = len(grid[0]) if height else 0

    if total == 0.0:
        return height / 2.0, width / 2.0

    return row_sum / total, col_sum / total


def place_centered(small_grid: list[list[float]], canvas_size: int = CANVAS_SIZE) -> list[list[float]]:
    """
    Pastes small_grid into a zeroed canvas_size square so its center of mass lands on the center
    ("translating the image so as to position this point at the center of the 28x28 field").
    Pixels translated off the canvas are dropped, at most a thin edge.
    """

    canvas = [[0.0] * canvas_size for _ in range(canvas_size)]

    height = len(small_grid)
    width = len(small_grid[0]) if height else 0
    if height == 0 or width == 0:
        return canvas

    centroid_row, centroid_col = center_of_mass(small_grid)
    target_center = canvas_size / 2.0
    row_offset = round(target_center - centroid_row)
    col_offset = round(target_center - centroid_col)

    for row in range(height):
        canvas_row = row + row_offset
        if not (0 <= canvas_row < canvas_size):
            continue
        for col in range(width):
            canvas_col = col + col_offset
            if 0 <= canvas_col < canvas_size:
                canvas[canvas_row][canvas_col] = small_grid[row][col]

    return canvas


def preprocess_capture(capture_grid: list[list[float]]) -> list[list[float]]:
    """
    MNIST's preprocessing of a capture: crop to the drawn content, scale_to_fit(..., 20), then
    place_centered(..., 28). An empty capture gives an all-zero 28x28 canvas.
    """

    box = bounding_box(capture_grid)
    if box is None:
        return [[0.0] * CANVAS_SIZE for _ in range(CANVAS_SIZE)]

    cropped = crop(capture_grid, box)
    scaled = scale_to_fit(cropped, TARGET_MAX_DIMENSION)
    return place_centered(scaled, CANVAS_SIZE)
