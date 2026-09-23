import math

from indrajala_ml.capture_common import stamp_brush

TARGET_MAX_DIMENSION = 20
CANVAS_SIZE = 28

# the interactive capture tool's own painting resolution - deliberately higher than
# TARGET_MAX_DIMENSION, so there's real room for scale_to_fit's aspect-preserving normalization
# to do something meaningful (see demo_mnist_ensemble_capture.py)
CAPTURE_GRID_SIZE = 64

# empirically tuned against a real trained model (mirroring exactly how
# digit_capture.CAPTURE_BRUSH_RADIUS was chosen) - a single mouse-cell-wide stroke is far
# thinner than any real digit stroke once cropped and scaled down to fit the 20px box, the
# same failure mode fixed for the smaller UCI-digits capture tool
CAPTURE_BRUSH_RADIUS = 4


def paint_brush_stroke(
    grid: list[list[float]], row: int, col: int, radius: int = CAPTURE_BRUSH_RADIUS
) -> list[list[float]]:
    """
    Returns a new capture grid (grid itself is left untouched) with a radius-cell square brush
    stamped fully on, centered at (row, col), clipped to the grid's bounds - the same actual
    stamping algorithm digit_capture.paint_brush_stroke uses for the smaller UCI-digits capture
    tool, shared via capture_common.stamp_brush; only this grid-size-specific validation lives
    here.
    """

    assert len(grid) == CAPTURE_GRID_SIZE and all(
        len(r) == CAPTURE_GRID_SIZE for r in grid
    ), "grid must be CAPTURE_GRID_SIZE x CAPTURE_GRID_SIZE"

    return stamp_brush(grid, row, col, radius)


def bounding_box(grid: list[list[float]]) -> tuple[int, int, int, int] | None:
    """
    The (min_row, min_col, max_row, max_col) of every pixel with a positive value, or None if
    the grid is empty (nothing drawn yet) - the first step of MNIST's real preprocessing
    (crop to the digit's own extent before size-normalizing it).
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
    General-purpose area-weighted resampling: each output pixel's value is the intensity-
    weighted average of every source pixel it overlaps, by exact overlap area. This is what
    MNIST's own documentation calls "anti-aliasing" - unlike nearest-neighbor or simple
    non-overlapping block-counting (which only works for integer scale ratios, see
    digit_capture.downsample_to_target_grid for the UCI digits dataset's own reference
    algorithm), this handles arbitrary source/target sizes correctly, needed here since a
    cropped bounding box is an arbitrary size and 64 doesn't evenly divide 20 or 28.
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
    Aspect-ratio-preserving resize so the longer dimension equals max_dimension - MNIST's real
    size-normalization step ("size normalized to fit in a 20x20 pixel box while preserving
    aspect ratio"). A tall, narrow crop (e.g. a "1") stays narrow; a wide one fills more of
    both dimensions - the box is a ceiling on the longer side, not a fixed size to stretch into.
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

    # resize_area_weighted's average is mathematically bounded by source's own min/max - here
    # always [0.0, 1.0], a binary capture grid - but summing many small floating-point overlap
    # contributions can overshoot that bound by a tiny amount (observed directly:
    # 1.0000000000000002), which intensity_to_color's strict [0.0, 1.0] assertion then rejects.
    # Clamping here (where the [0.0, 1.0] input range is actually guaranteed, unlike the
    # general-purpose resize_area_weighted itself) corrects the representation, not the math.
    return [[min(1.0, max(0.0, value)) for value in row] for row in resized]


def center_of_mass(grid: list[list[float]]) -> tuple[float, float]:
    """
    The intensity-weighted centroid (row, col) of grid - MNIST's real centering step uses this,
    not the bounding box's geometric center, to decide where to place a normalized glyph within
    the 28x28 field.
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
    Pastes small_grid into a canvas_size x canvas_size zero-filled canvas, translated so its
    center_of_mass lands on the canvas center - MNIST's real centering step ("translating the
    image so as to position this point at the center of the 28x28 field"). Pixels that would
    fall outside the canvas after translation are simply clipped (dropped), which only happens
    for content already close to the intended max_dimension x max_dimension size, so at most a
    thin edge.
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
    The full pipeline demo_mnist_ensemble_capture.py calls on every stroke: crop to the drawn content's
    bounding box, scale_to_fit(..., 20), then place_centered(..., 28) - genuinely reproducing
    MNIST's own three-step reference preprocessing (crop -> aspect-preserving anti-aliased
    scale-to-20 -> center-of-mass placement into 28), not an approximation of it. Returns an
    all-zero 28x28 canvas when nothing has been drawn yet.
    """

    box = bounding_box(capture_grid)
    if box is None:
        return [[0.0] * CANVAS_SIZE for _ in range(CANVAS_SIZE)]

    cropped = crop(capture_grid, box)
    scaled = scale_to_fit(cropped, TARGET_MAX_DIMENSION)
    return place_centered(scaled, CANVAS_SIZE)
