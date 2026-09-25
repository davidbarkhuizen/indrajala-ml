import pytest

from indrajala_ml.digit_capture import (
    BLOCK_SIZE,
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    GRID_SIZE,
    MAX_BLOCK_VALUE,
    downsample_to_target_grid,
    intensity_to_color,
    paint_brush_stroke,
    pixel_to_tile,
    tile_grid_to_state,
)
from tests.helpers import approx


def test_tile_grid_to_state_flattens_row_major():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
    grid[2][3] = 1.0

    state = tile_grid_to_state(grid)

    assert len(state) == 64
    assert state[2 * GRID_SIZE + 3] == 1.0
    assert sum(state) == 1.0


def test_tile_grid_to_state_all_off():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    assert tile_grid_to_state(grid) == tuple(0.0 for _ in range(64))


def test_tile_grid_to_state_all_on():

    grid = [[1.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    assert tile_grid_to_state(grid) == tuple(1.0 for _ in range(64))


def test_tile_grid_to_state_rejects_wrong_row_count():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE - 1)]

    with pytest.raises(AssertionError):
        tile_grid_to_state(grid)


def test_tile_grid_to_state_rejects_wrong_column_count():

    grid = [[0.0] * (GRID_SIZE - 1) for _ in range(GRID_SIZE)]

    with pytest.raises(AssertionError):
        tile_grid_to_state(grid)


def test_pixel_to_tile_maps_coordinates_to_the_containing_tile():

    tile_size = 35

    assert pixel_to_tile(0, 0, tile_size) == (0, 0)
    assert pixel_to_tile(34, 34, tile_size) == (0, 0)
    assert pixel_to_tile(35, 0, tile_size) == (0, 1)
    assert pixel_to_tile(0, 35, tile_size) == (1, 0)
    assert pixel_to_tile(35 * 7 + 5, 35 * 7 + 5, tile_size) == (7, 7)


def test_pixel_to_tile_does_not_clamp_out_of_range_coordinates():

    # the caller's job to reject these, not pixel_to_tile's - see demo_uci_digit_capture.py
    assert pixel_to_tile(-1, -1, 35) == (-1, -1)
    assert pixel_to_tile(35 * 8, 0, 35) == (0, 8)


def _empty_capture_grid() -> list[list[float]]:
    return [[0.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]


def test_block_size_and_max_block_value_match_the_reference_work():

    # 32x32 divided into 8x8 nonoverlapping blocks -> each block is 4x4 -> 16 sub-pixels,
    # matching the bundled training data's own 0-16 pixel grading exactly
    assert BLOCK_SIZE == 4
    assert MAX_BLOCK_VALUE == 16


def test_downsample_to_target_grid_all_off_is_all_zero():

    capture_grid = _empty_capture_grid()

    target_grid = downsample_to_target_grid(capture_grid)

    assert target_grid == [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]


def test_downsample_to_target_grid_all_on_is_all_one():

    capture_grid = [[1.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]

    target_grid = downsample_to_target_grid(capture_grid)

    assert target_grid == [[1.0] * GRID_SIZE for _ in range(GRID_SIZE)]


def test_downsample_to_target_grid_counts_a_single_on_pixel_within_its_block():

    capture_grid = _empty_capture_grid()
    # top-left pixel of the block feeding target cell (0, 0)
    capture_grid[0][0] = 1.0

    target_grid = downsample_to_target_grid(capture_grid)

    assert target_grid[0][0] == approx(1 / 16)
    assert sum(sum(row) for row in target_grid) == approx(1 / 16)


def test_downsample_to_target_grid_a_fully_lit_block_is_exactly_full_intensity():

    capture_grid = _empty_capture_grid()
    # every pixel in the 4x4 block feeding target cell (1, 2)
    block_row, block_col = 1 * BLOCK_SIZE, 2 * BLOCK_SIZE
    for dr in range(BLOCK_SIZE):
        for dc in range(BLOCK_SIZE):
            capture_grid[block_row + dr][block_col + dc] = 1.0

    target_grid = downsample_to_target_grid(capture_grid)

    assert target_grid[1][2] == 1.0
    # nothing else was touched
    assert sum(sum(row) for row in target_grid) == 1.0


def test_downsample_to_target_grid_blocks_do_not_overlap():

    capture_grid = _empty_capture_grid()
    # last pixel of the block feeding (0, 0) and the first pixel of the block feeding (0, 1)
    # are adjacent on the capture grid but must land in different target cells
    capture_grid[BLOCK_SIZE - 1][BLOCK_SIZE - 1] = 1.0
    capture_grid[0][BLOCK_SIZE] = 1.0

    target_grid = downsample_to_target_grid(capture_grid)

    assert target_grid[0][0] == approx(1 / 16)
    assert target_grid[0][1] == approx(1 / 16)
    assert sum(sum(row) for row in target_grid) == approx(2 / 16)


def test_downsample_to_target_grid_rejects_wrong_size_input():

    with pytest.raises(AssertionError):
        downsample_to_target_grid([[0.0] * GRID_SIZE for _ in range(GRID_SIZE)])


def test_intensity_to_color_maps_to_grayscale_hex():

    assert intensity_to_color(0.0) == "#000000"
    assert intensity_to_color(1.0) == "#ffffff"
    # 0.375 * 255 = 95.625, rounds to 96 = 0x60
    assert intensity_to_color(0.375) == "#606060"


def _empty_capture_grid_for_brush() -> list[list[float]]:
    return [[0.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]


def test_paint_brush_stroke_stamps_a_square_neighborhood_fully_on():

    grid = _empty_capture_grid_for_brush()

    new_grid = paint_brush_stroke(grid, 10, 10, radius=2)

    for delta_row in range(-2, 3):
        for delta_col in range(-2, 3):
            assert new_grid[10 + delta_row][10 + delta_col] == 1.0

    # nothing further away than the radius is touched
    assert new_grid[10 - 3][10] == 0.0
    assert new_grid[10][10 + 3] == 0.0

    # the original grid is left untouched - paint_brush_stroke returns a new one
    assert grid[10][10] == 0.0


def test_paint_brush_stroke_clips_to_the_grid_at_a_corner():

    grid = _empty_capture_grid_for_brush()

    new_grid = paint_brush_stroke(grid, 0, 0, radius=2)

    # only the in-bounds quarter of the brush is stamped
    on_count = sum(sum(row) for row in new_grid)
    assert on_count == 3 * 3
    assert new_grid[0][0] == 1.0
    assert new_grid[2][2] == 1.0
    assert new_grid[3][0] == 0.0


def test_paint_brush_stroke_default_radius_matches_capture_brush_radius():

    grid = _empty_capture_grid_for_brush()

    new_grid = paint_brush_stroke(grid, 10, 10)

    stroke_width = 2 * CAPTURE_BRUSH_RADIUS + 1
    assert sum(sum(row) for row in new_grid) == stroke_width * stroke_width


def test_paint_brush_stroke_rejects_out_of_bounds_center():

    grid = _empty_capture_grid_for_brush()

    with pytest.raises(AssertionError):
        paint_brush_stroke(grid, -1, 0)

    with pytest.raises(AssertionError):
        paint_brush_stroke(grid, 0, CAPTURE_GRID_SIZE)


def test_intensity_to_color_rejects_out_of_range_values():

    with pytest.raises(AssertionError):
        intensity_to_color(-0.01)

    with pytest.raises(AssertionError):
        intensity_to_color(1.01)
