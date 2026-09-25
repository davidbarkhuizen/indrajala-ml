import pytest

from indrajala_ml.mnist_capture import (
    CANVAS_SIZE,
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    bounding_box,
    center_of_mass,
    crop,
    paint_brush_stroke,
    place_centered,
    preprocess_capture,
    resize_area_weighted,
    scale_to_fit,
)


def _empty_grid(size: int) -> list[list[float]]:
    return [[0.0] * size for _ in range(size)]


def test_bounding_box_of_an_empty_grid_is_none():

    assert bounding_box(_empty_grid(8)) is None


def test_bounding_box_of_a_single_point():

    grid = _empty_grid(8)
    grid[3][5] = 1.0

    assert bounding_box(grid) == (3, 5, 3, 5)


def test_bounding_box_of_a_rectangle():

    grid = _empty_grid(8)
    grid[2][1] = 1.0
    grid[2][4] = 1.0
    grid[6][1] = 1.0
    # a fourth interior point, shouldn't change the box
    grid[4][2] = 1.0

    assert bounding_box(grid) == (2, 1, 6, 4)


def test_crop_extracts_exactly_the_given_box():

    grid = [[float(r * 10 + c) for c in range(5)] for r in range(5)]

    cropped = crop(grid, (1, 2, 3, 4))

    assert cropped == [[12.0, 13.0, 14.0], [22.0, 23.0, 24.0], [32.0, 33.0, 34.0]]


def test_resize_area_weighted_integer_ratio_matches_block_averaging():

    # a 4x4 checkerboard of 2x2 blocks resized to 2x2 should exactly match averaging each block
    source = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]

    result = resize_area_weighted(source, 2, 2)

    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_resize_area_weighted_uniform_input_is_invariant_to_scale():

    source = [[1.0] * 5 for _ in range(5)]

    result = resize_area_weighted(source, 2, 2)

    assert result == [[1.0, 1.0], [1.0, 1.0]]


def test_resize_area_weighted_splits_a_single_source_pixel_across_two_output_rows():

    # 3x1 -> 2x1 (row_scale=1.5): the middle pixel (4.0) straddles both output rows.
    # Hand-derived: (0*1.0 + 4*0.5) / 1.5 = 4/3 and (4*0.5 + 8*1.0) / 1.5 = 20/3. A target
    # size of 1 splits nothing, and a mutation test showed it missed bugs
    source = [[0.0], [4.0], [8.0]]

    result = resize_area_weighted(source, 2, 1)

    assert result[0][0] == pytest.approx(4 / 3)
    assert result[1][0] == pytest.approx(20 / 3)


def test_resize_area_weighted_splits_a_single_source_pixel_across_two_output_pixels():

    # the column-direction version of the test above: 1x3 -> 1x2, the same 4/3 and 20/3
    source = [[0.0, 4.0, 8.0]]

    result = resize_area_weighted(source, 1, 2)

    assert result[0][0] == pytest.approx(4 / 3)
    assert result[0][1] == pytest.approx(20 / 3)


def test_scale_to_fit_preserves_aspect_ratio_for_a_tall_source():

    # 10 rows x 1 col, scaled so the longer dimension (rows) hits max_dimension=20 - the
    # narrower dimension scales by the same 2x factor, 1 -> 2, not forced to also become 20
    source = [[1.0] for _ in range(10)]

    result = scale_to_fit(source, max_dimension=20)

    assert len(result) == 20
    assert len(result[0]) == 2


def test_scale_to_fit_a_square_source_stays_square():

    source = [[1.0] * 7 for _ in range(7)]

    result = scale_to_fit(source, max_dimension=20)

    assert len(result) == 20
    assert len(result[0]) == 20


def test_scale_to_fit_clamps_floating_point_overshoot_to_the_source_range():

    # summed overlap fractions can overshoot the source range: a 1x1 all-ones source resized
    # to 1x3 gives 1.0000000000000002 unclamped, which intensity_to_color rejects (a real bug in
    # demo_mnist_ensemble_capture.py)
    source = [[1.0]]

    result = scale_to_fit(source, max_dimension=3)

    assert all(0.0 <= value <= 1.0 for row in result for value in row)


def test_center_of_mass_of_a_single_pixel_is_its_own_center():

    grid = _empty_grid(4)
    grid[2][3] = 1.0

    assert center_of_mass(grid) == (2.5, 3.5)


def test_center_of_mass_of_a_symmetric_block_is_the_grid_center():

    grid = _empty_grid(4)
    for row in (1, 2):
        for col in (1, 2):
            grid[row][col] = 1.0

    assert center_of_mass(grid) == (2.0, 2.0)


def test_center_of_mass_of_asymmetric_weights_is_hand_computed():

    # a weight of 1.0 at column 0 and 3.0 at column 2 (single row): col centroid =
    # (1*0.5 + 3*2.5) / 4 = 2.0; row centroid = (1*0.5 + 3*0.5) / 4 = 0.5
    grid = [[0.0, 0.0, 0.0, 0.0]]
    grid[0][0] = 1.0
    grid[0][2] = 3.0

    assert center_of_mass(grid) == (0.5, 2.0)


def test_center_of_mass_of_an_empty_grid_is_the_geometric_center():

    grid = _empty_grid(4)

    assert center_of_mass(grid) == (2.0, 2.0)


def test_place_centered_translates_by_the_center_of_mass():

    small_grid = [[1.0, 1.0], [1.0, 1.0]]  # centroid exactly (1.0, 1.0)

    placed = place_centered(small_grid, canvas_size=6)

    expected = _empty_grid(6)
    expected[2][2] = expected[2][3] = expected[3][2] = expected[3][3] = 1.0
    assert placed == expected


def test_place_centered_clips_content_that_does_not_fit():

    # larger than the canvas itself - content past the edge is dropped, not an error
    small_grid = [[1.0] * 5 for _ in range(5)]

    placed = place_centered(small_grid, canvas_size=4)

    assert len(placed) == 4 and len(placed[0]) == 4
    assert sum(sum(row) for row in placed) == 16.0  # less than the source's 25.0


def test_place_centered_of_an_empty_small_grid_is_an_empty_canvas():

    assert place_centered([], canvas_size=4) == _empty_grid(4)


def test_preprocess_capture_on_an_empty_grid_is_an_all_zero_canvas():

    result = preprocess_capture(_empty_grid(64))

    assert len(result) == CANVAS_SIZE
    assert len(result[0]) == CANVAS_SIZE
    assert all(value == 0.0 for row in result for value in row)


def test_preprocess_capture_centers_an_off_center_stroke():

    # a vertical stroke drawn near one edge of a 64x64 capture, not the center
    capture = _empty_grid(64)
    for row in range(10, 50):
        for col in (5, 6, 7):
            capture[row][col] = 1.0

    result = preprocess_capture(capture)

    result_centroid_row, result_centroid_col = center_of_mass(result)
    # centered on the 28x28 canvas - well within a couple pixels of dead center, unlike the
    # source capture's own centroid (column ~6, nowhere near 32 - the source's own center)
    assert result_centroid_col == pytest.approx(14.0, abs=1.5)
    assert result_centroid_row == pytest.approx(14.0, abs=1.5)
    assert any(value > 0.0 for row in result for value in row)


def _empty_capture_grid() -> list[list[float]]:
    return [[0.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]


def test_paint_brush_stroke_stamps_a_square_neighborhood_fully_on():

    grid = _empty_capture_grid()

    new_grid = paint_brush_stroke(grid, 30, 30, radius=2)

    for delta_row in range(-2, 3):
        for delta_col in range(-2, 3):
            assert new_grid[30 + delta_row][30 + delta_col] == 1.0

    assert new_grid[30 - 3][30] == 0.0
    assert new_grid[30][30 + 3] == 0.0
    assert grid[30][30] == 0.0  # the original grid is left untouched


def test_paint_brush_stroke_clips_to_the_grid_at_a_corner():

    grid = _empty_capture_grid()

    new_grid = paint_brush_stroke(grid, 0, 0, radius=2)

    on_count = sum(sum(row) for row in new_grid)
    assert on_count == 3 * 3
    assert new_grid[0][0] == 1.0
    assert new_grid[2][2] == 1.0
    assert new_grid[3][0] == 0.0


def test_paint_brush_stroke_default_radius_matches_capture_brush_radius():

    grid = _empty_capture_grid()

    new_grid = paint_brush_stroke(grid, 30, 30)

    stroke_width = 2 * CAPTURE_BRUSH_RADIUS + 1
    assert sum(sum(row) for row in new_grid) == stroke_width * stroke_width


def test_paint_brush_stroke_rejects_out_of_bounds_center():

    grid = _empty_capture_grid()

    with pytest.raises(AssertionError):
        paint_brush_stroke(grid, -1, 0)

    with pytest.raises(AssertionError):
        paint_brush_stroke(grid, 0, CAPTURE_GRID_SIZE)
