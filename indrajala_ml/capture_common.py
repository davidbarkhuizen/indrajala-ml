def stamp_brush(grid: list[list[float]], row: int, col: int, radius: int) -> list[list[float]]:
    """
    A copy of grid with a square brush of the given radius stamped fully on at (row, col), clipped
    to the grid. Shared by digit_capture.paint_brush_stroke (32x32 grid, radius 2) and
    mnist_capture.paint_brush_stroke (64x64, radius 4), which add their own validation.
    """

    grid_size = len(grid)
    assert grid_size >= 1 and all(len(r) == grid_size for r in grid), "grid must be square"
    assert 0 <= row < grid_size and 0 <= col < grid_size, f"(row, col) must be within the grid; got ({row}, {col})"

    new_grid = [list(r) for r in grid]
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            brush_row, brush_col = row + delta_row, col + delta_col
            if 0 <= brush_row < grid_size and 0 <= brush_col < grid_size:
                new_grid[brush_row][brush_col] = 1.0

    return new_grid
