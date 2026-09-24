class XORTarget:
    """
    Category 1.0 where (x > 0) != (y > 0): two diagonally opposite quadrants. A predicate
    exposing the same input_bounds/classify_state interface as LinearClassifierNetwork, so it
    drops straight into train.py/evaluate.py's machinery unchanged. Unlike a
    LinearClassifierNetwork target of the student's own architecture - representable by
    construction - this positive region isn't a union of convex regions any single-hidden-layer
    combination of half-planes can express, so no LinearClassifierNetwork gate can learn it.
    """

    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        self.input_bounds = bounds

    def classify_state(self, state: tuple[float, float]) -> float:
        x, y = state
        return 1.0 if (x > 0) != (y > 0) else 0.0
