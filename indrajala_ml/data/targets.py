class XORTarget:
    """
    Category 1.0 where (x > 0) != (y > 0): two diagonally opposite quadrants. It has
    LinearClassifierNetwork's input_bounds/classify_state interface, so train.py and evaluate.py use
    it as a target. No LinearClassifierNetwork gate can learn it: the region isn't a combination of
    half-planes one hidden layer can express.
    """

    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        self.input_bounds = bounds

    def classify_state(self, state: tuple[float, ...]) -> float:
        x, y = state
        return 1.0 if (x > 0) != (y > 0) else 0.0
