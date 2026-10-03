import matplotlib

matplotlib.use("Agg")


from indrajala_ml.graphics.chart import disagreement_axis_bounds


def test_disagreement_axis_bounds():

    assert disagreement_axis_bounds(500.0) == [(0.0, 500.0), (0.0, 1.0)]
    assert disagreement_axis_bounds(500.0, log=True) == [(0.0, 500.0), (1.0e-3, 1.0)]
