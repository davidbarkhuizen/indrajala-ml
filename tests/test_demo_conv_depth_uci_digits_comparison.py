from indrajala_ml.demos import demo_conv_depth_uci_digits_comparison as demo
from indrajala_ml.digits_data import load_digits_dataset


def test_parameter_counts_match_a_hand_count():

    # conv1:      conv 8*(3*3*1 + 1) = 80,   dense 32*(8*6*6 + 1) = 9248, output 10*(32 + 1) = 330
    # conv2:      conv 80 + 8*(3*3*8 + 1) = 664,  dense 32*(8*4*4 + 1) = 4128, output 330
    # conv2-wide: conv 80 + 16*(3*3*8 + 1) = 1248, dense 32*(16*4*4 + 1) = 8224, output 330
    # dense:      hidden 32*(64 + 1) = 2080, output 330
    assert demo.parameter_count(demo._build("conv1")) == 80 + 9248 + 330
    assert demo.parameter_count(demo._build("conv2")) == 664 + 4128 + 330
    assert demo.parameter_count(demo._build("conv2-wide")) == 1248 + 8224 + 330
    assert demo.parameter_count(demo._build("dense")) == 2080 + 330
    # conv1-pool and conv1-stride2 both end at 3x3x8: conv 80, dense 32*(8*3*3 + 1) = 2336,
    # output 330 - the pool layer itself adds no parameters
    assert demo.parameter_count(demo._build("conv1-pool")) == 80 + 2336 + 330
    assert demo.parameter_count(demo._build("conv1-stride2")) == 80 + 2336 + 330


def test_run_one_is_deterministic_per_seed(monkeypatch):

    # a small subset and one epoch keep this fast; the same (config, seed) must reproduce
    # exactly, since the demo's paired comparison relies on each seed fixing split and init
    monkeypatch.setattr(demo, "EPOCHS", 1)
    subset = load_digits_dataset()[:60]

    first = demo.run_one(subset, "conv1-pool", seed=3)
    second = demo.run_one(subset, "conv1-pool", seed=3)

    assert (first["train"], first["test"]) == (second["train"], second["test"])
    assert 0.0 <= first["test"] <= 1.0
