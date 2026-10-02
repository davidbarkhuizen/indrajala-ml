"""
Format 2's layer entries (model/persistence/format2_json.py): an entry added to a spec after files
were saved without it is left out at its default, so those files are what such a network saves now.
"""

from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import BatchNorm, Conv


def test_a_batch_norm_entry_holds_a_group_size_only_when_it_has_one():
    # no "group_size" without ghost groups: files saved before BatchNorm had one are what such a
    # network saves now, and they load with the default
    plain = {"kind": "batch_norm", "activation": "relu", "epsilon": 1e-5, "running_rate": 0.1}
    assert layer_to_json(BatchNorm("relu")) == plain
    assert layer_to_json(BatchNorm("relu", group_size=32)) == {**plain, "group_size": 32}
    assert layer_from_json(plain) == BatchNorm("relu")
    assert layer_from_json({**plain, "group_size": 32}) == BatchNorm("relu", group_size=32)


def test_a_relu_conv_entry_is_unchanged():
    # no "activation": files saved before ConvSpec had one are what a ReLU conv network saves now
    assert layer_to_json(Conv(3, 8, stride=2)) == {"kind": "conv", "kernel_size": 3, "channel_count": 8, "stride": 2}
