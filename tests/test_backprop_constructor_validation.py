from collections.abc import Callable

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork

# BackpropNetworkBase's own precondition checks (layer_sizes/dimension/input_bounds - see
# backprop_network_base.py and bounds.validate_input_bounds), verified identically on both
# direct subclasses here rather than duplicating each check once per subclass -
# MultiClassBackpropClassifierNetwork just needs one extra trailing positional (class_count).
NETWORK_CLASSES_WITH_EXTRA_ARGS = [
    (BackpropClassifierNetwork, ()),
    (MultiClassBackpropClassifierNetwork, (3,)),
]


@pytest.mark.parametrize("network_cls,extra_args", NETWORK_CLASSES_WITH_EXTRA_ARGS)
def test_layer_sizes_must_specify_at_least_one_hidden_layer(
    network_cls: Callable[..., object], extra_args: tuple[int, ...]
):

    with pytest.raises(AssertionError):
        network_cls([], 2, square_bounds(10.0), *extra_args)


@pytest.mark.parametrize("network_cls,extra_args", NETWORK_CLASSES_WITH_EXTRA_ARGS)
def test_every_hidden_layer_size_must_be_at_least_one(network_cls: Callable[..., object], extra_args: tuple[int, ...]):

    with pytest.raises(AssertionError):
        network_cls([4, 0], 2, square_bounds(10.0), *extra_args)


@pytest.mark.parametrize("network_cls,extra_args", NETWORK_CLASSES_WITH_EXTRA_ARGS)
def test_dimension_must_match_bounds_length(network_cls: Callable[..., object], extra_args: tuple[int, ...]):

    with pytest.raises(AssertionError):
        network_cls([4], 2, [(-1.0, 1.0)], *extra_args)


@pytest.mark.parametrize("network_cls,extra_args", NETWORK_CLASSES_WITH_EXTRA_ARGS)
def test_input_bounds_must_all_have_positive_width(network_cls: Callable[..., object], extra_args: tuple[int, ...]):

    with pytest.raises(AssertionError):
        network_cls([4], 2, [(-10.0, 10.0), (5.0, 5.0)], *extra_args)

    # inverted bounds (hi < lo) are equally nonsensical
    with pytest.raises(AssertionError):
        network_cls([4], 2, [(-10.0, 10.0), (5.0, -5.0)], *extra_args)


def test_class_count_must_be_at_least_two():
    # MultiClassBackpropClassifierNetwork-only precondition - BackpropClassifierNetwork has no
    # class_count argument to validate, so this isn't parametrized like the checks above

    with pytest.raises(AssertionError):
        MultiClassBackpropClassifierNetwork([4], 2, square_bounds(10.0), 1)
