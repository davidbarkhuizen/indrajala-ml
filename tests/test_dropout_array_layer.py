from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from tests.helpers import Backend, approx

LayerCls = type[DropoutArrayLayer] | type[DropoutRustArrayLayer]
LAYER_CLS: dict[str, LayerCls] = {"numpy": DropoutArrayLayer, "rust": DropoutRustArrayLayer}
BASE_LAYER_CLS = {"numpy": ArrayLayer, "rust": RustArrayLayer}

# test_dropout_layer.py's DropoutNode fixture: z = 1.1, base = sigmoid(1.1)
W = [[0.5]]
B = [0.1]
X = [2.0]
BASE_ACTIVATION = 0.7502601055951177


@pytest.fixture
def layer_cls(backend: Backend) -> LayerCls:
    return LAYER_CLS[backend.name]


def _dropout_layer(backend: Backend, drop_probability: float = 0.5):
    layer = LAYER_CLS[backend.name](1, 1, drop_probability)
    layer.W = backend.owned(W)
    layer.b = backend.owned(B)
    return layer


# Any: paired with a layer of the same backend, which a union can't express
def _next_layer(backend: Backend) -> Any:
    next_layer = BASE_LAYER_CLS[backend.name](1, 1)
    next_layer.W = backend.owned([[0.8]])
    next_layer.delta = backend.owned([-0.5])
    return next_layer


# seeds whose first draw keeps (0.5488... >= 0.5) or drops (0.4170... < 0.5) the one unit
KEEP_SEED, DROP_SEED = 0, 1


def _forward_with_outcome(layer: DropoutArrayLayer | DropoutRustArrayLayer, backend: Backend, kept: bool) -> Any:
    # a training forward pass whose one unit is kept or dropped: both backends draw numpy's
    # stream, so one seed gives the same outcome on each
    backend.seed(KEEP_SEED if kept else DROP_SEED)
    result = layer.forward(backend.owned(X))
    assert layer._mask.tolist() == [1.0 if kept else 0.0]
    return result


def test_forward_at_eval_mode_matches_a_plain_sigmoid_no_rescale(backend: Backend):

    # training defaults to False
    layer = _dropout_layer(backend)

    result = layer.forward(backend.owned(X))

    plain_layer = BASE_LAYER_CLS[backend.name](1, 1)
    plain_layer.W = backend.owned(W)
    plain_layer.b = backend.owned(B)
    expected = plain_layer.forward(backend.owned(X))

    assert np.allclose(result.tolist(), expected.tolist())
    assert np.allclose(result.tolist(), [BASE_ACTIVATION])


def test_forward_in_training_mode_when_kept_rescales_by_one_over_keep_probability(backend: Backend):

    layer = _dropout_layer(backend, drop_probability=0.5)
    layer.set_training_mode(True)

    result = _forward_with_outcome(layer, backend, kept=True)

    assert np.allclose(result.tolist(), [BASE_ACTIVATION / 0.5])


def test_forward_in_training_mode_when_dropped_is_exactly_zero(backend: Backend):

    layer = _dropout_layer(backend, drop_probability=0.5)
    layer.set_training_mode(True)

    result = _forward_with_outcome(layer, backend, kept=False)

    assert result.tolist()[0] == 0.0


def test_forward_and_hidden_delta_in_training_mode_are_internally_consistent_across_many_draws(backend: Backend):

    # each draw against the mask the layer returns: a kept unit is base / keep_probability, a
    # dropped one exactly 0.0. 200 draws at 0.5 from a fixed seed include both outcomes
    layer = _dropout_layer(backend, drop_probability=0.5)
    layer.set_training_mode(True)
    backend.seed(0)

    saw_kept = False
    saw_dropped = False
    for _ in range(200):
        result = layer.forward(backend.owned(X)).tolist()
        mask_value = layer._mask.tolist()[0]
        assert mask_value in (0.0, 1.0)
        if mask_value == 1.0:
            saw_kept = True
            assert result[0] == approx(BASE_ACTIVATION / 0.5)
        else:
            saw_dropped = True
            assert result[0] == 0.0
        assert layer._base_activation.tolist()[0] == approx(BASE_ACTIVATION)

    assert saw_kept and saw_dropped


def test_set_training_mode_false_reverts_to_eval_behavior(backend: Backend):

    layer = _dropout_layer(backend)
    layer.set_training_mode(True)
    result = _forward_with_outcome(layer, backend, kept=False)
    assert result.tolist()[0] == 0.0

    layer.set_training_mode(False)
    result = layer.forward(backend.owned(X))

    assert np.allclose(result.tolist(), [BASE_ACTIVATION])


def test_compute_hidden_delta_when_kept_uses_the_unscaled_sigmoid_derivative(backend: Backend):

    # the derivative is base*(1-base) on the activation before the rescale, not a*(1-a)
    layer = _dropout_layer(backend, drop_probability=0.5)
    layer.set_training_mode(True)
    _forward_with_outcome(layer, backend, kept=True)

    layer.compute_hidden_delta(_next_layer(backend))

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative / 0.5
    assert np.allclose(layer.delta.tolist(), [expected])


def test_compute_hidden_delta_is_zero_when_the_unit_was_dropped(backend: Backend):

    layer = _dropout_layer(backend, drop_probability=0.5)
    layer.set_training_mode(True)
    _forward_with_outcome(layer, backend, kept=False)

    layer.compute_hidden_delta(_next_layer(backend))

    assert layer.delta.tolist()[0] == 0.0


def test_compute_hidden_delta_at_eval_mode_uses_no_rescale(backend: Backend):

    layer = _dropout_layer(backend)
    layer.forward(backend.owned(X))

    layer.compute_hidden_delta(_next_layer(backend))

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative
    assert np.allclose(layer.delta.tolist(), [expected])


def test_forward_batch_draws_an_independent_mask_per_row_not_one_shared_per_batch(
    layer_cls: LayerCls, backend: Backend
):

    layer = layer_cls(4, 3, drop_probability=0.5)
    layer.W = backend.owned(np.random.default_rng(0).uniform(-1.0, 1.0, size=(4, 3)).tolist())
    layer.b = backend.owned(np.random.default_rng(1).uniform(-1.0, 1.0, size=4).tolist())
    layer.set_training_mode(True)

    X_batch = backend.owned(np.random.default_rng(2).uniform(-1.0, 1.0, size=(20, 3)).tolist())
    layer.forward_batch(X_batch)

    rows = layer._mask_batch.tolist()
    assert (len(rows), len(rows[0])) == (20, 4)
    assert len({tuple(row) for row in rows}) > 1


def test_forward_batch_at_eval_mode_matches_forward_per_row_stacked(layer_cls: LayerCls, backend: Backend):

    layer = layer_cls(3, 2, drop_probability=0.5)
    layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.b = backend.owned([0.1, 0.2, 0.3])

    x_rows = [[1.0, -1.0], [0.5, 0.5], [-2.0, 3.0]]
    expected_rows = [layer.forward(backend.owned(row)).tolist() for row in x_rows]

    result = layer.forward_batch(backend.owned(x_rows))

    assert np.allclose(result.tolist(), expected_rows)


def test_compute_hidden_delta_batch_at_eval_mode_matches_per_row_single_example_results(
    layer_cls: LayerCls, backend: Backend
):

    layer = layer_cls(3, 2, drop_probability=0.5)
    layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.b = backend.owned([0.1, 0.2, 0.3])

    next_layer: Any = BASE_LAYER_CLS[backend.name](
        2, 3
    )  # Any: paired with a layer of the same backend, which a union can't express
    next_layer.W = backend.owned([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    delta_rows = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.1]]
    next_layer.delta_batch = backend.owned(delta_rows)

    x_rows = [[1.0, -1.0], [0.5, 0.5], [-2.0, 3.0]]
    layer.forward_batch(backend.owned(x_rows))

    expected_rows: list[list[float]] = []
    for row_index in range(3):
        row_layer = layer_cls(3, 2, drop_probability=0.5)
        row_layer.W = layer.W
        row_layer.b = layer.b
        row_layer.forward(backend.owned(x_rows[row_index]))
        next_layer.delta = backend.owned(delta_rows[row_index])
        row_layer.compute_hidden_delta(next_layer)
        expected_rows.append(row_layer.delta.tolist())

    layer.compute_hidden_delta_batch(next_layer)

    assert np.allclose(layer.delta_batch.tolist(), expected_rows)


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer(backend: Backend):

    layer = _dropout_layer(backend)
    layer.W = backend.owned([[1.0]])
    layer.b = backend.owned([5.0])
    layer.delta = backend.owned([1.0])
    layer.accumulate_gradient(backend.owned([1.0]))
    layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(layer.W.tolist(), [[0.9]])
    assert np.allclose(layer.b.tolist(), [4.9])


def test_drop_probability_of_one_is_rejected(layer_cls: LayerCls):

    with pytest.raises(AssertionError):
        layer_cls(1, 1, drop_probability=1.0)
