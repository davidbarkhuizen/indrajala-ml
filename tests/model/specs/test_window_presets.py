"""The window-recurrent network's windows as layer specs (indrajala_ml/model/specs/window_presets.py)."""

from typing import Any

import pytest

from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Branches,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Residual,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import OUTPUT, AttentionPooling, Codes, Factor, WindowRecurrence

TOKENS, CLASSES = 27, 3  # the window-recurrent adder's column tokens and digits


def _inner(specs: list[LayerSpec]) -> list[Any]:
    return [part for spec in specs for part in (spec.body if isinstance(spec, Residual) else (spec,))]


def _heads(specs: tuple[LayerSpec, ...]) -> list[int]:
    return [part.heads for part in _inner(list(specs)) if isinstance(part, Attention)]


def test_the_transformer_window_is_w11s_and_its_width_widens_the_stream():
    window = wp.transformer_window(TOKENS, CLASSES)
    assert window == wp.transformer_window(TOKENS, CLASSES, wp.D)
    assert sum(isinstance(spec, StateInput) for spec in window) == wp.BLOCKS
    assert _heads(tuple(window)) == [wp.HEADS] * wp.BLOCKS
    assert window[-1] == Dense(CLASSES, output=True, activation="softmax", loss="cross_entropy")
    wide = wp.transformer_window(9, 14, 32)
    assert wide[0] == Embedding(9, 32) and wide[4] == Residual(
        (LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))
    )


def test_the_linear_and_ffn_windows_drop_the_attention_or_every_block():
    linear, ffn = _inner(wp.linear_window(TOKENS, CLASSES)), _inner(wp.ffn_window(TOKENS, CLASSES))
    assert not any(isinstance(spec, Attention) for spec in linear + ffn)
    assert sum(isinstance(spec, StateInput) for spec in ffn) == wp.BLOCKS
    assert sum(isinstance(spec, StateInput) for spec in linear) == 1
    assert all(getattr(spec, "activation", "linear") in ("linear", "softmax") for spec in linear)


@pytest.mark.parametrize("heads", [1, 2])
def test_the_branched_window_shares_a_trunk_and_branches_each_factor_and_the_answers(heads: int):
    window = wp.transformer_window(TOKENS, CLASSES, 32)
    branched = wp.branched_window(TOKENS, CLASSES, 2, 32, heads)
    branches = branched[-2]
    assert branched[:-2] == window[: 2 + 3 * (wp.BLOCKS - 1)] and branched[-1] == window[-1]
    assert isinstance(branches, Branches) and len(branches.branches) == 3 and not branches.merge
    assert all(_heads(branch) == [heads] for branch in branches.branches)


def test_the_recombining_window_merges_the_factors_branches_into_w11s_last_block():
    window = wp.transformer_window(TOKENS, CLASSES, 32)
    recombined = wp.recombined_window(TOKENS, CLASSES, 2, 32)
    branches = recombined[8]
    assert recombined[:8] == window[:8] and recombined[9:] == window[-5:]
    assert isinstance(branches, Branches) and branches.merge and len(branches.branches) == 2
    assert all(_heads(branch) == [wp.HEADS] for branch in branches.branches)


def test_the_answer_branch_window_reads_every_factor_from_the_state_branch():
    window = wp.transformer_window(TOKENS, CLASSES)
    answers = wp.answer_branch_window(TOKENS, CLASSES, 3)
    branches = answers[-2]
    assert isinstance(branches, Branches) and branches.reads == (0, 0, 0)
    assert branches.branches[0] == tuple(window[8:-1]) and _heads(branches.branches[1]) == [1]


@pytest.mark.parametrize(
    "window",
    [
        wp.transformer_window(TOKENS, CLASSES),
        wp.linear_window(TOKENS, CLASSES),
        wp.ffn_window(TOKENS, CLASSES),
        wp.branched_window(TOKENS, CLASSES, 2),
        wp.recombined_window(TOKENS, CLASSES, 2),
        wp.answer_branch_window(TOKENS, CLASSES, 2),
    ],
    ids=["transformer", "linear", "ffn", "branched", "recombined", "answer-branch"],
)
def test_every_window_builds_a_network(window: list[LayerSpec]):
    recurrence = WindowRecurrence((Factor(16, Codes(8)), Factor(16, Codes(8))), False, pooling=AttentionPooling(OUTPUT))
    network = WindowRecurrentRustArrayNetwork.randomized((1,), window, recurrence, Adam(), seed=0)
    assert network.recurrence.snapped
