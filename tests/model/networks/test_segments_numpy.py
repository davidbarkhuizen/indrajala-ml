"""
Segments on numpy (the segments workplan, stage 4, D4-D6, D9, D11): the context from a batch's ids,
the mask in attention, positions per segment, the loss over counted targets, the pad's row, the
evaluation, and the equivalences: a packed segment gives what it gives alone, padded or not.
"""

import json
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.forward_context import NO_CONTEXT
from indrajala_ml.model.layers.numpy.segments import attention_blocked, segment_context
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, LayerSpec, Position, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.training.sequence_evaluate import sequence_evaluate

SYMBOLS = 5
VOCABULARY = SYMBOLS + 2  # the separator, then the pad
SEP, PAD = SYMBOLS, SYMBOLS + 1
T = 8


def _specs(positions: str) -> list[LayerSpec]:
    attention = Attention(heads=2, key_size=2, causal=True, positions="rotary" if positions == "rotary" else "none")
    return [
        Embedding(VOCABULARY, 4, segments=True),
        *([Position()] if positions == "learned" else []),
        Residual((LayerNorm(), attention)),
        Residual((LayerNorm(), Dense(6, activation="relu"), Dense(4, activation="linear", bias=True))),
        LayerNorm(),
        Dense(SYMBOLS + 1, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _network(positions: str = "learned", seed: int = 0) -> Any:
    network = SequentialArrayNetwork((T,), _specs(positions), Adam(), "sequence", NUMPY)
    network.rng = NUMPY.default_rng(seed)
    network.randomize()
    if positions == "learned":
        # a learned table that isn't zero, so a position read wrong shows
        position = next(layer for layer in network.layers if hasattr(layer, "P"))
        position.P = np.random.default_rng(seed + 1).normal(size=position.P.shape)
    return network


def _outputs(network: Any, rows: list[list[int]]) -> np.ndarray:
    X = np.array(rows, dtype=np.float64)
    (output,) = network.forward_rows(network.prepare_dataset([(tuple(row), tuple([0] * T)) for row in X]))
    return np.asarray(output).reshape(len(rows), T, SYMBOLS + 1)


def test_the_context_reads_segments_positions_and_counted_targets_from_the_ids():
    ids = np.array([[1, 2, SEP, 3, 4, SEP, 0, 1], [PAD, PAD, PAD, 2, 3, 4, SEP, 1]], dtype=np.float64)
    context = segment_context(ids, SEP, PAD)
    assert context.segments.tolist() == [[1, 1, 1, 2, 2, 2, 3, 3], [0, 0, 0, 1, 1, 1, 1, 2]]
    assert context.positions.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1], [0, 1, 2, 0, 1, 2, 3, 0]]
    assert context.counted.tolist() == [[1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1]]
    assert context.count == 10
    assert segment_context(np.array([[0, 1, 2, 3]], dtype=np.float64), SEP, PAD) is NO_CONTEXT


def test_attention_is_blocked_across_segments_and_from_pads_but_never_on_the_diagonal():
    blocked = attention_blocked(np.array([[0, 1, 1, 2]]))[0]
    assert blocked.tolist() == [
        [False, True, True, True],
        [True, False, False, True],
        [True, False, False, True],
        [True, True, True, False],
    ]


@pytest.mark.parametrize("positions", ["learned", "rotary"])
def test_a_segment_gives_the_same_outputs_whatever_was_packed_before_it(positions: str):
    network = _network(positions)
    paragraph = [3, 1, 4, SEP]
    first, other = [2, 2, 0, SEP], [1, 0, 4, SEP]
    outputs = _outputs(network, [first + paragraph, other + paragraph])
    np.testing.assert_allclose(outputs[0, 4:], outputs[1, 4:], rtol=0, atol=1e-12)
    assert not np.allclose(outputs[0, :4], outputs[1, :4])  # the first paragraphs differ, and so do theirs


@pytest.mark.parametrize("positions", ["learned", "rotary"])
def test_a_left_padded_rows_real_tokens_match_the_same_text_packed_after_another(positions: str):
    network = _network(positions)
    paragraph = [3, 1, 4, 1, SEP]
    packed = _outputs(network, [[2, 0, SEP, *paragraph]])
    padded = _outputs(network, [[PAD, PAD, PAD, *paragraph]])
    np.testing.assert_allclose(padded[0, 3:], packed[0, 3:], rtol=0, atol=1e-12)


def test_a_batch_without_special_ids_trains_by_bits_as_a_network_without_segments():
    rows = [((0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0), (1, 2, 3, 4, 0, 1, 2, 3))] * 3
    with_segments = _network("learned")
    plain = SequentialArrayNetwork((T,), [Embedding(VOCABULARY, 4), *_specs("learned")[1:]], Adam(), "sequence", NUMPY)
    plain.restore(with_segments.snapshot())
    for _ in range(2):
        with_segments.learn_batch(0.01, rows)
        plain.learn_batch(0.01, rows)
    for entry, plain_entry in zip(with_segments.snapshot(), plain.snapshot(), strict=True):
        for a, b in zip(entry, plain_entry, strict=True):
            assert np.array_equal(a, b)


def test_the_output_delta_is_the_mean_over_counted_targets_and_zero_elsewhere():
    network = _network("learned")
    rows = [
        ((1.0, 2.0, float(SEP), 3.0, 4.0, float(SEP), 0.0, 1.0), (2, SEP, 3, 4, SEP, 0, 1, 2)),
        ((float(PAD), float(PAD), 2.0, 3.0, float(SEP), 1.0, 2.0, 3.0), (0, 2, 3, SEP, 1, 2, 3, 4)),
    ]
    network.learn_batch(0.0, rows)  # a step at rate 0 leaves the network, and its deltas set
    output = network.output_layer
    delta = np.asarray(output.delta_batch).reshape(2, T, SYMBOLS + 1)
    counted = segment_context(np.array([state for state, _ in rows]), SEP, PAD).counted
    assert np.all(delta[counted == 0] == 0.0)
    A = np.asarray(output.A).reshape(2, T, SYMBOLS + 1)
    Y = np.zeros_like(A)
    for n, (_state, labels) in enumerate(rows):
        Y[n, np.arange(T), labels] = 1.0
    expected = (A - Y) * 2 / counted.sum()
    np.testing.assert_allclose(delta[counted == 1], expected[counted == 1], rtol=1e-15)


def test_the_pads_row_is_zero_and_stays_zero_through_training():
    network = _network("rotary")
    embedding = network.layers[0]
    assert np.all(embedding.E[PAD] == 0.0) and np.any(embedding.E[SEP] != 0.0)
    rows = [((float(PAD), float(PAD), 1.0, 2.0, float(SEP), 3.0, 4.0, 0.0), (0, 1, 2, SEP, 3, 4, 0, 1))] * 4
    for _ in range(3):
        network.learn_batch(0.05, rows)
    assert np.all(embedding.E[PAD] == 0.0)


def test_the_evaluation_counts_the_targets_the_loss_counts():
    network = _network("rotary")
    window = ((1.0, 2.0, float(SEP), 3.0, 4.0, float(SEP), 0.0, 1.0), (2, SEP, 3, 4, SEP, 0, 1, 2))
    evaluation = sequence_evaluate(network, [window])
    assert evaluation.tokens == 6
    probabilities = _outputs(network, [[int(x) for x in window[0]]])[0]
    kept = [t for t in range(T) if window[0][t] not in (SEP, PAD)]
    expected = -np.mean([np.log(probabilities[t, window[1][t]]) for t in kept])
    assert evaluation.cross_entropy == pytest.approx(expected, rel=1e-12)


def test_a_segmented_embedding_is_written_only_when_it_has_segments():
    assert layer_to_json(Embedding(7, 4)) == {"kind": "embedding", "vocabulary": 7, "size": 4}
    entry = json.loads(json.dumps(layer_to_json(Embedding(7, 4, segments=True))))
    assert entry["segments"] is True and layer_from_json(entry) == Embedding(7, 4, segments=True)
    with pytest.raises(ValueError, match="separator and a pad"):
        Embedding(2, 4, segments=True)


def test_segments_are_refused_in_pure_python_until_its_stage():
    with pytest.raises(NotImplementedError, match="stage 7"):
        SequentialSequenceBackpropNetwork((T,), _specs("rotary"), Adam())


def test_a_packed_batch_trains_and_its_loss_falls():
    network = _network("rotary", seed=3)
    rng = random.Random(0)
    text = [i for _ in range(40) for i in (*[rng.randrange(SYMBOLS) for _ in range(rng.randrange(2, 6))], SEP)]
    rows = [(tuple(float(i) for i in text[s : s + T]), tuple(text[s + 1 : s + T + 1])) for s in range(0, 120, T + 1)]
    before = sequence_evaluate(network, rows).cross_entropy
    for _ in range(30):
        network.learn_batch(0.01, rows)
    assert sequence_evaluate(network, rows).cross_entropy < before
