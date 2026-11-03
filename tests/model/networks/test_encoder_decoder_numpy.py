"""
The numpy encoder-decoder network (the cross-attention workplan, stage 4b, D2): its gradients
against finite differences through both stacks, the memory's gradient folded over two
CrossAttention layers; training on packed pair windows lowers the loss; a checkpoint restores it by
bits; and what it refuses.
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.parallel_data import Pair, PairWindow, pack_pairs
from indrajala_ml.data.text_data import Vocabulary
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.protocols.array_protocols import TrainedArrayLayer
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    CrossAttention,
    Dense,
    Embedding,
    EncoderDecoder,
    LayerNorm,
    Residual,
)
from indrajala_ml.model.specs.update_rules import SGD, Adam

SOURCE, TARGET = Vocabulary.of("αβγδ"), Vocabulary.of("abcd")
WINDOW = 10
PAIRS = [
    Pair("w", str(k), "train", grc, eng)
    for k, (grc, eng) in enumerate(
        [("αβ", "ab"), ("γ", "c"), ("δα", "da"), ("ββγ", "bbc"), ("α", "a"), ("γδ", "cd"), ("δ", "d"), ("βα", "ba")]
    )
]


def spec(memory_width: int = 8, width: int = 6) -> EncoderDecoder:
    def ffn(d: int) -> Residual:
        return Residual((LayerNorm(), Dense(2 * d, "relu"), Dense(d, "linear", bias=True)))

    return EncoderDecoder(
        (
            Embedding(len(SOURCE) + 2, memory_width, segments=True),
            Residual((LayerNorm(), Attention(heads=2))),
            ffn(memory_width),
            LayerNorm(),
        ),
        (
            Embedding(len(TARGET) + 2, width, segments=True, separator_opens=True),
            Residual((LayerNorm(), Attention(heads=2, causal=True))),
            Residual((LayerNorm(), CrossAttention(heads=2))),
            ffn(width),
            Residual((LayerNorm(), CrossAttention(heads=2, key_value_heads=1))),
            LayerNorm(),
            Dense(len(TARGET) + 1, "softmax", output=True, loss="cross_entropy"),
        ),
    )


def windows() -> list[PairWindow]:
    return pack_pairs(PAIRS, SOURCE, TARGET, WINDOW)


def network(rule: Any = None, seed: int = 0) -> EncoderDecoderArrayNetwork:
    return EncoderDecoderArrayNetwork.randomized(spec(), WINDOW, WINDOW, rule or SGD(), seed=seed)


def batch(ws: list[PairWindow]) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    return [(tuple(float(i) for i in (*w.source, *w.decoder_input)), w.labels) for w in ws]


def loss(net: EncoderDecoderArrayNetwork, examples: list[tuple[tuple[float, ...], tuple[int, ...]]]) -> float:
    # the mean over the batch's counted targets (those whose input isn't a pad) of their cross-entropy
    X = np.array([state for state, _ in examples])
    P = net._forward_batch(X).reshape(len(examples), WINDOW, -1)  # pyright: ignore[reportPrivateUsage]
    pad = len(TARGET) + 1
    total, count = 0.0, 0
    for n, (state, labels) in enumerate(examples):
        for t, label in enumerate(labels):
            if state[WINDOW + t] != pad:
                total -= float(np.log(P[n, t, label]))
                count += 1
    return total / count


class _Recorder:
    # the network's optimizer for one step: keeps every layer's accumulated gradients
    def __init__(self) -> None:
        self.gradients: list[list[FloatArray]] = []

    def step(self, layers: Any, learning_rate: float, batch_size: int, clip_norm: float | None = None) -> None:
        self.gradients = [
            [g.copy() for g in layer.gradients()] if hasattr(layer, "gradients") else [] for layer in layers
        ]
        for layer in layers:
            if hasattr(layer, "reset_gradient_accum"):
                layer.reset_gradient_accum()


def test_the_gradients_through_both_stacks_are_finite_differences():
    net = network()
    examples = batch(windows())
    recorder = _Recorder()
    net.optimizer = recorder  # pyright: ignore[reportAttributeAccessIssue]
    net.learn_batch(1.0, examples)
    rng = np.random.default_rng(0)
    checked = 0
    for index, layer in enumerate(net.layers):
        if not isinstance(layer, TrainedArrayLayer):
            continue
        parameters = layer.parameters()
        for parameter, gradient in zip(parameters, recorder.gradients[index], strict=True):
            for flat in rng.choice(parameter.size, size=min(2, parameter.size), replace=False):
                at = tuple(int(i) for i in np.unravel_index(int(flat), parameter.shape))
                saved = parameter[at]
                parameter[at] = saved + 1e-6
                up = loss(net, examples)
                parameter[at] = saved - 1e-6
                down = loss(net, examples)
                parameter[at] = saved
                # the accumulated gradient is the batch's sum, N times the mean loss's
                assert float(gradient[at]) / len(examples) == pytest.approx((up - down) / 2e-6, rel=2e-5, abs=1e-8), (
                    f"layer {index} ({type(layer).__name__}) parameter at {at}"
                )
                checked += 1
    # the encoder's layers got their gradients from the memory
    encoder = len(net.encoder_layers)
    assert any(np.abs(g).sum() > 0 for grads in recorder.gradients[:encoder] for g in grads)
    assert checked > 60


def test_training_on_packed_windows_lowers_the_loss_and_classifies_per_token():
    net = network(Adam(), seed=1)
    examples = batch(windows())
    before = loss(net, examples)
    for _ in range(60):
        net.learn_batch(0.01, examples)
    assert loss(net, examples) < 0.5 * before
    prediction = net.classify_state(examples[0][0])
    assert len(prediction) == WINDOW and all(0 <= c <= len(TARGET) for c in prediction)


def test_a_checkpoint_restores_the_network_by_bits():
    net = network(Adam(), seed=2)
    examples = batch(windows())
    net.learn_batch(0.01, examples)
    checkpoint = net.checkpoint()
    net.learn_batch(0.01, examples)
    after = net._forward_batch(np.array([s for s, _ in examples]))  # pyright: ignore[reportPrivateUsage]
    net.restore_checkpoint(checkpoint)
    net.learn_batch(0.01, examples)
    again = net._forward_batch(np.array([s for s, _ in examples]))  # pyright: ignore[reportPrivateUsage]
    np.testing.assert_array_equal(after, again)


def test_the_same_seed_draws_the_same_network_and_a_one_example_step_is_a_batch_of_one():
    a, b = network(seed=3), network(seed=3)
    example = batch(windows())[0]
    a.learn(0.1, *example)
    b.learn_batch(0.1, [example])
    np.testing.assert_array_equal(a.predict_probabilities(example[0]), b.predict_probabilities(example[0]))


def test_it_refuses_a_state_of_the_wrong_width_and_saving_before_stage_4c(tmp_path: Any):
    net = network()
    with pytest.raises(AssertionError, match="source ids then its"):
        net.classify_state((0.0,) * WINDOW)
    with pytest.raises(NotImplementedError, match="stage 4c"):
        net.save(str(tmp_path / "net.json"))
