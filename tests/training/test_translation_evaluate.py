"""
An encoder-decoder's evaluation (the cross-attention workplan, stage 8c, D6): the teacher-forced
cross-entropy is the training loss's (the mean over the counted targets), chrF scores the greedy
and the sampled translations of each pair's Greek, and each translation is the target's characters,
no longer than the window holds; with a seeded generator, the evaluation repeats by bits.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.specs.update_rules import SGD
from indrajala_ml.training.translation_evaluate import translation_evaluate, translations
from tests.model.networks import test_encoder_decoder_numpy as numpy_tests


def network() -> EncoderDecoderArrayNetwork:
    return EncoderDecoderArrayNetwork.randomized(
        numpy_tests.spec(), numpy_tests.WINDOW, numpy_tests.WINDOW, SGD(), seed=2
    )


def test_the_teacher_forced_cross_entropy_is_the_training_loss():
    net = network()
    windows = numpy_tests.windows()
    evaluation = translation_evaluate(
        net, windows, numpy_tests.PAIRS, numpy_tests.SOURCE, numpy_tests.TARGET, np.random.default_rng(0)
    )
    assert evaluation.cross_entropy == pytest.approx(numpy_tests.loss(net, numpy_tests.batch(windows)), rel=1e-12)
    assert evaluation.bits_per_character == pytest.approx(evaluation.cross_entropy / math.log(2.0))
    assert evaluation.characters == sum(len(p.eng) + 1 for p in numpy_tests.PAIRS)
    assert 0.0 <= evaluation.chrf_greedy <= 100.0 and 0.0 <= evaluation.chrf_sampled <= 100.0
    assert evaluation.pairs == len(numpy_tests.PAIRS)


def test_translations_are_the_targets_characters_within_the_window_and_repeat_from_a_seed():
    net = network()
    first, again = (
        translations(
            net, numpy_tests.PAIRS, numpy_tests.SOURCE, numpy_tests.TARGET, Sampler(), np.random.default_rng(5), batch=3
        )
        for _ in range(2)
    )
    assert first == again and len(first) == len(numpy_tests.PAIRS)
    assert all(set(text) <= set(numpy_tests.TARGET.symbols) and len(text) <= numpy_tests.WINDOW - 1 for text in first)
