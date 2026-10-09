"""
A sequence network's evaluation on held-out windows (the sequence task workplan, D8): the mean
per-token cross-entropy, in nats and in bits per token (bits per character on a character-level
corpus), and the per-token accuracy, every token of every window counted once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from indrajala_ml.model.protocols.classifier_protocols import Example


@dataclass(frozen=True)
class SequenceEvaluation:
    """The mean per-token cross-entropy in nats, and the per-token accuracy, over tokens tokens."""

    cross_entropy: float
    accuracy: float
    tokens: int

    @property
    def bits_per_token(self) -> float:
        return self.cross_entropy / math.log(2.0)


def sequence_evaluate(network: Any, examples: Sequence[Example[tuple[int, ...]]]) -> SequenceEvaluation:
    """
    network, a sequence network on either array backend, on examples: its inference forward pass
    over them in chunks (forward_rows), each token's -log of the probability it gives the token's
    label, and whether its argmax is the label. A probability that underflows to 0 is an infinite
    loss.
    """
    assert len(examples) >= 1, "examples must not be empty"
    prepared = network.prepare_dataset(examples)
    classes = network.class_count
    labels = np.array([label for _state, window in examples for label in window])
    assert labels.size == len(examples) * network.tokens, f"each window's labels are {network.tokens}, one per token"
    loss = 0.0
    correct = 0
    start = 0
    for output_batch in network.forward_rows(prepared):
        probabilities = np.asarray(output_batch.tolist()).reshape(-1, classes)
        stop = start + probabilities.shape[0]
        chunk = labels[start:stop]
        with np.errstate(divide="ignore"):
            loss += float(-np.log(probabilities[np.arange(chunk.size), chunk]).sum())
        correct += int((probabilities.argmax(axis=1) == chunk).sum())
        start = stop
    return SequenceEvaluation(loss / labels.size, correct / labels.size, int(labels.size))
