"""
The encoder-decoder's first runs on the parallel corpora (next-steps.md, From cross-attention: "the
encoder-decoder has never trained on the corpora"): a probe, not a registered study. It asks
whether the model learns to translate at all, held-out bits per character falling and chrF above an
English-only model's, and what an epoch costs; it claims nothing (experiment-design.md, §2).

The model: d = 128, 4 heads of d_k = 32 (the cross-attention workplan's D12 floor), an encoder of
`blocks` bidirectional attention blocks and ReLU FFNs (4 d wide), a decoder of `blocks` blocks of
causal attention, cross-attention and FFN, each pre-LN residual, about 0.9M parameters at 2 blocks.
Both sides' windows are 1,024 (D11), at the token budget's batches of 2 windows, under the sequence
studies' recipe: warmup over the first epoch, then cosine to a tenth of the rate, AdamW 0.1,
clipping at 1.0, the training windows packed anew every epoch from the shuffled pairs.

`time` runs a few steps and reports seconds a step and an epoch's estimate; `run` trains, and after
each epoch evaluates the validation windows teacher-forced (bits per character, the counted
targets), the same windows with another window's source (the source ablation: the gap is what the
Greek buys) and a sample of training windows; at the end it translates a sample of validation pairs
greedily and scores them by chrF, keeping the translations. Results are JSON, gitignored, under
data/studies/translation-probe/.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from indrajala_ml.data.parallel_data import (
    PARALLEL_CORPORA,
    Pair,
    PairWindow,
    ParallelDataset,
    fits,
    load_pairs,
    load_parallel_dataset,
)
from indrajala_ml.generation.chrf import chrf
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.encoder_decoder_network import EncoderDecoderNetwork
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.networks.rust.encoder_decoder_rust_array_network import EncoderDecoderRustArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    CrossAttention,
    Dense,
    Embedding,
    EncoderDecoder,
    LayerNorm,
    Residual,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies.common import STUDIES_DIR
from indrajala_ml.training.lr_schedule import CosineDecay, Schedule, Warmup
from indrajala_ml.training.translation_evaluate import teacher_forced, translations

STUDY = "translation-probe"
WINDOW = 1024
BATCH = 2
WIDTH = 128
HEADS = 4
RATE = 0.002
WARMUP_EPOCHS = 1
FLOOR = 0.1
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
CLASSES = {"numpy": EncoderDecoderArrayNetwork, "rust": EncoderDecoderRustArrayNetwork}


def spec(
    source_symbols: int, target_symbols: int, width: int = WIDTH, heads: int = HEADS, blocks: int = 2
) -> EncoderDecoder:
    """The probe's model over source_symbols Greek and target_symbols English characters."""

    def ffn() -> Residual:
        return Residual((LayerNorm(), Dense(4 * width, "relu"), Dense(width, "linear", bias=True)))

    encoder = [Embedding(source_symbols + 2, width, segments=True)]
    for _ in range(blocks):
        encoder += [Residual((LayerNorm(), Attention(heads=heads))), ffn()]
    decoder = [Embedding(target_symbols + 2, width, segments=True, separator_opens=True)]
    for _ in range(blocks):
        decoder += [
            Residual((LayerNorm(), Attention(heads=heads, causal=True))),
            Residual((LayerNorm(), CrossAttention(heads=heads))),
            ffn(),
        ]
    return EncoderDecoder(
        (*encoder, LayerNorm()),
        (*decoder, LayerNorm(), Dense(target_symbols + 1, "softmax", output=True, loss="cross_entropy")),
    )


def examples(windows: Sequence[PairWindow]) -> list[Example[tuple[int, ...]]]:
    """Each window as a training example: its source's ids then its decoder input's, and its labels."""
    return [(tuple(float(i) for i in (*w.source, *w.decoder_input)), w.labels) for w in windows]


def schedule(epoch_steps: int, epochs: int, rate: float = RATE) -> Schedule:
    warmup = WARMUP_EPOCHS * epoch_steps
    return Warmup(warmup, CosineDecay(rate, max(1, epochs * epoch_steps - warmup), rate * FLOOR))


def network(data: ParallelDataset, seed: int, backend: str = "rust", blocks: int = 2) -> EncoderDecoderNetwork[Any]:
    """The probe's model for data, its weights drawn from seed."""
    net = CLASSES[backend](
        spec(len(data.source), len(data.target), blocks=blocks), WINDOW, WINDOW, Adam(weight_decay=WEIGHT_DECAY)
    )
    net.rng = (RUST if backend == "rust" else NUMPY).default_rng(seed)
    net.randomize()
    return net


def parameters(net: Any) -> int:
    """The network's trained values."""
    return sum(int(np.asarray(part.tolist()).size) for entry in net.snapshot() for part in entry)


def validation_pairs(corpus: str) -> list[Pair]:
    """The corpus's validation pairs that fit a window, in order: the ones its validation windows pack."""
    return [p for p in load_pairs(PARALLEL_CORPORA[corpus]) if p.split == "validation" and fits(p, WINDOW)]


def bits(cross_entropy: float) -> float:
    return cross_entropy / math.log(2.0)


def time_steps(corpus: str, seed: int, steps: int, backend: str = "rust", blocks: int = 2) -> dict[str, Any]:
    """steps training steps' seconds, and an epoch's estimate from the first epoch's window count."""
    data = load_parallel_dataset(corpus, WINDOW)
    net = network(data, seed, backend, blocks)
    windows = examples(data.train.epoch(random.Random(seed)))
    epoch_steps = math.ceil(len(windows) / BATCH)
    rate = schedule(epoch_steps, 1)
    seconds: list[float] = []
    for step in range(steps):
        start = time.perf_counter()
        net.learn_batch(rate(step), windows[step * BATCH : (step + 1) * BATCH], clip_norm=CLIP_NORM)
        seconds.append(time.perf_counter() - start)
    per_step = sorted(seconds[1:] or seconds)[len(seconds[1:] or seconds) // 2]
    return {
        "corpus": corpus,
        "backend": backend,
        "blocks": blocks,
        "parameters": parameters(net),
        "train_windows": len(windows),
        "epoch_steps": epoch_steps,
        "step_seconds": seconds,
        "median_step_seconds": per_step,
        "epoch_hours_estimate": per_step * epoch_steps / 3600,
    }


def run(
    corpus: str,
    seed: int,
    epochs: int,
    out: Path,
    backend: str = "rust",
    blocks: int = 2,
    sample: int = 32,
) -> dict[str, Any]:
    """epochs epochs, evaluated after each; the result and the final network written under out."""
    data = load_parallel_dataset(corpus, WINDOW)
    net = network(data, seed, backend, blocks)
    shuffle = random.Random(seed)
    first = examples(data.train.epoch(random.Random(seed)))
    epoch_steps = math.ceil(len(first) / BATCH)
    rate = schedule(epoch_steps, epochs)
    pad = len(data.target) + 1
    train_sample = data.train.epoch(random.Random(seed + 1))[:sample]
    swapped = swapped_sources(data.validation)
    result: dict[str, Any] = {
        "corpus": corpus,
        "seed": seed,
        "epochs": epochs,
        "backend": backend,
        "blocks": blocks,
        "parameters": parameters(net),
        "epoch_steps": epoch_steps,
        "validation_bits": [],
        "other_source_bits": [],
        "train_bits": [],
        "epoch_seconds": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{corpus}-seed{seed}-blocks{blocks}.json"
    step = 0
    for _ in range(epochs):
        windows = examples(data.train.epoch(shuffle))
        steps, seconds = bss.train_epoch(net, windows, BATCH, rate, step, shuffle, clip_norm=CLIP_NORM)
        step += steps
        result["epoch_seconds"].append(seconds)
        result["validation_bits"].append(bits(teacher_forced(net, data.validation, pad)[0]))
        # the source ablation: how much the Greek is read (swapped_sources)
        result["other_source_bits"].append(bits(teacher_forced(net, swapped, pad)[0]))
        result["train_bits"].append(bits(teacher_forced(net, train_sample, pad)[0]))
        path.write_text(json.dumps(result, indent=1))
        # the network after every epoch, so a stopped or inspected run loses nothing
        network_path = out / f"{corpus}-seed{seed}-blocks{blocks}.network.json"
        network_path.with_suffix(".partial").write_text(json.dumps(network_to_json(net)))
        network_path.with_suffix(".partial").rename(network_path)
    pairs = validation_pairs(corpus)[:sample]
    hypotheses = translations(net, pairs, data.source, data.target, Sampler(temperature=0.0), random.Random(seed))
    result["chrf_greedy"] = chrf(hypotheses, [pair.eng for pair in pairs])
    result["translations"] = [{"grc": p.grc, "eng": p.eng, "out": h} for p, h in zip(pairs, hypotheses, strict=True)]
    path.write_text(json.dumps(result, indent=1))
    return result


def swapped_sources(windows: Sequence[PairWindow]) -> list[PairWindow]:
    """
    Each window with another's source of as many pairs (the next such window, cyclically), so every
    target pair still has a source segment to read, but not its own; a window alone in its pair
    count keeps its source.
    """
    groups: dict[int, list[int]] = {}
    for i, window in enumerate(windows):
        groups.setdefault(window.pairs, []).append(i)
    out = list(windows)
    for indices in groups.values():
        for k, i in enumerate(indices):
            donor = windows[indices[(k + 1) % len(indices)]]
            out[i] = PairWindow(donor.source, windows[i].target, windows[i].pairs)
    return out


def source_ablation(network_path: Path, corpus: str) -> dict[str, Any]:
    """A trained network's validation bits per character with each window's own source, then with another's."""
    from indrajala_ml.model.persistence.load_network import load_network

    data = load_parallel_dataset(corpus, WINDOW)
    net = load_network(str(network_path))
    pad = len(data.target) + 1
    swapped = swapped_sources(data.validation)
    kept = sum(a.source == b.source for a, b in zip(data.validation, swapped, strict=True))
    own = bits(teacher_forced(net, data.validation, pad)[0])
    other = bits(teacher_forced(net, swapped, pad)[0])
    return {
        "own_source_bits": own,
        "other_source_bits": other,
        "gap": other - own,
        "windows": len(swapped),
        "unswapped": kept,
    }


OUT = STUDIES_DIR / STUDY
