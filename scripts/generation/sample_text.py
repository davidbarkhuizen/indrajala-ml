"""
Sampling text from a trained sequence model (the generation workplan, D5, D6): `train` trains the
sequence study's 2-layer transformer on a corpus and saves it with its tokenizer file; `sample`
loads one and samples from a prompt through the cached decoder.

    python scripts/generation/sample_text.py train --corpus tinyshakespeare --epochs 10 --seed 0 --out data/generation/shakespeare.json
    python scripts/generation/sample_text.py sample --model data/generation/shakespeare.json --prompt "ROMEO:" --length 300 \\
        --temperature 0.8 --top-k 20 --seed 1

`train`: the sequence study's 2-layer arm (indrajala_ml/studies/sequence_study.py), Adam at batch 32,
at the study's tuned rate for the corpus (rotary_study.RATES), with rotary attention and no Position
by default (the rotary-positions workplan's D9; `--positions learned` for the study's own arm), and
`--dropout P` on its attention weights and residual paths (the attention-dropout study's "both" arms).
numpy or `--backend rust`. It prints each epoch's held-out bits per character (the last 10% of the
text held out) and saves the model in format 2, its tokenizer beside it (`<model>.tokenizer.json`).

`sample`: the model on its own backend, the prompt encoded by its tokenizer, then `--length`
characters, each drawn by the sampler's chain (temperature, top-k, top-p) from one generator seeded
by `--seed`: one model file and one seed give the same text.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

from indrajala_ml.data import text_data
from indrajala_ml.generation.generate import cached, generate
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.generation.tokenizer import CharacterTokenizer, tokenizer_path
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Dropout, Embedding, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import rotary_study
from indrajala_ml.studies import sequence_study as ss
from indrajala_ml.training.sequence_evaluate import sequence_evaluate

ARM = "2-layer"


def model_specs(vocabulary: int, positions: str, dropout: float) -> list[LayerSpec]:
    """The sequence study's 2-layer arm, rotary or learned, with dropout on attention and the residual paths."""
    if dropout == 0.0:
        return rotary_study.arm_specs(ARM, vocabulary, positions)
    assert positions == "rotary", "--dropout builds a rotary model (the attention-dropout arms were learned)"
    rotary = "rotary"
    attention_block = Residual(
        (LayerNorm(), Attention(heads=ss.HEADS, causal=True, dropout=dropout, positions=rotary), Dropout(dropout))
    )
    ffn_block = Residual(
        (
            LayerNorm(),
            Dense(ss.FFN_SIZE, activation="relu"),
            Dense(ss.TOKEN_SIZE, activation="linear", bias=True),
            Dropout(dropout),
        )
    )
    return [
        Embedding(vocabulary, ss.TOKEN_SIZE),
        *[attention_block, ffn_block] * 2,
        LayerNorm(),
        Dense(vocabulary, output=True, activation="softmax", loss="cross_entropy"),
    ]


def train(args: argparse.Namespace) -> None:
    text = text_data.load_text(text_data.CORPORA[args.corpus])
    tokenizer = CharacterTokenizer.of(args.corpus, text)
    train_windows, held_out, vocabulary = ss.load(args.corpus, args.limit)
    assert vocabulary == len(tokenizer)
    backend = RUST if args.backend == "rust" else NUMPY
    specs = model_specs(vocabulary, args.positions, args.dropout)
    network = SequentialArrayNetwork((ss.CONTEXT,), specs, Adam(), "sequence", backend)
    network.rng = backend.default_rng(args.seed)
    network.randomize()
    rate = args.rate if args.rate is not None else rotary_study.RATES[(args.corpus, ARM)]
    shuffle = random.Random(args.seed)
    step = 0
    print(f"{args.corpus}: {ARM}, {args.positions} positions, dropout {args.dropout:g}, rate {rate:g}, {args.backend}")
    for epoch in range(args.epochs):
        steps, seconds = bss.train_epoch(network, train_windows, ss.BATCH_SIZE, rate, step, shuffle)
        step += steps
        bits = sequence_evaluate(network, held_out).cross_entropy / math.log(2)
        print(f"epoch {epoch + 1}: held-out {bits:.3f} bits per character ({seconds:.0f} s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    network.save(str(out))
    tokenizer.save(tokenizer_path(out))
    print(f"saved {out} and {tokenizer_path(out)}")


def sample(args: argparse.Namespace) -> None:
    network: Any = load_network(args.model)
    tokenizer = CharacterTokenizer.load(tokenizer_path(args.model))
    if tokenizer.output_classes != network.class_count:
        sys.exit(
            f"{tokenizer_path(args.model)} has {tokenizer.output_classes} output classes; the model's output has {network.class_count}"
        )
    if not args.prompt:
        sys.exit("a prompt has at least one character")
    sampler = Sampler(args.temperature, args.top_k, args.top_p)
    prefill, step = cached(network)
    ids = generate(prefill, step, tokenizer.encode(args.prompt), args.length, sampler, default_rng(args.seed))
    print(args.prompt + tokenizer.decode(ids))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="train the 2-layer model on a corpus and save it with its tokenizer")
    t.add_argument("--corpus", required=True, choices=list(text_data.CORPORA))
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", required=True, help="the model's format-2 file; its tokenizer file goes beside it")
    t.add_argument("--positions", choices=["rotary", "learned"], default="rotary")
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--rate", type=float, help="default: the sequence study's tuned rate for the corpus")
    t.add_argument("--backend", choices=["numpy", "rust"], default="numpy")
    t.add_argument("--limit", type=int, help="the first LIMIT training and held-out windows only: a smoke run")
    s = commands.add_parser("sample", help="sample text from a saved model")
    s.add_argument("--model", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--length", type=int, default=300)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int)
    s.add_argument("--top-p", type=float)
    s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train(args) if args.command == "train" else sample(args)


if __name__ == "__main__":
    main()
