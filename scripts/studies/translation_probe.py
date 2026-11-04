"""
The encoder-decoder's first runs on the parallel corpora: a probe (indrajala_ml/studies/translation_probe.py).

    python scripts/studies/translation_probe.py time --corpus lives --steps 6
    python scripts/studies/translation_probe.py run --corpus lives --epochs 2
    python scripts/studies/translation_probe.py ablate --corpus lives --network data/studies/translation-probe/lives-seed0-blocks2.network.json

The model: d = 128, 4 heads of 32, 2 encoder and 2 decoder blocks (968,646 parameters), windows of
1,024 characters, batches of 2, rate 0.002 under the recipe, Rust, seed 0, the Lives (4,929
training windows, 2,465 steps an epoch). The source ablation (`other_source_bits`) scores each
validation window's English against another window's Greek; the gap, its bits less the window's
own, is how much the model reads its source.

Findings (2026-11-03 and 04; a probe, so they claim nothing):
- jebel, 2 epochs (0.72 s a step, 29 min an epoch, 1.4 GB): validation bits per character 3.33,
  then 2.84; greedy chrF 14.8 on 32 validation pairs; the gap 0.013 on the saved network.
- stone, 23 epochs overnight (0.69 s a step, 28-32 min an epoch): validation bits 3.33 to 1.81,
  still falling about 0.005 an epoch at the end; training 0.09 below validation, flat since
  epoch 5. The gap grew steadily, 0.001 (epoch 1), 0.008 (10), 0.016 (15), 0.024 (20), 0.027
  (23), about 0.001 an epoch with no plateau: the model learns to read the Greek, but the Greek
  carries about 1.5% of the bits. Greedy chrF 20.6, from more plausible English, not translation:
  the 32 outputs are nearly one looping Plutarch-like sentence ("And in the senate of the city was
  a strength of the city ..."), the same whatever the source.
- Reading: the decoder is an English language model whose cross-attention carries little. Under
  teacher forcing most characters follow from the English before them, and at batch 2 over 57k
  steps the alignment comes slowly. More epochs of this setup buy about 0.001 bits of gap an epoch.
- Abandoned there (the owner, 2026-11-04): no diagnosis and no further runs. Results are
  gitignored under data/studies/translation-probe/ (jebel-2epoch/, stone-23epoch/) on pyramidon.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from indrajala_ml.studies import translation_probe as probe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["time", "run", "ablate"])
    parser.add_argument("--network", help="ablate: a saved network's file")
    parser.add_argument("--corpus", default="lives", choices=["histories", "lives"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="rust", choices=["rust", "numpy"])
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sample", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "time":
        result = probe.time_steps(args.corpus, args.seed, args.steps, args.backend, args.blocks)
    elif args.command == "ablate":
        result = probe.source_ablation(Path(args.network), args.corpus)
    else:
        result = probe.run(args.corpus, args.seed, args.epochs, probe.OUT, args.backend, args.blocks, args.sample)
        result = {k: v for k, v in result.items() if k != "translations"}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
