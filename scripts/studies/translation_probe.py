"""
The encoder-decoder's first runs on the parallel corpora: a probe (indrajala_ml/studies/translation_probe.py).

    python scripts/studies/translation_probe.py time --corpus lives --steps 6
    python scripts/studies/translation_probe.py run --corpus lives --epochs 2

Findings:
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from indrajala_ml.studies import translation_probe as probe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["time", "run"])
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
    else:
        result = probe.run(args.corpus, args.seed, args.epochs, probe.OUT, args.backend, args.blocks, args.sample)
        result = {k: v for k, v in result.items() if k != "translations"}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
