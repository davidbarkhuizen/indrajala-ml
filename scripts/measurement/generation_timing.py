"""
The generation benchmark (the generation workplan, D8): characters per second sampled from the
sequence study's 2-layer rotary model (64-token window, 64 features, 4 heads, a 65-character
vocabulary, untrained: decoding's cost doesn't depend on the weights), cached (the Decoder: one
token's pass per character, the window sliding past 64) against uncached (a full pass over the
last 64 tokens per character), on numpy and Rust, one thread.

    python scripts/measurement/generation_timing.py [--characters 1000] [--repeats 3] [--json out.json]

Each cell samples --characters characters from a 16-character prompt, greedy (the sampler's cost
is the same in both), --repeats times in turn, and reports the median rate (min-max). One tree, so
no A/B: cached against uncached is the measurement. Run on jebel (docs/measurement.md, §2), nothing
else running. It prints one JSON line per cell ({"case", "metric", "value", "unit"}, ab.py's probe
format), then a table.
"""

from __future__ import annotations

import os

# one thread for numpy's BLAS, set before numpy loads; the crate's products thread only past 8M flops,
# which no product here reaches
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import statistics
import sys
import time
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indrajala_ml.generation.generate import cached, generate, uncached
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.studies import rotary_study
from indrajala_ml.studies import sequence_study as ss

VOCABULARY = 65
PROMPT = 16


def _network(backend: Any) -> Any:
    network = SequentialArrayNetwork(
        (ss.CONTEXT,), rotary_study.arm_specs("2-layer", VOCABULARY), Adam(), "sequence", backend
    )
    network.rng = backend.default_rng(0)
    network.randomize()
    return network


def rate(network: Any, mode: str, characters: int) -> float:
    """Characters per second, one sample of characters from a fixed prompt, greedy."""
    prefill, step = cached(network) if mode == "cached" else uncached(network)
    prompt = [i % VOCABULARY for i in range(PROMPT)]
    start = time.perf_counter()
    generate(prefill, step, prompt, characters, Sampler(temperature=0.0), default_rng(0))
    return characters / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--characters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", help="also write the cells here")
    args = parser.parse_args()
    cells: list[dict[str, Any]] = []
    for backend in (NUMPY, RUST):
        network = _network(backend)
        rates: dict[str, list[float]] = {"cached": [], "uncached": []}
        for _ in range(args.repeats):
            for mode, values in rates.items():
                values.append(rate(network, mode, args.characters))
        for mode, values in rates.items():
            cell = {
                "case": f"{backend.name} {mode}",
                "metric": "characters per second",
                "value": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "unit": "chars/s",
            }
            print(json.dumps(cell), flush=True)
            cells.append(cell)
    print()
    print(f"{'case':<16} {'chars/s':>10} {'min-max':>18}")
    for cell in cells:
        print(f"{cell['case']:<16} {cell['value']:>10.0f} {cell['min']:>8.0f}-{cell['max']:<8.0f}")
    by_case = {cell["case"]: cell["value"] for cell in cells}
    for name in (NUMPY.name, RUST.name):
        print(f"{name}: cached {by_case[f'{name} cached'] / by_case[f'{name} uncached']:.1f}x uncached")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(cells, f, indent=2)


if __name__ == "__main__":
    main()
