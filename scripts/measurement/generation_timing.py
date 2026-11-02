"""
The generation benchmark (the generation workplan, D8): characters per second sampled from the
sequence study's 2-layer rotary model (64-token window, 64 features, 4 heads, a 65-character
vocabulary, untrained: decoding's cost doesn't depend on the weights), cached (the Decoder: one
token's pass per character, the window sliding past 64) against uncached (a full pass over the
last 64 tokens per character), on numpy and Rust, one thread.

    python scripts/measurement/generation_timing.py [--characters 1000] [--repeats 3] [--json out.json]
        [--heads 4] [--key-value-heads G] [--context 64] [--batch 1]

The model's options (the grouped-query attention workplan, D11): --heads query heads (d_k = 64 /
heads), sharing --key-value-heads key/value heads (every head its own when left out), over a
--context-token window; --batch prompts decoded together (prefill_batch, then step_batch, greedy
per row), the rate counting every row's characters. A batch decodes cached only: the uncached
path is one row's full pass. Each backend also reports its decode cache's bytes per token, 2 g d_k
8 bytes per layer (keys and values, float64), the cost grouped-query attention divides by h / g.

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
from dataclasses import replace
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indrajala_ml.generation.generate import cached, generate, uncached
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.studies import rotary_study
from indrajala_ml.studies import sequence_study as ss

VOCABULARY = 65
PROMPT = 16


def specs(heads: int, key_value_heads: int | None) -> list[LayerSpec]:
    """The 2-layer rotary model with its attentions at heads heads sharing key_value_heads (D11)."""

    def attention(spec: LayerSpec) -> LayerSpec:
        if isinstance(spec, Residual):
            return Residual(tuple(attention(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            grouped = None if key_value_heads == heads else key_value_heads
            return replace(spec, heads=heads, key_size=None, key_value_heads=grouped)
        return spec

    return [attention(spec) for spec in rotary_study.arm_specs("2-layer", VOCABULARY)]


def cache_bytes_per_token(model: list[LayerSpec], features: int) -> int:
    """The decode cache's bytes per token: each attention's keys and values, g d_k float64s each."""
    attentions = [layer for spec in model for layer in getattr(spec, "body", (spec,)) if isinstance(layer, Attention)]
    return sum(2 * a.kv_heads() * a.head_size(features) * 8 for a in attentions)


def _network(backend: Any, model: list[LayerSpec], context: int) -> Any:
    network = SequentialArrayNetwork((context,), model, Adam(), "sequence", backend)
    network.rng = backend.default_rng(0)
    network.randomize()
    return network


def rate(network: Any, mode: str, characters: int, batch: int = 1) -> float:
    """Characters per second, one sample of characters from a fixed prompt (each row's), greedy."""
    prompt = [i % VOCABULARY for i in range(PROMPT)]
    if batch > 1:
        assert mode == "cached", "a batch decodes cached only"
        decoder = network.decoder()
        start = time.perf_counter()
        rows = decoder.prefill_batch([prompt] * batch)
        for _ in range(characters - 1):
            rows = decoder.step_batch([max(range(len(p)), key=p.__getitem__) for p in rows])
        return batch * characters / (time.perf_counter() - start)
    prefill, step = cached(network) if mode == "cached" else uncached(network)
    start = time.perf_counter()
    generate(prefill, step, prompt, characters, Sampler(temperature=0.0), default_rng(0))
    return characters / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--characters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", help="also write the cells here")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-value-heads", type=int, default=None)
    parser.add_argument("--context", type=int, default=ss.CONTEXT)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    model = specs(args.heads, args.key_value_heads)
    cells: list[dict[str, Any]] = []
    for backend in (NUMPY, RUST):
        network = _network(backend, model, args.context)
        modes = ["cached"] if args.batch > 1 else ["cached", "uncached"]
        rates: dict[str, list[float]] = {mode: [] for mode in modes}
        for _ in range(args.repeats):
            for mode, values in rates.items():
                values.append(rate(network, mode, args.characters, args.batch))
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
        cell = {
            "case": f"{backend.name} cache",
            "metric": "cache bytes per token",
            "value": cache_bytes_per_token(model, network.layer_specs[0].size),
            "unit": "bytes",
        }
        print(json.dumps(cell), flush=True)
        cells.append(cell)
    print()
    print(f"{'case':<16} {'chars/s':>10} {'min-max':>18}")
    for cell in cells:
        if cell["unit"] == "chars/s":
            print(f"{cell['case']:<16} {cell['value']:>10.0f} {cell['min']:>8.0f}-{cell['max']:<8.0f}")
    by_case = {cell["case"]: cell["value"] for cell in cells}
    for name in (NUMPY.name, RUST.name):
        if f"{name} uncached" in by_case:
            print(f"{name}: cached {by_case[f'{name} cached'] / by_case[f'{name} uncached']:.1f}x uncached")
    print(f"cache: {by_case[f'{NUMPY.name} cache']} bytes per token")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(cells, f, indent=2)


if __name__ == "__main__":
    main()
