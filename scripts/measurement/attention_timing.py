"""
The attention benchmark (the context-length workplan, D7): one attention layer's forward and backward
passes, time and peak memory, at T in {64, 256, 1024} characters, on numpy and Rust, one thread.

    python scripts/measurement/attention_timing.py [--tokens 64 256 1024] [--backends numpy rust]
        [--masks causal packed] [--repeats 5] [--memory | --no-memory] [--json out.json]
        [--features 64] [--heads 4]

The layer is the study model's (D6): d = 64 features, 4 heads, causal, rotary, attention dropout
--dropout (0.1) in training mode, so the forward draws its mask. Each T runs at the token budget
(D5): B * T = 2,048, a batch of 32 at 64, 8 at 256, 2 at 1,024. Two masks: causal alone, and
packed, the causal mask with step 11's segments (paragraphs of PARAGRAPHS' lengths in turn, each
ended by the separator, cut into the batch's rows), the case tiled attention skips across. Weights
and inputs are seeded uniforms: the cost doesn't depend on them.

--features and --heads set a larger layer than the study model's (the crate performance workplan's
D13: `--features 512 --heads 8 --tokens 256`, where the weights are past every L2 here); its rows are
named `d=512 ...`, so the study model's keep their names.

Time: per case, a warm-up pass, then --repeats passes, each the forward (forward_batch) and the
backward (compute_hidden_delta_batch, then accumulate_gradient_batch), timed apart; the median seconds per call of
each. Memory (on by default): each case again in a fresh process, one forward and backward, its
peak resident set (VmHWM, reset just before) less its resident set before the layer is built, in
MB, page-granular and including the allocator's slack: one measure for numpy and the crate alike,
transient arrays included (D7). For numpy, tracemalloc's peak over the same passes too, as a
cross-check (it can't see the crate's allocations).

It prints one JSON line per value ({"case", "metric", "value", "unit"}, ab.py's probe format;
ab.py's attention_timing adapter reads --json), then a table. Run on jebel (docs/measurement.md,
§2), nothing else running.
"""

from __future__ import annotations

import os

# one thread for numpy's BLAS, set before numpy loads; the crate's products thread only past 8M
# flops, which the study's shapes reach only at the largest T: measured as they run in training
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from process_runs import set_rust_threads

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext

TOKENS = (64, 256, 1024)
BUDGET = 2048  # B * T, the token budget (D5)
FEATURES = 64
HEADS = 4
DROPOUT = 0.1
MASKS = ("causal", "packed")
BACKENDS = ("numpy", "rust")
# paragraph lengths in characters, the separator included, used in turn to cut the packed rows: a
# spread around the long-form corpora's medians (729 to 1,266), and shorter ones, as a corpus has
PARAGRAPHS = (729, 140, 1130, 410, 1266, 60, 905)
SYMBOLS = 64  # the ids' symbols; the separator is SYMBOLS, the pad SYMBOLS + 1 (never used here)


def batch_size(tokens: int) -> int:
    assert BUDGET % tokens == 0, f"T divides the token budget {BUDGET}; got {tokens}"
    return BUDGET // tokens


def packed_ids(examples: int, tokens: int) -> np.ndarray:
    """(examples, tokens) ids: paragraphs of PARAGRAPHS' lengths in turn, each ending in the separator."""
    ids: list[int] = []
    k = 0
    while len(ids) < examples * tokens:
        length = PARAGRAPHS[k % len(PARAGRAPHS)]
        ids += [i % SYMBOLS for i in range(length - 1)] + [SYMBOLS]
        k += 1
    return np.array(ids[: examples * tokens], dtype=np.float64).reshape(examples, tokens)


def _layer_class(backend: str) -> Any:
    if backend == "numpy":
        from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer

        return AttentionArrayLayer
    from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer

    return AttentionRustArrayLayer


class Case:
    """One case's layer, input, upstream delta and context, built and seeded."""

    def __init__(
        self, backend: str, tokens: int, mask: str, dropout: float, features: int = FEATURES, heads: int = HEADS
    ) -> None:
        self.backend, self.tokens, self.mask = backend, tokens, mask
        examples = batch_size(tokens)
        rng = np.random.default_rng(0)
        layer = _layer_class(backend)(tokens, features, heads, causal=True, dropout=dropout, positions="rotary")
        shapes = [np.asarray(p.tolist() if backend == "rust" else p).shape for p in layer.parameters()]
        weights = [rng.uniform(-0.5, 0.5, shape) for shape in shapes]
        inputs = rng.uniform(-1.0, 1.0, (examples, tokens * features))
        upstream = rng.uniform(-1.0, 1.0, (examples, tokens * features))
        array_backend: Any = RUST if backend == "rust" else NUMPY

        def as_backend(values: np.ndarray) -> Any:
            if backend == "numpy":
                return values
            from indrajala_math_rust import Array

            return Array(values.tolist())

        layer.set_parameters([as_backend(w) for w in weights])
        self.inputs, self.delta = as_backend(inputs), as_backend(upstream)
        self.context: ForwardContext = NO_CONTEXT
        if mask == "packed":
            self.context = array_backend.segment_context(as_backend(packed_ids(examples, tokens)), SYMBOLS, SYMBOLS + 1)
        layer.set_rng(array_backend.default_rng(1))
        layer.set_training_mode(True)
        self.layer = layer

    def forward(self) -> None:
        self.layer.forward_batch(self.inputs, context=self.context)

    def backward(self) -> None:
        self.layer.compute_hidden_delta_batch(self, context=self.context)
        self.layer.accumulate_gradient_batch(self.inputs, context=self.context)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> Any:
        # the layer after the attention, as the network's backward pass asks it: the upstream delta
        return self.delta


def case_name(backend: str, tokens: int, mask: str, features: int = FEATURES) -> str:
    size = "" if features == FEATURES else f"d={features} "
    return f"{size}T={tokens} B={batch_size(tokens)} {mask} / {backend}"


def time_case(case: Case, repeats: int) -> dict[str, float]:
    """The median seconds per forward and per backward call, after a warm-up pass."""
    case.forward()
    case.backward()
    forwards: list[float] = []
    backwards: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        case.forward()
        middle = time.perf_counter()
        case.backward()
        backwards.append(time.perf_counter() - middle)
        forwards.append(middle - start)
    return {"forward": statistics.median(forwards), "backward": statistics.median(backwards)}


def _status_mb(field: str) -> float:
    # a /proc/self/status field, reported in kB
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) / 1024.0
    raise ValueError(f"/proc/self/status has no {field}")


def memory_case(backend: str, tokens: int, mask: str, dropout: float, features: int, heads: int) -> dict[str, float]:
    """
    In this (fresh) process: one forward and backward's peak resident set over the baseline, in MB.
    The peak is VmHWM, reset to the current resident set first (clear_refs' 5): getrusage's
    ru_maxrss would carry the parent's peak across the fork and exec.
    """
    with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
        f.write("5")
    before = _status_mb("VmRSS")
    if backend == "numpy":
        tracemalloc.start()
    case = Case(backend, tokens, mask, dropout, features, heads)
    case.forward()
    case.backward()
    result = {"peak RSS": _status_mb("VmHWM") - before}
    if backend == "numpy":
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["tracemalloc peak"] = peak / 2**20
    return result


def memory_in_subprocess(
    backend: str, tokens: int, mask: str, dropout: float, features: int, heads: int
) -> dict[str, float]:
    command = [sys.executable, os.path.abspath(__file__), "--memory-case", backend, str(tokens), mask]
    command += ["--dropout", str(dropout), "--features", str(features), "--heads", str(heads)]
    out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--features", type=int, default=FEATURES)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", help="write the results here as {case: {metric: value}}")
    parser.add_argument("--rust-threads", help="N:T, set_matmul_threading(N, T) before timing Rust")
    parser.add_argument("--memory-case", nargs=3, metavar=("BACKEND", "T", "MASK"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_case:
        backend, tokens, mask = args.memory_case
        print(json.dumps(memory_case(backend, int(tokens), mask, args.dropout, args.features, args.heads)))
        return
    set_rust_threads(args.rust_threads)
    results: dict[str, dict[str, float]] = {}
    units = {"forward": "s", "backward": "s", "peak RSS": "MB", "tracemalloc peak": "MB"}
    for backend in args.backends:
        for tokens in args.tokens:
            for mask in args.masks:
                name = case_name(backend, tokens, mask, args.features)
                values = time_case(Case(backend, tokens, mask, args.dropout, args.features, args.heads), args.repeats)
                if args.memory:
                    values |= memory_in_subprocess(backend, tokens, mask, args.dropout, args.features, args.heads)
                results[name] = values
                for metric, value in values.items():
                    print(
                        json.dumps({"case": name, "metric": metric, "value": value, "unit": units[metric]}), flush=True
                    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    print()
    print(f"{'case':<34} {'forward s':>11} {'backward s':>11} {'peak RSS MB':>12} {'tracemalloc MB':>15}")
    for name, values in results.items():
        cells = [
            f"{values['forward']:>11.5f}",
            f"{values['backward']:>11.5f}",
            f"{values.get('peak RSS', float('nan')):>12.1f}",
            f"{values.get('tracemalloc peak', float('nan')):>15.1f}",
        ]
        print(f"{name:<34} {' '.join(cells)}")


if __name__ == "__main__":
    main()
