"""
The RNG audit's measurements (docs/rng-audit.md): the crate's uniform/bernoulli_mask against
numpy's legacy np.random (MT19937) and its PCG64 Generator, for statistical quality and speed.

    python scripts/rng_audit.py quality [--repeats 20]
    python scripts/rng_audit.py time [--repeats 5]

`quality` draws 10M uniforms from each generator and reports a 4096-bin chi-square (as a
Wilson-Hilferty z), a Kolmogorov-Smirnov p, the lag-1 correlation (as z = r * sqrt(n)), and the
correlation between the first draws of consecutive small calls (the position carries across calls),
then bernoulli_mask's keep rate (as z) at three drop probabilities. --repeats reruns the crate's KS and
lag-1 on 2M draws, since one borderline p is expected by chance somewhere in a table this size.

`time` runs every (backend, repeat) in its own process, rotating the order, and reports the median
per draw for weight-init and dropout-mask shapes.
"""

import argparse
import json
import math
import statistics
import sys
import timeit
from collections.abc import Callable
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import numpy.typing as npt
from process_runs import interleaved_runs, run_json_worker

FloatArray = npt.NDArray[np.float64]
BACKENDS = ["rust", "numpy-legacy", "numpy-pcg64"]
# (op, shape): weight init at small/MNIST/large sizes, then dropout masks at batch 1, 32, 512
CASES: list[tuple[str, tuple[int, int]]] = [
    ("uniform", (128, 64)),
    ("uniform", (784, 128)),
    ("uniform", (1000, 1000)),
    ("mask", (1, 128)),
    ("mask", (32, 128)),
    ("mask", (512, 128)),
]


def rust_uniform(n: int) -> FloatArray:
    return np.array(pa.uniform(0.0, 1.0, n).tolist())


def chi_square_z(counts: FloatArray) -> float:
    dof = len(counts) - 1
    expected = float(counts.mean())
    statistic = float(((counts - expected) ** 2 / expected).sum())
    return ((statistic / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))


def ks_p(values: FloatArray) -> float:
    ordered = np.sort(values)
    n = len(ordered)
    i = np.arange(1, n + 1)
    d = max(float((i / n - ordered).max()), float((ordered - (i - 1) / n).max()))
    t = d * math.sqrt(n)
    return min(1.0, 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * t * t) for j in range(1, 100)))


def lag1_z(values: FloatArray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1]) * math.sqrt(len(values))


def quality(repeats: int) -> None:
    draw: dict[str, Callable[[int], FloatArray]] = {
        "rust": rust_uniform,
        "numpy-legacy": lambda n: np.random.uniform(0.0, 1.0, n),
        "numpy-pcg64": np.random.default_rng().random,
    }
    for backend, fn in draw.items():
        values = fn(10_000_000)
        counts = np.histogram(values, bins=4096, range=(0.0, 1.0))[0].astype(np.float64)
        firsts = np.array([fn(8)[0] for _ in range(50_000)])
        print(
            f"{backend:13s} chi2 z {chi_square_z(counts):+.2f}  KS p {ks_p(values):.3f}  "
            f"lag-1 z {lag1_z(values):+.2f}  call-to-call z {lag1_z(firsts):+.2f}"
        )
    for drop_probability in (0.1, 0.5, 0.9):
        mask = np.array(pa.bernoulli_mask(drop_probability, 2_000_000).tolist())
        keep = 1.0 - drop_probability
        z = (float(mask.mean()) - keep) / math.sqrt(keep * drop_probability / mask.size)
        print(f"bernoulli_mask p={drop_probability}: keep rate {mask.mean():.5f}, z {z:+.2f}")
    if repeats > 0:
        runs = [rust_uniform(2_000_000) for _ in range(repeats)]
        print(f"rust KS p over {repeats} runs (uniform if sound): " + " ".join(f"{ks_p(v):.2f}" for v in runs))
        zs = [lag1_z(v) for v in runs]
        print(f"rust lag-1 z over {repeats} runs: mean {statistics.fmean(zs):+.2f}, sd {statistics.pstdev(zs):.2f}")


def time_worker(backend: str) -> dict[str, Any]:
    uniform: Callable[[tuple[int, int]], object]
    mask: Callable[[tuple[int, int]], object]
    if backend == "rust":
        uniform = lambda shape: pa.uniform(-0.1, 0.1, shape)
        mask = lambda shape: pa.bernoulli_mask(0.3, shape)
    elif backend == "numpy-legacy":
        uniform = lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        mask = lambda shape: (np.random.random(shape) >= 0.3).astype(np.float64)
    else:
        generator = np.random.default_rng(0)
        uniform = lambda shape: generator.uniform(-0.1, 0.1, size=shape)
        mask = lambda shape: (generator.random(shape) >= 0.3).astype(np.float64)
    ops = {"uniform": uniform, "mask": mask}
    result: dict[str, float] = {}
    for op, shape in CASES:
        timer = timeit.Timer(lambda: ops[op](shape))  # noqa: B023  (called within this iteration)
        number = timer.autorange()[0]
        best = min(timer.repeat(5, number)) / number
        result[f"{op} {shape}"] = best / (shape[0] * shape[1]) * 1e9
    return result


def time_all(repeats: int) -> None:
    runs = interleaved_runs(
        BACKENDS, repeats, lambda backend: run_json_worker([sys.executable, __file__, "time-worker", backend])
    )
    print(f"ns per draw, median of {repeats} processes per backend")
    print("| case | " + " | ".join(BACKENDS) + " |")
    print("|---|" + "---:|" * len(BACKENDS))
    for op, shape in CASES:
        key = f"{op} {shape}"
        cells = [f"{statistics.median(run[key] for run in runs[backend]):.2f}" for backend in BACKENDS]
        print(f"| {key} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("quality").add_argument("--repeats", type=int, default=20)
    commands.add_parser("time").add_argument("--repeats", type=int, default=5)
    commands.add_parser("time-worker").add_argument("backend", choices=BACKENDS)
    args = parser.parse_args()
    if args.command == "quality":
        quality(args.repeats)
    elif args.command == "time":
        time_all(args.repeats)
    else:
        print(json.dumps(time_worker(args.backend)))


if __name__ == "__main__":
    main()
