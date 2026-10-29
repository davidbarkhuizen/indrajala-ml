"""
Per-op parity probes of the crate across machines (the aarch64 workplan, stage 1): each probe runs
one op on fixed inputs and keeps every output's bits, so two machines' records compare value by
value. The golden run trains on few inputs; these reach the four places the workplan's "Where the
bits stand" names, where the crate can still differ between machines:

    python scripts/measurement/parity_probe.py record litshe.npz
    python scripts/measurement/parity_probe.py compare jebel.npz litshe.npz

- exp: the crate's `exp` (Rust's f64::exp, the C library's) and Python's math.exp (the pure-Python
  networks', the same C library) over a large sample: uniform in [-20, 20], where the networks'
  logits live, and over the whole finite range [-745, 709.78].
- relu, softmax: signed zeros, NaN of both signs and infinities, where x86's maxsd and ARM's
  fmaxnm answer differently.
- adam: one Adam update at every t from 1 to 64 and at powers of ten to 10^6, where the bias
  corrections' powi is LLVM's order, not ours.

Inputs come from random.Random, whose stream and uniform() are the same on every machine, and are
recorded beside the outputs, so compare checks that both records ran the same inputs. Bits are
compared as uint64: a NaN's sign and payload count.
"""

import argparse
import math
import random
import struct
import sys
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from indrajala_math_rust import Array, array_relu, array_softmax, exp, layer_adam_apply_accumulated_gradient

SEED = "parity probe"
CHUNK = 65536
SPECIALS = [0.0, -0.0, math.nan, -math.nan, math.inf, -math.inf, 5e-324, -5e-324, 1.0, -1.0]


def _bits(values: Sequence[float]) -> npt.NDArray[np.uint64]:
    return np.frombuffer(struct.pack(f"<{len(values)}d", *values), dtype="<u8").copy()


def _uniform(name: str, count: int, low: float, high: float) -> list[float]:
    rng = random.Random(f"{SEED} {name}")
    return [rng.uniform(low, high) for _ in range(count)]


def _crate_exp(inputs: list[float]) -> list[float]:
    outputs: list[float] = []
    for start in range(0, len(inputs), CHUNK):
        outputs.extend(exp(Array(inputs[start : start + CHUNK])).tolist())
    return outputs


def _softmax_rows() -> list[list[float]]:
    n, z, inf = math.nan, -0.0, math.inf
    return [[z, 0.0, -1.0], [0.0, z, z], [z, z, z], [n, 1.0, 0.0], [1.0, -n, 0.0], [-inf, 0.0, z], [inf, 1.0, 0.0]]


def _adam(t: int) -> list[float]:
    rng = random.Random(f"{SEED} adam")
    w, grad_w, m_w, v_w = ([[rng.uniform(-1, 1) for _ in range(4)] for _ in range(3)] for _ in range(4))
    b, grad_b, m_b, v_b = ([rng.uniform(-1, 1) for _ in range(3)] for _ in range(4))
    v_w, v_b = [[abs(x) for x in row] for row in v_w], [abs(x) for x in v_b]
    results = layer_adam_apply_accumulated_gradient(
        Array(w), Array(b), Array(grad_w), Array(grad_b), Array(m_w), Array(v_w), Array(m_b), Array(v_b),
        t, 0.9, 0.999, 1e-8, 0.001, 1,
    )  # fmt: skip
    flat: list[float] = []
    for result in results:
        flat.extend(np.asarray(result.tolist(), dtype=np.float64).ravel().tolist())
    return flat


ADAM_STEPS = [*range(1, 65), 100, 1000, 10_000, 100_000, 1_000_000]


def probes() -> dict[str, tuple[list[float], Callable[[], list[float]]]]:
    """Each probe's inputs (as floats, for the record) and the call producing its outputs."""
    near = _uniform("exp near", 1 << 20, -20.0, 20.0)
    full = [*_uniform("exp full", 1 << 18, -745.0, 709.78), *SPECIALS]
    rows = _softmax_rows()
    return {
        "exp near": (near, lambda: _crate_exp(near)),
        "exp full": (full, lambda: _crate_exp(full)),
        "math.exp near": (near, lambda: [math.exp(x) for x in near]),
        "math.exp full": (full, lambda: [math.exp(x) for x in full]),
        "relu": (SPECIALS, lambda: array_relu(Array(SPECIALS)).tolist()),
        "softmax": ([x for row in rows for x in row], lambda: [x for row in rows for x in array_softmax(Array(row)).tolist()]),
        "adam": ([float(t) for t in ADAM_STEPS], lambda: [x for t in ADAM_STEPS for x in _adam(t)]),
    }  # fmt: skip


def record(path: str) -> None:
    arrays: dict[str, npt.NDArray[np.uint64]] = {}
    for name, (inputs, run) in probes().items():
        arrays[f"{name}/inputs"] = _bits(inputs)
        arrays[f"{name}/outputs"] = _bits(run())
        print(f"{name}: {len(inputs)} inputs, {arrays[f'{name}/outputs'].size} outputs")
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]


def _hex(bits: np.uint64) -> str:
    return float.hex(struct.unpack("<d", struct.pack("<Q", int(bits)))[0]) + f" ({int(bits):#018x})"


def compare(first_path: str, second_path: str, examples: int = 5) -> bool:
    first, second = np.load(first_path), np.load(second_path)
    names = sorted({key.rsplit("/", 1)[0] for key in [*first.files, *second.files]})
    identical = True
    for name in names:
        keys = (f"{name}/inputs", f"{name}/outputs")
        if not all(key in record.files for key in keys for record in (first, second)):
            print(f"{name}: only in one record")
            identical = False
            continue
        if not np.array_equal(first[keys[0]], second[keys[0]]):
            print(f"{name}: the inputs differ, so the outputs can't be compared")
            identical = False
            continue
        a, b = first[keys[1]], second[keys[1]]
        if a.shape != b.shape:
            print(f"{name}: {a.size} outputs against {b.size}")
            identical = False
            continue
        differing = np.flatnonzero(a != b)
        print(f"{name}: {a.size - differing.size} of {a.size} identical")
        inputs = first[keys[0]]
        for index in differing[:examples]:
            source = _hex(inputs[index]) if a.size == inputs.size else f"output {index}"
            print(f"  at {source}: {_hex(a[index])} against {_hex(b[index])}")
        identical = identical and differing.size == 0
    print("bit-identical" if identical else "NOT bit-identical")
    return identical


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "compare"])
    parser.add_argument("paths", nargs="+", help="record: the file to write; compare: two records")
    args = parser.parse_args(argv)
    if args.mode == "record":
        if len(args.paths) != 1:
            parser.error("record takes one file")
        record(args.paths[0])
    else:
        if len(args.paths) != 2:
            parser.error("compare takes two files")
        sys.exit(0 if compare(*args.paths) else 1)


if __name__ == "__main__":
    main()
