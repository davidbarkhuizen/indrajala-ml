"""
Reading a snapped network's machine against its target, beyond the certificate's verdict (the
code-consolidation workplan's stage 2, the D9 probes): `by_length`, how exact the network is on
every word of each length; `breaks`, which of its transitions break, from codes that are right.

The network's machine is walked as the certificate walks it, one column a window (k 1), but on its
snapped states only (a near tie is taken on its nearest code), together with the target's: the
pairs (code, state) the start reaches. A code is **right** for a state when the network's outputs
from it agree with the target's from the state on every word of up to `d` tokens, `d` the target's
distinguishing length plus one (the workplan's D2): with right outputs, enough to tell every pair
of the target's states apart, with a token's margin. A state's **first code** is the one its
breadth-first shortest word reaches; its other codes are **extra**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_tasks as wt

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Product:
    """
    The pairs (network code, target state) the start reaches, breadth first, pair 0 the start's:
    each pair's next pair and whether its output is right, per token; its depth (the shortest
    word's length); and the codes in order of reach.
    """

    codes: list[wc.State]  # the network's reachable codes, breadth first
    pairs: list[tuple[int, int]]  # (index into codes, target state)
    next: IntArray  # (P, T) the pair after each token
    right_output: BoolArray  # (P, T)
    depth: IntArray  # (P,)

    @property
    def states(self) -> IntArray:
        return np.array([s for _, s in self.pairs], dtype=np.int64)


def product(machine: wc.WindowMachine, target: wt.Mealy) -> Product:
    """The network's machine walked with the target's from both starts (k 1, snapped states only)."""
    assert machine.columns == 1, "one column a window"
    windows = wc.all_windows(1, target.tokens)
    start = tuple(int(x) for x in machine.start()[0])
    code_index: dict[wc.State, int] = {start: 0}
    codes = [start]
    steps: dict[int, tuple[list[int], list[int]]] = {}  # code -> (next codes, outputs) per token
    pair_index: dict[tuple[int, int], int] = {(0, target.start): 0}
    pairs: list[tuple[int, int]] = [(0, target.start)]
    depth: list[int] = [0]
    nexts: list[list[int]] = []
    right: list[list[bool]] = []
    for p, (c, s) in enumerate(pairs):  # breadth first: pairs grows as it's walked
        if c not in steps:
            st = machine.step(codes[c], windows)
            following: list[int] = []
            for row in st.states:
                code = tuple(int(x) for x in row)
                if code not in code_index:
                    code_index[code] = len(codes)
                    codes.append(code)
                following.append(code_index[code])
            steps[c] = (following, [int(x) for x in st.digits[:, 0]])
        following, outputs = steps[c]
        row_next: list[int] = []
        row_right: list[bool] = []
        for t in range(target.tokens):
            pair = (following[t], int(target.transition[s, t]))
            if pair not in pair_index:
                pair_index[pair] = len(pairs)
                pairs.append(pair)
                depth.append(depth[p] + 1)
            row_next.append(pair_index[pair])
            row_right.append(outputs[t] == int(target.output[s, t]))
        nexts.append(row_next)
        right.append(row_right)
    return Product(
        codes,
        pairs,
        np.array(nexts, dtype=np.int64),
        np.array(right, dtype=np.bool_),
        np.array(depth, dtype=np.int64),
    )


def word_length(target: wt.Mealy) -> int:
    """`d`: the target's distinguishing length plus one (the workplan's D2)."""
    length = target.distinguishing_length()
    assert length is not None, "a minimal target"
    return length + 1


def right_codes(walk: Product, d: int) -> BoolArray:
    """Each pair's code right for its state: every output agrees on every word of up to d tokens."""
    right = np.ones(len(walk.pairs), dtype=np.bool_)
    for _ in range(d):
        right = (walk.right_output & right[walk.next]).all(axis=1)
    return right


def by_length(walk: Product, lengths: list[int], d: int) -> list[dict[str, Any]]:
    """
    Over all words of each length (uniform tokens, the start's pair): the share whose last answer is
    right, the share right at every answer (exact), and the share ending on a code right for its
    state; each from the mass propagated over the pairs, so exact, not sampled.
    """
    right = right_codes(walk, d)
    tokens = walk.next.shape[1]
    mass = np.zeros(len(walk.pairs))
    mass[0] = 1.0
    exact = mass.copy()  # the mass of words right at every answer so far
    rows: list[dict[str, Any]] = []
    for n in range(1, max(lengths) + 1):
        last = float((mass[:, np.newaxis] * walk.right_output).sum() / tokens)
        after, after_exact = np.zeros_like(mass), np.zeros_like(exact)
        for t in range(tokens):
            np.add.at(after, walk.next[:, t], mass / tokens)
            np.add.at(after_exact, walk.next[:, t], exact * walk.right_output[:, t] / tokens)
        mass, exact = after, after_exact
        if n in lengths:
            rows.append(
                {
                    "n": n,
                    "last": round(last, 4),
                    "exact": round(float(exact.sum()), 4),
                    "named": round(float(mass[right].sum()), 4),
                }
            )
    return rows


def breaks(walk: Product, d: int) -> dict[str, Any]:
    """
    Each transition that lands on a code wrong for the next state, split by the code it leaves (each
    state's first code, or an extra one), over every pair and over the pairs whose code is right (a
    break from a right code is the fault itself, not one inherited), and by the pair's depth.
    """
    right = right_codes(walk, d)
    first: dict[int, int] = {}  # state -> its first code
    for c, s in walk.pairs:  # breadth first, so the first seen is the shortest word's
        first.setdefault(s, c)
    tokens = walk.next.shape[1]
    tallies: dict[str, list[int]] = {k: [0, 0] for k in ("first", "extra", "first_right", "extra_right")}
    depths: dict[int, list[int]] = {}
    for p, (c, s) in enumerate(walk.pairs):
        broken = int((~right[walk.next[p]]).sum())
        kind = "first" if first[s] == c else "extra"
        for key in (kind, f"{kind}_right") if right[p] else (kind,):
            tallies[key][0] += broken
            tallies[key][1] += tokens
        if right[p]:
            at = depths.setdefault(int(walk.depth[p]), [0, 0])
            at[0] += broken
            at[1] += tokens
    states = set(first)
    return {
        "d": d,
        "codes": len(walk.codes),
        "pairs": len(walk.pairs),
        "states": len(states),
        "first_codes": len({first[s] for s in states}),
        "right_pairs": int(right.sum()),
        **{k: {"broken": v[0], "transitions": v[1]} for k, v in tallies.items()},
        "depth_right": {str(k): {"broken": v[0], "transitions": v[1]} for k, v in sorted(depths.items())},
    }
