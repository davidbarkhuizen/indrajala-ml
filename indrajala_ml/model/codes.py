"""
A window-recurrent network's codes set from its own states (the audit-remedies workplan, stage 3,
D3): each snapped factor's codes the k-means centres of its cells' outputs over the rows given,
k-means++ seeding (Arthur & Vassilvitskii 2007) then Lloyd's iterations, in place of the random
draw that ignores where the states lie (Łańcucki et al. 2020: a scale-mismatched start is one of
the codebook's collapses).

Ported from indrajala-studies' harness (modular_family.py's kmeans and the code-setting part of
codes_from_states, the state formation workplan's D6) bit for bit: the same draws from the same
generator in the same order, the states read batch by batch and window by window, so a study that
switches to it keeps its bits.

Re-seeding dead codes (stage 4, D4; Dhariwal et al. 2020's restarts, Zheng & Vedaldi 2023): a code
no state on a sample snaps to (at most min_use of them) is moved onto a sample state, drawn by
k-means++'s D^2 rule against the live codes and those already moved, so it lands where the codebook
serves the states worst; its optimizer state is zeroed with it. An option a caller runs at its own
cadence; whether it helps is a registered study's (D11).

Every implementation: a network gives its cells' outputs (cell_outputs) and its codes
(factor_codes), and takes its codes (set_codes, reseed_codes); the arithmetic here is numpy's, once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.specs.window_recurrence import WindowRecurrence

FloatArray = npt.NDArray[np.float64]
Rows = Sequence[Sequence[float]]


class CodedNetwork(Protocol):
    """A window-recurrent network, on any implementation, as the codes' functions read and write it."""

    recurrence: WindowRecurrence

    def cell_outputs(self, rows: Rows, first: bool = False) -> list[list[list[float]]]:
        """Each window's cells' outputs before the snap, W of them (first: the initial state's first), (N, state_size) each."""
        ...

    def factor_codes(self, factor: int) -> list[list[float]]:
        """Factor factor's codes, (count, state_size)."""
        ...

    def set_codes(self, factor: int, codes: list[list[float]]) -> None:
        """Factor factor's codes, (count, state_size)."""
        ...

    def reseed_codes(self, factor: int, rows: dict[int, list[float]]) -> None:
        """Factor factor's codes rows set to the vectors given, and their optimizer state zeroed."""
        ...


@dataclass(frozen=True)
class CodesFit:
    """
    A factor's codes as k-means set them: the codes some state is nearest to (in_use), and the
    pooled within-cluster covariance's top eigenvalue (within_lambda_max), the clusters' spread;
    twice it is their critical temperature (Rose 1998), which the harness logs.
    """

    in_use: int
    within_lambda_max: float


def kmeans(
    points: FloatArray, k: int, rng: np.random.Generator, iterations: int
) -> tuple[FloatArray, npt.NDArray[np.intp]]:
    """
    k centres of the points (n, d), k-means++ seeding then Lloyd's iterations, and each point's nearest
    centre. A centre with no points (fewer distinct points than k) is a drawn point moved by a small
    random step, so no two codes start equal (deterministic annealing's perturbation).
    """
    n = len(points)
    centres: list[FloatArray] = []
    for _ in range(k):
        centres.append(points[d2_draw(points, centres, rng)])
    C = np.array(centres, dtype=float)
    scale = max(float(np.sqrt(np.var(points, axis=0).sum())), 1e-6)
    for _ in range(iterations + 1):
        nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        for j in range(k):
            members = points[nearest == j]
            C[j] = (
                members.mean(axis=0)
                if len(members)
                else points[int(rng.integers(n))] + rng.normal(0.0, 1e-3 * scale, size=points.shape[1])
            )
    nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    return C, nearest


def d2_draw(points: FloatArray, anchors: Sequence[FloatArray], rng: np.random.Generator) -> int:
    """
    A point's index drawn by k-means++'s D^2 rule (Arthur & Vassilvitskii 2007): with probability
    proportional to its squared distance to its nearest anchor; uniformly with no anchors, or when
    every point is on one.
    """
    n = len(points)
    if not anchors:
        return int(rng.integers(n))
    d = np.min(np.sum((points[:, np.newaxis, :] - np.array(anchors)[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    total = float(d.sum())
    return int(rng.choice(n, p=d / total)) if total > 0 else int(rng.integers(n))


def snap_indices(states: FloatArray, codes: FloatArray) -> npt.NDArray[np.intp]:
    """Each state's code as the snap picks it: by squared distance, a left fold over the features, the lowest index on a tie."""
    differences = states[:, np.newaxis, :] - codes[np.newaxis, :, :]
    distances = np.take(np.cumsum(differences * differences, axis=2), -1, axis=2)
    return np.argmin(distances, axis=1)


def factor_states(network: CodedNetwork, batches: Sequence[Rows], first: bool = False) -> list[FloatArray]:
    """
    Each factor's cell outputs h (windows by its state size) over every window of the batches given,
    batch by batch and, within a batch, window by window (the harness's order); first: each batch's
    initial states (one per example) before its windows'.
    """
    factors = network.recurrence.factors
    starts = [sum(f.state_size for f in factors[:g]) for g in range(len(factors))]
    blocks: list[list[FloatArray]] = [[] for _ in factors]
    for batch in batches:
        for cell in network.cell_outputs(batch, first=first):
            H = np.array(cell, dtype=float)
            for g, f in enumerate(factors):
                blocks[g].append(H[:, starts[g] : starts[g] + f.state_size])
    return [np.concatenate(b) for b in blocks]


def fit_codes(
    network: CodedNetwork, states: Sequence[FloatArray], rng: np.random.Generator, iterations: int
) -> list[CodesFit | None]:
    """
    Each snapped factor's codes set by kmeans of its states (factor_states'), factor by factor, from
    rng; a continuous factor's None, untouched and drawing nothing.
    """
    fits: list[CodesFit | None] = []
    for g, factor in enumerate(network.recurrence.factors):
        if factor.codes is None:
            fits.append(None)
            continue
        h = states[g]
        E, nearest = kmeans(h, factor.codes.count, rng, iterations)
        within = np.zeros((h.shape[1], h.shape[1]))
        for code in np.unique(nearest):
            centred = h[nearest == code] - h[nearest == code].mean(axis=0)
            within += centred.T @ centred
        lam = float(np.linalg.eigvalsh(within / max(len(h) - 1, 1))[-1])
        network.set_codes(g, E.tolist())
        fits.append(CodesFit(in_use=len(np.unique(nearest)), within_lambda_max=lam))
    return fits


def initialise_codes(
    network: CodedNetwork, batches: Sequence[Rows], rng: np.random.Generator, iterations: int
) -> list[CodesFit | None]:
    """Each snapped factor's codes from its states over the batches given: fit_codes of factor_states."""
    return fit_codes(network, factor_states(network, batches), rng, iterations)


def reseed_dead_codes(
    network: CodedNetwork, batches: Sequence[Rows], rng: np.random.Generator, min_use: int = 0
) -> list[list[int]]:
    """
    Each snapped factor's dead codes, those at most min_use of the sample's states snap to (its
    initial states and every window's, over the batches given), each moved in index order onto a
    state drawn by d2_draw against the live codes and the codes already moved, with its optimizer
    state zeroed (reseed_codes). Returns each factor's moved codes (none for a continuous factor).
    A live code, and every other parameter, is untouched.
    """
    states = factor_states(network, batches, first=True)
    moved: list[list[int]] = []
    for g, factor in enumerate(network.recurrence.factors):
        if factor.codes is None:
            moved.append([])
            continue
        h = states[g]
        E = np.array(network.factor_codes(g), dtype=float)
        uses = np.bincount(snap_indices(h, E), minlength=len(E))
        dead = [i for i in range(len(E)) if uses[i] <= min_use]
        anchors = [E[i] for i in range(len(E)) if uses[i] > min_use]
        rows: dict[int, list[float]] = {}
        for i in dead:
            point = h[d2_draw(h, anchors, rng)]
            rows[i] = point.tolist()
            anchors.append(point)
        if rows:
            network.reseed_codes(g, rows)
        moved.append(dead)
    return moved
