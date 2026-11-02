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

Every implementation: a network gives its cells' outputs (cell_outputs) and takes its codes
(set_codes); the arithmetic here is numpy's, once.
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

    def cell_outputs(self, rows: Rows) -> list[list[list[float]]]:
        """Each window's cells' outputs before the snap, W of them, (N, state_size) each."""
        ...

    def set_codes(self, factor: int, codes: list[list[float]]) -> None:
        """Factor factor's codes, (count, state_size)."""
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
    centres = [points[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min(np.sum((points[:, np.newaxis, :] - np.array(centres)[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        total = float(d.sum())
        centres.append(points[int(rng.choice(n, p=d / total))] if total > 0 else points[int(rng.integers(n))])
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


def factor_states(network: CodedNetwork, batches: Sequence[Rows]) -> list[FloatArray]:
    """
    Each factor's cell outputs h (windows by its state size) over every window of the batches given,
    batch by batch and, within a batch, window by window (the harness's order).
    """
    factors = network.recurrence.factors
    starts = [sum(f.state_size for f in factors[:g]) for g in range(len(factors))]
    blocks: list[list[FloatArray]] = [[] for _ in factors]
    for batch in batches:
        for cell in network.cell_outputs(batch):
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
