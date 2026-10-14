"""
A window-recurrent network's recurrence as backend-free data (the window-recurrent adder workplan,
The network): what it adds to its window, a sequence model's layer specs (layer_specs.py) over one
window of k columns, k the window's input length.

The state is G factors side by side (the factored-state workplan, D1, D2), factor g's state c_g of
its own state_size, and the state c = [c_1, ..., c_G] of their total width. Per window: the window's
tokens, its StateInputs adding P c before their blocks, give the window's k digits; and each factor
g has its own pooling over the tokens the output layer reads, giving its summary u_g, its own cell,
h_g = cell_g(c_g, u_g), and its own codes, which snap h_g to the nearest of their count learned
codes, straight-through, the next window's c_g (W1, W4). With the cross-term on (cross=True), each
factor's cell reads the other factors' states too, cell_g(c_g, [u_g, c_(-g)]). Off, and with the
poolings reading the tokens before the first StateInput (AttentionPooling(BEFORE_STATE), D10), a
factor's next state is a function of its own state and the window's columns only; reading the
output layer's tokens (the default), it sees the other factors through the StateInputs. Each
factor's first state is a
learned vector, snapped alike (W8). A factor without codes (codes=None) is continuous: h_g itself is
its next state, and it has no codes' losses (the window-recurrent claims workplan's ablation, E2,
D3). A single factor is the network of one state.

    WindowRecurrence((Factor(4, Codes(8)),))
    WindowRecurrence((Factor(4, Codes(8)), Factor(4, Codes(8))), cross=False)
    WindowRecurrence((Factor(4, None),))

validate_window_recurrent (spec_validation.py) checks a window and its recurrence together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from indrajala_ml.model.specs.layer_specs import LayerSpec, StateInput


@dataclass(frozen=True)
class GRU:
    """
    The cell (W1b-iii; Cho et al. 2014): reset and update gates r and z, the candidate h~ = tanh(W u
    + U (r * c)), and h = z * c + (1 - z) * h~, reading the summary u and the state c. update_bias is
    the update gate's bias at initialisation, toward keeping the state (W13).
    """

    update_bias: float = 1.0


# where a factor's pooling reads the window's tokens (the factored-state workplan, D10)
OUTPUT = "output"
BEFORE_STATE = "before_state"


@dataclass(frozen=True)
class AttentionPooling:
    """
    The window's summary (W1b-i): one learned query, one head, over the window's tokens, u = the
    attention-weighted sum of their values. reads says which tokens: OUTPUT, those the window's
    output layer reads, which every StateInput has added the whole state to; or BEFORE_STATE, those
    its first StateInput reads, before any state is added, so that with the cross-term off a
    factor's next state is a function of its own state and the window's columns only (D10).
    """

    reads: str = OUTPUT


@dataclass(frozen=True)
class Codes:
    """
    The state's discretiser (W4, W14; van den Oord et al. 2017): count learned codes of state_size
    features, the cell's output snapped to the nearest, its gradient passed straight through. The
    codes learn by ||sg[h] - e||^2 and the cell is held to them by commitment * ||h - sg[e]||^2,
    both averaged over the windows and added to the digits' loss.
    """

    count: int
    commitment: float = 0.25


@dataclass(frozen=True)
class Factor:
    """One factor of the state: its width and its codes (None: a continuous factor)."""

    state_size: int
    codes: Codes | None


@dataclass(frozen=True)
class WindowRecurrence:
    """
    What a window-recurrent network adds to its window's specs: the state's factors, in order,
    whether each factor's cell reads the others' states (cross), and the cell and pooling every
    factor has.
    """

    factors: tuple[Factor, ...]
    cross: bool = False
    cell: GRU = GRU()
    pooling: AttentionPooling = AttentionPooling()

    @property
    def state_size(self) -> int:
        """The whole state's width, the factors' together."""
        return sum(factor.state_size for factor in self.factors)

    @property
    def snapped(self) -> bool:
        """Whether every factor has codes, so the network's states are finite (a machine's)."""
        return all(factor.codes is not None for factor in self.factors)


def pooling_reads(specs: Sequence[LayerSpec], recurrence: WindowRecurrence) -> int:
    """
    The index among a window's specs of the layer whose input tokens the poolings read (D10): the
    output layer's, or the first StateInput's for BEFORE_STATE.
    """
    if recurrence.pooling.reads == BEFORE_STATE:
        return next(i for i, spec in enumerate(specs) if isinstance(spec, StateInput))
    return len(specs) - 1
