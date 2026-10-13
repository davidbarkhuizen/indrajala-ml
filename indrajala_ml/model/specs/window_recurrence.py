"""
A window-recurrent network's recurrence as backend-free data (the window-recurrent adder workplan,
The network): what it adds to its window, a sequence model's layer specs (layer_specs.py) over one
window of k columns, k the window's input length.

Per window, with state c of state_size features: the window's tokens, its StateInputs adding P c
before their blocks, give the window's k digits; the pooling reads the tokens the output layer reads
and gives the window's summary u; the cell gives h = cell(c, u); and the codes snap h to the
nearest of their count learned codes, straight-through, the next window's state (W1, W4). The
first window's state is a learned vector, snapped alike (W8). Without codes (codes=None) the state
is continuous: h itself is the next window's state, and the codes' losses are gone (the
window-recurrent claims workplan's ablation, E2, D3).

    WindowRecurrence(state_size=4, codes=Codes(8))
    WindowRecurrence(state_size=4, codes=None)

validate_window_recurrent (spec_validation.py) checks a window and its recurrence together.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRU:
    """
    The cell (W1b-iii; Cho et al. 2014): reset and update gates r and z, the candidate h~ = tanh(W u
    + U (r * c)), and h = z * c + (1 - z) * h~, reading the summary u and the state c. update_bias is
    the update gate's bias at initialisation, toward keeping the state (W13).
    """

    update_bias: float = 1.0


@dataclass(frozen=True)
class AttentionPooling:
    """
    The window's summary (W1b-i): one learned query, one head, over the tokens the window's output
    layer reads, u = the attention-weighted sum of their values.
    """


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
class WindowRecurrence:
    """
    What a window-recurrent network adds to its window's specs: the state's width, its codes (None:
    a continuous state), the cell and the pooling.
    """

    state_size: int
    codes: Codes | None
    cell: GRU = GRU()
    pooling: AttentionPooling = AttentionPooling()
