# Information flow and its constraints (a feasibility checklist)

**Written 2026-10-19 at the owner's request: "we should be starting to think in terms of the
constraints that our architecture imposes on the flow-through of information - we should always be
sure that, if things work as we want, that the solution is at least always feasible".** A diagram of
one window step, with these constraints marked where they act, is in the owner's artifacts ("Window
Information Flow"). This file is the register; [experiment-design.md](experiment-design.md),
section 1, requires every study to check its cells against it before they run.

## One window step

The window's tokens (one column, `k` 1, and the query tokens) and the snapped state `c` enter one
residual stream of width `D`: Embedding and Position, with a StateInput adding `P c` before every
block. Three transformer blocks (`H` heads, FFN `2D`) work on it, and its output tokens feed two
consumers:

- **the transition (`delta`)**: each factor `g`'s pooling (one learned query), its cell (state
  `s_g` wide) and its snap to one of `m_g` codes give the next state `c`, the only memory carried
  to the next window;
- **the output (`lambda`)**: the linear output layer gives the answers, the target's and each
  approximation's.

Everything both consumers need passes through the stream, so its width is a capacity bound like
the codes'.

## The register

| # | Where it acts | Requirement | Why, and the evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | the snap | joint codes `m = prod m_g` at least the target's minimal states `q`; a surplus of 3-5x | fewer can't represent it (E7: 32 codes for 49 states, 20 runs that couldn't work) | imposed rule |
| 2 | the training data | longest length `L >= m + q - 1` (the exclusion bound) | below it a wrong machine fits every case (E8: 37 of 40 memorised); with an unknown target a state bound stands in for `q` | imposed rule |
| 3 | the residual stream | width `D` enough for every consumer at once | division 13: 0 of 6 at 16, 3 of 6 at 22, 5 of 6 at 32; L3's 16-wide stream held an exact state but couldn't separate the answers; the base is now 32 | measured (2026-10-19) |
| 4 | attention, per consumer | splitting the stream adds no capacity; each consumer keeps enough heads and width | a single-head branch per consumer at 16 starved the transitions (B1 1 of 6; B3's S4 0 of 5 against 4 of 6) | measured (2026-10-19) |
| 5 | the output layer | the answers linearly separable on the stream's features | the layer is linear; no linear readout of L3's 16 features fixed its last 1% (the separability check) | measured |
| 6 | the poolings | each reads the window's output, not the tokens before the state | before the state a cell must compute the whole transition (the D10 check: 0 of 2 against 2 of 2) | imposed rule |
| 7 | the codes | one code, one state (pairs per code 1.0) | every certified network has it; bottleneck 2 is its failure; width and branches don't fix it; R3 and R4 under test | required, open |
| 8 | the commitment | a state can form before the commitment pulls outputs to their codes | off early, 3 of 6 failure-A runs formed a state; the thermostat's code-use signal returns it too soon (its probe so far) | required, open |
| 9 | the cell | each factor's state `s_g` enough for its transition | 32 in use; 16 to 32 (L3) didn't cure division 13 | design choice |
| 10 | the window | `k` 1: the state handed on every symbol | keeps the step one symbol and the certificate fast; a chunked step is another capacity question | design choice |
| 11 | the data | every example ends in a final zero window, whose answer needs the state | without it a one-column step passes on the column alone | imposed rule |
| 12 | the state input | the state reaches every block (`P c` before each) | W11's window | design choice |
| 13 | training control | stop at the first certified step | training past a proof destroys it (E8b) | imposed rule |
| 14 | the proof | every state reachable and every pair told apart within the check; near ties branched | sampled tests overstate (networks passing 328,000 cases provably wrong) | imposed rule |

## Before a study

1. Codes and lengths (rows 1, 2): computed and written beside each cell.
2. The stream (row 3): at least the width that worked on a comparable cell.
3. Any split or new path (row 4): does it take capacity from a consumer that already needed it?
4. The readout (row 5): can a linear layer separate the answers at this width?
5. The proof (row 14): can the check see every state?

**What would break it next:** more consumers (each added approximation's answers share the stream:
rows 3, 5); more states (more codes, longer training, more of the stream for the state: rows 1-3);
chunked windows (rows 3, 10); discovery (an unknown target makes a state bound mandatory: rows 1,
2; [machine-discovery.md](machine-discovery.md)).
