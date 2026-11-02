# Workplan: the implementation audit's remedies

**Status: stages 2 (the explicit loss, #1204), 3 (the k-means codes, #1205) and 4 (re-seeding,
#1206) done; stage 5 (the codebook rate, #1207) built, its jebel A/B after the GQA study; stage 6
(SimVQ) built.
Decisions D1-D11 settled 2026-11-02 by building for extensibility, speed and real-world production
(the owner's standing direction); written at the owner's request ("a detailed workplan to implement
the two supported changes, and the four optional ones").**

The implementation audit ([nn-implementation-analysis.md](../../nn-implementation-analysis.md),
§16) checked the library's design against the literature. Two changes have the literature's
support, and four are optional:

| audit item | kind | this plan |
| --- | --- | --- |
| the output layer's default loss (§16.2.1) | supported | D2, stage 2 |
| the codebook's start, unused codes and collapse (§16.2.2) | supported | D3-D6, stages 3-6 |
| a selection set for the trainer's pocket (§16.3.8) | optional | D7, stage 7 |
| He initialisation (§16.3.1) | optional | D8, stage 8 |
| a multiplicative (FiLM) state input (§16.3.2) | optional | D9, stage 9 |
| MRI, every needed unit adapted (§16.3.5) | optional (legacy) | D10, stage 10 |

**The codebook is the weightiest item** (D1): the owner reports codebook collapse as a recurring
failure of the window-recurrent network. It follows the small loss stage at once.

## Why

- **The loss.** `Dense(..., output=True)` defaults to the squared error on a sigmoid, which Glorot &
  Bengio 2010 find trains classifiers worse than cross-entropy (more severe plateaus). A network
  should name its loss rather than get the squared error by omission.
- **The codebook.** The library draws each factor's codes uniformly in `±1/√s`, wherever the GRU's
  states lie, steps them only through the selected codes' codebook loss, and never revives an unused
  code. The VQ literature traces collapse to exactly these: poor, scale-mismatched initialisation
  and non-stationary inputs (Łańcucki et al. 2020), dead codes (Dhariwal et al. 2020; Zheng &
  Vedaldi 2023), and disjoint optimisation, where only the selected codes move (Zhu et al. 2025,
  SimVQ). Huh et al. 2023 add the commitment's asymmetry.
- **The optional four.** A selection set for runs whose claims are held-out ones (Prechelt 1998);
  He initialisation for deep plain rectifier networks (He et al. 2015); FiLM's scale term, which its
  ablation found carries most of the conditioning (Perez et al. 2018); and MRI as Widrow & Lehr
  1990 define it.

## What the plan found (2026-11-02)

- **The research harness already initialises codes from the states.** The state-formation
  workplan's k-means codes (D6), in indrajala-studies (`indrajala_studies/modular_family.py`:
  `kmeans`, `codes_from_states`), sets each factor's codes by k-means++ seeding and Lloyd's
  iterations over its cells' outputs at the warm-up's end, writing `factor.codes.E`. The library's
  default is still the random draw. The audit, written from this repository alone, didn't see the
  harness; §16.2.2 is corrected with this plan.
- **What the research has tried against collapse, and what it hasn't.** Built and tried: the
  commitment off early, never on, or scheduled (the thermostat); annealing from the critical
  temperature (D7 there) with the k-means codes; a continuous warm-up; reheat and restarts (D63
  there); consolidation. Listed and shelved, chosen by fit to a diagnosed failure kind (D31 there):
  cell noise, the rotation trick, FSQ factors. **Not built anywhere:** re-seeding dead codes, a
  separate codebook learning rate, and reparameterising the codes so every code moves. Stages 4-6
  build those three, as library primitives.
- **Adoption is research, not this plan.** Each codebook primitive is a training-state option or a
  spec option that defaults to today's behaviour, bit for bit. Whether any becomes a default is for
  a registered study in indrajala-research (its experiment design: fresh-seed rates, every cell able
  to succeed), checked on every backdrop (a working cell is a candidate default). D11 names the
  failure kind each primitive targets, as the hypothesis such a study would test.
- **The window program reads the codes as a parameter.** `pa.window_program_learn` takes every
  parameter and gradient as arrays (`_program_parameters`, `_program_gradients`). So a codebook
  computed from other parameters (D6) can be formed in Python before the call and its gradient
  mapped back after it, with no crate change. Re-seeding (D4) and the k-means codes (D3) act on
  `E` between steps, also without a crate change. Only the FiLM state input (D9) changes the crate.
- **Format 2 always writes a dense layer's `loss`** (`format2_json.layer_to_json`: `asdict` minus
  `bias`). So every file, hidden layers included, carries `"loss": "squared"` or
  `"cross_entropy"`, and an explicit loss (D2) can keep every file byte for byte.
- **The linear classifier isn't in the golden run** (`data/refactoring/golden_run.json`: 143
  entries, none a `LinearClassifierNetwork`), so correcting MRI (D10) moves no golden bit; its
  tests pin its behaviour and move with it.
- **The pure-Python window-recurrent network exists** (`networks/python/window_recurrent_backprop_network.py`,
  `layers/python/recurrent_layer.py`), so every codebook primitive is built in all three
  implementations, as the library's parity requires.

## Decisions (settled 2026-11-02)

Each lists the options, with pros and cons, and the choice.

- **D1. Order. Settled: (a) (made exact 2026-11-02).**
  - (a) *Chosen.* The stages in number order: the loss (stage 2) first, then the codebook (3-6),
    then the optional four (7-10). The loss stage is small (an hour's mechanical edits) and changes
    every spec list that writes an output layer; done first, the codebook stages' new tests and
    windows are written once, already naming their loss. The codebook, the owner's recurring
    failure, follows at once. Pros: no spec list edited twice; the codebook's primitives a stage
    later at most. Cons: none found.
  - (b) The codebook first (3-6), then the loss. Cons: stages 3-6's new spec lists edited again in
    stage 2.
  - (c) The codebook only, the rest later. Cons: leaves the supported loss change waiting.
- **D2. The output layer's loss. Settled: (a).**
  - (a) *Chosen.* `Dense.loss: Loss | None = None`. An output layer must name its loss
    (`validate_layer_specs` refuses `output=True` with `loss=None`, naming the two choices). A
    hidden layer's is `None`, and naming one there is refused. The presets' `_output_spec` names
    `"squared"` (and the cross-entropy and softmax presets theirs, as today), so every preset's bits
    are unchanged. Format 2 keeps its files byte for byte: the writer writes a hidden layer's
    `None` as `"squared"`, which every existing file holds; the reader reads a hidden layer's
    `"squared"` as `None`. Pros: no new network gets the squared error by omission; no golden bit
    or file moves. Cons: every spec list in tests, scripts and studies that writes
    `Dense(n, output=True)` must name a loss: about 200 sites, mechanical.
  - (b) The default changed to `"cross_entropy"`. Cons: every preset's training bits move (a
    golden re-record), and a sigmoid output with BCE silently replaces the squared error in files
    that don't name it.
  - (c) Keep the default and document it. Cons: the omission the audit found stays the easy path.
- **D3. The k-means codes in the library. Settled: (a).**
  - (a) *Chosen.* A library function, `initialise_codes(network, rows, rng, iterations)`
    (`indrajala_ml/model/codes.py`), ported from the harness's `kmeans` and the code-setting part
    of `codes_from_states`: per snapped factor, the cells' outputs over `rows` (`window_states`'s
    `H`), k-means++ seeding and `iterations` Lloyd's iterations from `rng` (numpy's `Generator`),
    an empty cluster's centre a drawn state moved by a small random step, the codes written to
    `E`. Same draws, in the same order, as the harness, so a study switching to it keeps its bits.
    It returns per factor the codes in use and the within-cluster statistics the harness logs.
    Studies switch in an indrajala-studies PR after this repository's bump. Pros: one
    implementation; every network, not only the harness, can start its codes from its states (the
    library's general architecture). Cons: the harness's function is retired in a second
    repository.
  - (b) Leave it in the harness. Cons: the library's default stays the collapse-prone random draw
    with no library alternative.
  - (c) A crate kernel. Cons: no speed need (it runs once per run); a third implementation of the
    same arithmetic.
- **D4. Re-seeding dead codes. Settled: (a), (b) for the target, (a) for the optimiser state.**
  - *When a code is dead* (a): codes used by no state on an inference pass over a caller-given
    sample (`state_codes`), at or below `min_use` uses (default 0). Pros: backend-free (the Rust
    window program's training step doesn't expose its snaps' indices); the caller sets the cadence
    (every n epochs or at a screen), as the harness sets its screens; deterministic. Cons: a code
    used only in training but not on the sample counts as dead; the sample must cover the task.
    Rejected: counting the training step's snaps (needs the crate to return them; ties the cadence
    to the step).
  - *Where a dead code goes:*
    - (a) A state drawn uniformly from the sample (Dhariwal et al.'s restarts). Cons: it lands where
      states are dense, likely on a live code's region.
    - (b) *Chosen.* A state drawn with probability proportional to its squared distance to its
      nearest live code (k-means++'s `D²` rule, which D3's code already has). Pros: it lands where
      the codebook serves the states worst, which is where a missing state is; one rule shared
      with D3. Cons: drawn toward outliers; mitigated by drawing from the states, not beyond them.
    - (c) Split the busiest code: a perturbed copy (deterministic annealing's duplicate). Pros:
      Rose's and Mavridis's bifurcation. Cons: the annealing already does this; it targets a
      code's load, not the states' coverage.
  - *The re-seeded code's optimiser state:*
    - (a) *Chosen.* Its rows of every rule state (Adam's `m` and `v`, momentum's velocity) set to
      zero, through a new optimiser method `reset_rows(index, parameter, rows)`. Pros: a new code
      doesn't inherit a dead code's moments, whose stale `v` would shrink its first steps. Cons: an
      optimiser API addition.
    - (b) Keep the state. Cons: Adam's `v` from a code that stopped receiving gradients keeps
      decaying, and so mis-scales the new code's early steps.
  - The API: `network.reseed_dead_codes(X, rng, min_use=0) -> list[list[int]]`, returning each
    factor's re-seeded codes for the caller to log. Training state; nothing saved. **The minimal
    machine's concern:** re-seeding raises the codes in use, which a minimal machine doesn't want;
    it is an option, and a study decides (D11).
- **D5. A codebook learning rate. Settled: (a).**
  - (a) *Chosen.* A per-part `learning_rate_scale` (default 1.0), set by
    `network.set_code_learning_rate_scale(scales)`, one per factor. `OptimizerBase.step` reads
    each layer's or part's scale and passes `learning_rate * scale` to its `apply`, and passes the
    rate unchanged where no scale is set, so every other step keeps its bits. Generic: any layer or
    part can carry a scale. Łańcucki et al. raise the codebook's rate. Pros: one multiplication;
    every rule (SGD, momentum, Adam, AdamW) and both array optimisers at once; the pure-Python
    optimiser likewise. Cons: under AdamW the decay `lr·λ` scales too (codes aren't decayed, so
    moot).
  - (b) A second optimiser for the codes. Cons: two step counts, two checkpoints' states.
  - (c) Parameter groups, a general per-parameter rule. Cons: the general machinery for one need;
    (a) leaves it open.
- **D6. Reparameterising the codes so every code moves. Settled: (a).**
  - (a) *Chosen.* SimVQ (Zhu et al. 2025): `E = Q W`, `Q` `(m, s)` a fixed basis, `W` `(s, s)`
    trained, so a gradient from any selected code moves every code through `W`. `Q` is the codes'
    initial draw (random, or D3's k-means codes), and `W` starts at the identity, so the network
    starts with today's `E`. A spec field, `Codes(parameterisation="direct" | "linear")`, written
    to format 2 only when `"linear"`. The part's `parameters()` is `(W,)` and `Q` is held as
    running state (saved, never stepped). Forward, everywhere: `E` formed as `Q W` before the
    snap (on Rust, before `window_program_learn`, from the program's parameter list); backward:
    `dW = Qᵀ dE`, the program's `dE` read from a scratch gradient. No crate change. Pros: a
    published fix for disjoint optimisation; one product per step; composes with D3 (k-means sets
    `Q`), D4 (a re-seed writes `Q`'s row and keeps `W`) and the soft codes, consolidation and
    temperatures, which all read `E`. Cons: `W` couples the codes, so a code a factor needs fixed
    can drift with the rest; a study measures it.
  - (b) Huh et al.'s affine reparameterisation with alternating optimisation and a synchronised
    commitment. Cons: alternating updates break the one-step training loop that every
    implementation and the window program share; its affine form is a special case of (a)'s
    linear map.
  - (c) EMA codebook updates (van den Oord et al.'s appendix). Cons: replaces the codebook's
    gradient with running averages, which the soft codes (`softmax(−d/τ)`'s weighted codebook
    loss) and the consolidation don't define; a second update path beside the optimiser's.
  - (d) The rotation trick, or FSQ factors. Out of scope: the state-formation workplan lists both
    as its options 3 and 4, shelved under its D31; a build is that plan's call.
- **D7. A selection set for the pocket. Settled: (a).**
  - (a) *Chosen.* `train_backprop_network_mini_batch(..., selection=None)`: examples or a
    `PreparedDataset`; when given, each epoch's pocket compares accuracy on it, not on the training
    set, still strictly better (Gallant's ratchet). Training accuracy is still measured and
    recorded each epoch. The `RunCheckpoint` gains `selected_on` (`"training"` or `"selection"`)
    and the selection accuracies, written to its JSON only when a selection set is used, so
    existing run files are unchanged; resuming under a different `selected_on` is refused, as a
    different schedule is. `None` is today's path, bit for bit. Pros: Prechelt's validation-based
    selection when a claim is held-out; Gallant's when fitting is the goal; the run says which.
    Cons: an extra inference pass per epoch over the selection set.
  - (b) Validation-based early stopping (stop training on a validation rise). Cons: a stopping
    rule changes the epoch budget, which the studies fix (epochs are the time axis); selection
    alone keeps it.
  - (c) Leave the trainer. Cons: a held-out claim from a pocketed network stays biased.
- **D8. He initialisation. Settled: (a).**
  - (a) *Chosen.* `init: Literal["fan_in", "he"] = "fan_in"` on `Dense` (hidden) and `Conv`,
    written to format 2 only when `"he"`. `"he"` draws `W` uniform in `±√(6 / fan_in)` (variance
    `2/fan_in`, He et al.'s, as Kaiming-uniform) and sets the bias to zero, drawing `W` only, so
    it reuses the uniform draw all three generators already share. The other layers draw as today
    whatever their neighbours' `init`. Pros: He's variance with no new random algorithm; per
    layer, so it follows the activation. Cons: uniform, not He's normal, which matches its variance
    but not its shape.
  - (b) He's normal `N(0, 2/fan_in)`. Cons: a normal draw bit for bit with numpy's
    `standard_normal` (its ziggurat) in the pure-Python port and the crate's generator: a third
    generator algorithm for one option.
  - (c) A network-wide `init` argument. Cons: one scale for sigmoid and ReLU layers alike, where
    He's derivation is the rectifier's.
- **D9. A multiplicative state input. Settled: (a).**
  - (a) *Chosen.* `StateInput(scale=False)`; with `scale=True`, `out_t = x_t ⊙ (1 + G c) + P c`,
    `G` `(d, s)`, starting at zero and drawing nothing, so a seed's `P` draws are unchanged and the
    layer starts as the additive one (FiLM's `γ = 1 + Δγ`). `G` decayed, as `P`. Backward:
    `δx_t = δ_t ⊙ (1 + G c)`; `dc = (Σ_t δ_t) P + (Σ_t δ_t ⊙ x_t) G`; `dG = Σ_n (Σ_t δ_t ⊙ x_t) ⊗ c`.
    All three implementations, and the crate: `state_input_forward`/`_backward` gain an optional
    `G`, and the window program's `state_input` op its flag, in `indrajala-math-rust` first, then a
    bump here. Pros: FiLM's scale, the part its ablation found most of the conditioning in; a no-op
    at initialisation, so a study compares like with like. Cons: a crate change and a window-program
    op change.
  - (b) `γ = G c` without the `1 +`, with `G` initialised so `γ ≈ 1`. Cons: needs a bias or a
    special initialisation, and starts away from the additive layer.
  - (c) numpy and pure Python only. Cons: a window with it can't run on Rust, the studies' backend.
- **D10. MRI. Settled: (a).**
  - (a) *Chosen.* `LinearClassifierNetwork.learn` adapts as many hidden units as the output needs
    to change, nearest zero first (Widrow & Lehr's minimal disturbance): on a false negative,
    `required_active − active` inactive units; on a false positive, `active − required_active + 1`
    active units; each by today's step along its input. Replaces the one-unit rule (not in the
    golden run; its tests move). Pros: MRI as published; the network's documentation already
    claims the MADALINE committee. Cons: a legacy network's behaviour changes.
  - (b) Keep one unit per presentation as an option beside it. Cons: two rules for a demo network.
- **D11. What each codebook primitive is for, and who judges it. Settled.** The plan builds; a
  registered study in indrajala-research judges, by fresh-seed certified rates on every backdrop,
  each primitive against the failure kinds (`failure-kinds.md`) it may move:
  - D3, the k-means codes in the library: conversion (already the harness's warm-up's practice);
  - D4, re-seeding: formation's few-codes signature (1-8 codes a factor of the 15 available) and the late, abrupt
    index collapse after full use;
  - D5, the codebook rate: the codes lagging the states (Łańcucki's non-stationarity);
  - D6, SimVQ: unused codes never joining (disjoint optimisation), which the literature note
    already maps to the dead codes.

  This plan's last stage writes that hand-off into the state-formation workplan's options, as
  candidates, not as a study.

## Stages

Each stage is one PR, merged before the next, with tests in all three implementations where the
item has them, the golden run (new entries where the stage adds an option; no existing entry may
move), and its timing tier (docs/measurement.md §1).

1. **This plan.** With the audit's §16.2.2 corrected (the harness's k-means codes) and the
   workplans README's row. Tier 0.
2. **The explicit loss (D2).** `Dense.loss` optional; validation; the presets' `_output_spec`;
   format 2's writer and reader; every spec list naming its loss. Tests: refusals; every fixture
   and saved model loads and re-saves byte for byte. Golden: unchanged. Tier 0 (no timed path).
3. **The k-means codes (D3).** *Built:* `codes.py` splits the harness's `codes_from_states` into
   `factor_states` (the states, batch by batch then window by window, through each network's new
   `cell_outputs`) and `fit_codes` (k-means per snapped factor, written through `set_codes`), so
   the harness keeps its melting curve between them and its draws in order; `initialise_codes` is
   the two. The bits test is against the harness's own `kmeans` on five cases, a window network's
   states among them (`tests/fixtures/codes/harness_kmeans.json`, recorded from its source); three
   golden entries, one per implementation. Planned: `indrajala_ml/model/codes.py` with
   `initialise_codes`; the window networks (numpy, Rust, pure Python) through `window_states`. Tests: the harness's `kmeans` and
   the library's agree by bits on its saved samples (a fixture of a few runs' states); an empty
   cluster's perturbation; a continuous factor untouched. Golden: one entry (a window network,
   codes from its states, one step). Tier 0. Then the indrajala-studies PR: bump `ml`, call the
   library, delete `kmeans`, its golden replay unchanged.
4. **Re-seeding dead codes (D4).** *Built:* `codes.reseed_dead_codes(network, batches, rng,
   min_use=0)`, a function beside `initialise_codes` rather than a network method, so its
   arithmetic is written once for the three implementations. A code's uses count the sample's
   initial states and windows' states (`cell_outputs(first=True)`), snapped as the network snaps
   (`snap_indices`, the left fold); the dead codes move in index order, each drawn by `d2_draw`
   (k-means++'s seeding now calls it, its bits unchanged) against the live codes and those already
   moved; the network writes them and zeroes their optimizer rows (`reseed_codes`, then
   `reset_rows`). Golden: three entries, one per implementation, re-seeded between the
   single-example and the batch steps. Planned: `reseed_dead_codes`; the optimisers' `reset_rows` (numpy, Rust,
   pure Python); the `D²` draw shared with stage 3. Tests: a code dead on the sample is re-seeded
   to a sample state, a live one untouched; its Adam rows are zero; Rust and numpy agree. Golden:
   one entry (a re-seed between two steps). Tier 0.
5. **The codebook rate (D5).** `learning_rate_scale` in `OptimizerBase.step`; the setter. Tests:
   a scale of 1.0 keeps every bit; a scale of `k` under SGD is a `k`-times step on the codes alone,
   under Adam as Adam at `k·lr`. Golden: unchanged, plus one entry at a scale of 4. Tier 1: one
   A/B of `prepared_dataset_timing`, `--order ONNO` (the step's loop gained a read).
6. **SimVQ (D6).** *Built:* a linear codes part (numpy, Rust, pure Python) keeps E concrete,
   formed as `Q W` once a step: before the optimizer's step `dW = Qᵀ dE` (`before_step`), after
   it E anew (`after_step`); the snap, its losses and the window program read and accumulate E as
   for the direct codes (the program's E and dE slots, `_program_arrays`), and the optimizer sees
   `W` alone. Rust forms both products through numpy's matmul, so numpy's E by bits and no crate
   change. Q is running state, saved after W (numpy, Rust: `[W, Q]`; pure Python: W's rows then
   Q's). *A change from D6's text:* writing Q's row with W kept would put a k-means or re-seeded
   code at `x W`, not at the state x; so `set_codes` (D3's) sets Q and W back to the identity
   (with W's optimizer state), and a re-seed (D4's) solves Q's row so E's row is the state, W kept.
   Golden: three entries. Planned: `Codes(parameterisation="linear")`; the codes parts (numpy, Rust, pure Python)
   with `Q` as running state and `W` stepped; `E = Q W` before the window program's calls and
   `dW = Qᵀ dE` after; format 2's field; D3 and D4 writing `Q`. Tests: at `W = I` the forward is
   the direct codes' by bits; the gradient against a finite difference; Rust against numpy;
   save and load. Golden: one entry. Tier 0 (the default path untouched; the window program's
   call unchanged).
7. **The selection set (D7).** The trainer's `selection`; `RunCheckpoint`'s fields and JSON; the
   resume refusal. Tests: `None` keeps every bit; a selection set picks its own best epoch; resume
   by bits with it. Golden: unchanged (the golden run trains without one). Tier 1:
   `prepared_dataset_timing`, `--order ONNO`.
8. **He initialisation (D8).** *Built:* each backend's `he_weights` (numpy, Rust) and
   `fan_in_aware_init.he_weights` (pure Python), the shared uniform draw within `sqrt(6/fan_in)`;
   `randomize` reads each built layer's expanded spec beside it, so every layer a Dense or Conv
   builds (flat, token-wise, affine, linear, conv, linear conv) draws He's W and a zero bias. The
   output layer's `init="he"` is refused. Golden: three entries whose weights aren't injected, so
   they pin the draw. Planned: `init` on `Dense` and `Conv`; `randomize` in all three; format 2.
   Tests: the draw's bounds and variance; numpy, Rust and pure Python draw the same `W` from one
   seed; `"fan_in"` keeps every bit. Golden: one entry (a ReLU network with `"he"`). Tier 0.
9. **The FiLM state input (D9).** 9a, in `indrajala-math-rust`: `state_input_forward`/`_backward`
   with an optional `G`, and the program's op flag, with the crate's tests. 9b, here: the bump;
   `StateInput(scale=True)` in the three implementations and the window program's op; format 2.
   Tests: at `G = 0` the additive layer's bits; the gradient against a finite difference; the
   window program against the per-layer path by bits. Golden: one entry. Tier 1: the window
   program's timing (indrajala-studies' `window_study_timing.py` through `ab.py`'s `cmd`), since
   the op changed.
10. **MRI (D10).** `LinearClassifierNetwork.learn`; its tests. Tier 0 (pure Python, never timed).
11. **Retire.** The hand-off (D11) into the state-formation workplan's options; the workplans
    README; next-steps; this plan deleted.
