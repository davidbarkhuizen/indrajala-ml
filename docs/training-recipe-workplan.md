# Workplan: the training recipe: schedules, AdamW and gradient clipping (roadmap step 10)

**Status: decisions D1-D10 settled (2026-10-10): D7 and D8 by the owner, the rest by building
for extensibility, speed and real-world production (the owner's direction, 2026-10-10). Stage 1
(this plan) done.**

Roadmap step 10 ([primitives-roadmap.md](primitives-roadmap.md)): the optimization recipe
production language models train with (GPT-3, Brown et al. 2020, §2.3 and appendix B: Adam with
decoupled weight decay 0.1, the gradient's global norm clipped to 1.0, linear warmup then cosine
decay): learning-rate schedules as data the trainer reads and a run file records, AdamW, and
clipping by global norm; and the sequence models trained to convergence with them.

## Why

Neither of the sequence study's transformers had converged at 10 epochs: their held-out losses
still fell by 0.01 to 0.05 bits per character over the last three (`scripts/sequence_study.py`).
They train at a constant Adam rate. The decoders the study's models follow train with warmup then
cosine decay to a tenth of the peak (Radford et al. 2018: warmup, then cosine to 0; nanoGPT:
warmup, then cosine to `min_lr = lr / 10`; Loshchilov & Hutter 2017, SGDR, for the cosine form).
A decaying rate is how such a run ends near a minimum rather than bouncing around one at the
constant rate's noise level.

Today a schedule is a Python callable (`lr_schedule.linear_warmup`, step to rate) the trainer reads
once per batch. A run file (`run_checkpoint.py`) records the step count a schedule reads, but not
the schedule: resuming a scheduled run by bits needs the caller to pass the same callable again,
and nothing checks that it did. The composable-layers workplan kept "schedulers beyond today's
`lr_schedule.py`" out of scope; the owner lifted that on 2026-10-10.

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| a schedule | a callable, `linear_warmup(rate, steps)`, or a float | frozen dataclasses, as the update rules (D2) |
| forms | linear warmup, then hold | warmup composed with constant, cosine or linear decay to a floor (D3) |
| the trainer | `learning_rate: float \| Callable[[int], float]`, read per batch at the iteration count | `float \| Schedule`; a callable still accepted (D4) |
| a run file | the step count | the schedule too, checked on resume (D5) |
| the studies | constant rates; the batch-size study's warmup | the sequence models to convergence (D7) |

## Decisions (settled 2026-10-10)

Settled by building for extensibility and speed (the owner, 2026-10-10), with the alternatives
and why they lose:

- **D1. Scope. Settled: (a).** (a) Schedules as data, warmup with constant, cosine or linear decay,
  in the trainer and run files; the batch-size study moved onto them, bit-identically; AdamW
  (D9) and global-norm clipping (D10), the other two parts of the production recipe, in all three
  implementations and the crate; a study of longer training (D7). (b) Schedules only. Cons: a
  production run needs all three, and a study of the schedule alone measures a recipe nobody
  trains with. (c) A `cosine_decay` callable beside `linear_warmup`. Cons: a run still can't
  record or check its schedule; each new form another unrecorded callable.
- **D2. Schedules as data. Settled: (a).** (a) Frozen dataclasses in
  `indrajala_ml/training/lr_schedule.py`, each with `rate(step) -> float`: `Constant(rate)`,
  `Warmup(steps, then)` (linear from `rate / steps` at step 0, as `linear_warmup`, then `then`
  counted from the end of warmup), `CosineDecay(rate, steps, floor)` and `LinearDecay(rate, steps,
  floor)` (then holding at `floor`). Pros: the update rules' pattern (README, Update rules); a run
  file can write them; a new form is a new class; composition by `then` covers warmup with any
  decay. Cons: `linear_warmup` callers move (the batch-size study; tests). (b) Callables with a
  name attribute. Cons: neither data nor checkable.
- **D3. The cosine's arithmetic. Settled.** `floor + (rate - floor) * (1 + cos(pi * s / steps)) /
  2` at step `s` of `steps`, `floor` after: the published form (SGDR's eq. (5) without restarts),
  in this grouping, with `math.cos` (the trainer is shared Python, so all three implementations
  read the same rate bits).
- **D4. The trainer's argument. Settled.** `learning_rate: float | Schedule | Callable[[int],
  float]`; a float means `Constant`. Callables stay accepted, unrecorded, so no caller breaks; a
  run file refuses to record one (D5).
- **D5. Run files. Settled.** A run file writes its schedule as data (`{"schedule": "cosine",
  ...}`); resuming checks the given schedule equals the recorded one and refuses otherwise, naming
  both. A callable schedule makes `save_run` refuse with the reason.
- **D6. Golden run and timing. Settled.** `linear_warmup` as `Warmup(steps, Constant(rate))` gives
  the same rate bits, checked by a test over its steps; the golden run passes unchanged. New
  entries: a numpy sequence model under warmup then cosine (the trainer is shared, so one
  implementation covers the schedule), and one per implementation under the full recipe (AdamW
  and clipping are in each), re-recorded on both machines, archived as `new-functionality`. The
  schedule changes no timed path (the trainer reads one rate per batch): tier 0. The step split
  (D10) and AdamW (D9) touch `learn*` and the crate: tier 1 A/Bs in their stages.

- **D9. AdamW. Settled: (a).** (a) `Adam(weight_decay=λ)`, decoupled (Loshchilov & Hutter 2019,
  "Decoupled Weight Decay Regularization", Algorithm 2): `w - lr * (m_hat / (sqrt(v_hat) + eps) +
  λ * w)`, in that grouping, on weights only (biases, `gamma`, `beta`, `P` and `E` not decayed, as
  `WeightDecay` already treats them), default 0 so `Adam()` keeps its bits. This is the published
  form the composable-layers workplan asked for before weight decay with Adam (next-steps.md).
  Pros: the production default; one field on the existing rule. Cons: the crate's Adam op gains a
  term. (b) L2 added to the gradient under Adam. Cons: Loshchilov & Hutter show it isn't
  equivalent under Adam and regularizes less; no production recipe uses it.
- **D10. Gradient clipping. Settled: (a).** (a) Clipping by the global L2 norm over every
  parameter's batch gradient (Pascanu et al. 2013), a `clip_norm` on the trainer's call
  (`None` by default): if `||g|| > c`, every gradient is scaled by `c / ||g||` before the update
  rule. It needs every layer's gradient before any update, so the network's step splits into
  accumulate-all then apply-all (deltas are computed before either, so the split keeps every
  bit). The norm's sum is a left fold over layers in order, parameters in their order. Pros: the
  production form; the split is the structure later per-step logic (gradient accumulation over
  micro-batches, mixed precision's loss scaling) also needs. Cons: one more pass over the
  gradients when on. (b) Per-layer or per-value clipping. Cons: changes the update's direction;
  not what production recipes use.

Left to the owner (2026-10-10):

- **D7. The study. Settled: (a).** `scripts/lr_schedule_study.py`, numpy, Adam, batch 32: the
  best 2-layer sequence model steps 7 to 9 settle on (positions per step 8's D9, dropout per step
  7's study), on all four corpora, 30 epochs, 5 seeds, both held-out splits of step 7 where its
  loader has them. Three arms (added by D1's recipe, 2026-10-10): a constant rate; warmup (one
  epoch) then cosine to a tenth of the peak; and the full recipe, that schedule with AdamW 0.1 and
  clipping at 1.0. About 9 hours on `jebel` with the third arm.
  - (a) *Chosen.* 4 corpora, 30 epochs, 5 seeds. Pros: every corpus at three times the sequence
    study's length, where its losses were still falling. Cons: about 6 hours on `jebel` (about 100 s
    an epoch per job at 6 workers).
  - (b) Tiny Shakespeare and Euclid, 50 epochs. Pros: closer to convergence, about 4.5 hours.
    Cons: no answer on Herodotus or the *Muqaddimah*.
  - (c) 4 corpora, 50 epochs, 3 seeds. Cons: a wider spread on every difference quoted.
- **D8. The peak rate under decay. Settled: (a).**
  - (a) *Chosen.* The cosine's peak is the sequence study's tuned constant rate for each corpus.
    Pros: free; the arms differ only in the decay, which the comparison then isolates. Cons:
    a decay usually tolerates a higher peak, so the cosine arm may be under-tuned; the study says
    so if it loses.
  - (b) Retune with the sequence study's short `tune` (2 epochs). Cons: 2 epochs say little about
    a 30-epoch decay.
  - (c) A grid of peaks (1x, 2x, 4x) at full length on Tiny Shakespeare, the best used everywhere.
    Cons: about twice the cost.

## Stages

1. **This workplan.**
2. **Schedules as data** (D2-D4): the classes, `linear_warmup` as `Warmup`, the trainer; tests of
   each form by hand; the batch-size study moved, its rates by bits.
3. **Run files** (D5): the schedule recorded and checked; resume by bits under a cosine schedule.
4. **The step split and clipping** (D10): accumulate-all then apply-all in all three
   implementations (golden run bit-identical; tier 1 A/B of the dense and attention cases, which
   must not move), then `clip_norm`.
5. **AdamW** (D9): numpy, the crate's Adam op and a "Bump rust/" PR (tier 1 A/B with
   `weight_decay=0`), Rust, pure Python; parity.
6. **Golden entries** (D6, and one under the full recipe).
7. **The study** (D7, D8).
8. **Docs**: README (Update rules, a Schedules section), roadmap, next-steps; the workplan retired.
