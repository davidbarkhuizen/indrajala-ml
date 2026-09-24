# Optimizations

Performance work on the Rust backend (`rust/`, the `indrajala-math-rust` crate) against numpy,
the benchmark it is compared with. It started from the Rust CNN timing (#317-#320 and the Rust
CNN stages), which found where Rust was slower than numpy or slower than it needed to be. Each
item is measured before and after, and must keep every parity test passing.

This document and the ones under `docs/optimizations/` are the only record of optimization
work: there are no separate optimization workplans. Four earlier workplans have been folded in
(see [Old names](optimizations/history.md#old-names)).

**Where it stands:** Rust is 0.09-0.49x numpy's wall-clock time end to end on the conv demo,
and about 0.21x (single-example) and 0.54x (mini-batch 32) on a dense MNIST epoch. The gaps
left are in batch ops. Details in [Where things stand](optimizations/status.md).

## Documents

Current state:

- [Where things stand](optimizations/status.md): the end-to-end Rust / numpy ratios, the per-op
  table and where a profiled epoch spends its Rust time.
- [The kernels](optimizations/kernels.md): the threading policy and what it rests on, and each
  kernel's fixed summation order.
- [Lessons](optimizations/lessons.md): findings that apply beyond one candidate.

Work:

- [Open candidates](optimizations/candidates.md): committed work, ranked, with stage plans.
- [Future work](optimizations/future-work.md): deferred work, unmeasured leads and open
  questions.
- [Method](optimizations/method.md): how to measure, and the rules for an optimization PR.

Record:

- [History](optimizations/history.md): every finished or closed change with its measurements,
  each candidate as recorded while it was open, and numbers since replaced.

## Candidates

Candidate numbers are permanent: re-ranking changes the rank column, never the numbers, so a
reference to "candidate 2" in code or a PR stays valid. Open candidates are ranked by the Rust
time each would save in configurations that are actually trained (see
[Open candidates](optimizations/candidates.md)).

| # | candidate | status | rank | record |
| --- | --- | --- | --- | --- |
| 1 | The dataset as one backend array | done (crate #19, #372-#374) | - | [history](optimizations/history.md#candidate-1-the-dataset-as-one-backend-array) |
| 2 | A batched accuracy pass | done (#378-#379); Rust conv keeps the per-row pass | - | [history](optimizations/history.md#candidate-2-a-batched-accuracy-pass) |
| 3 | Dense single-example `downstream` through the tiled kernel | done (crate #20) | - | [history](optimizations/history.md#candidate-3-dense-single-example-downstream-through-the-tiled-kernel) |
| 4 | Conv `forward_batch` one example at a time | done (#383, crate #21); stage 2 closed (#385) | - | [history](optimizations/history.md#candidate-4-conv-forward_batch-one-example-at-a-time) |
| 5 | `max_pool_forward_batch` | open, not yet examined | 1 | [candidates](optimizations/candidates.md) |
| 6 | Dense `downstream_batch` and `accumulate_gradient_batch` at short `k` | open: stage 0 done (#354), next stage A then B | 2 | [candidates](optimizations/candidates.md) |
| 7 | Dense `accumulate_gradient_batch` at long `k` | open: waits on a one-thread `k`-blocking probe | 3 | [candidates](optimizations/candidates.md) |
| 8 | Conv accumulate with a large `cols` | open: alongside candidate 7 | 4 | [candidates](optimizations/candidates.md) |
| 9 | Threading past the threshold | deferred until something trains at large batch | - | [future work](optimizations/future-work.md#deferred) |
| 10 | Leads from candidate 4 | retired: now two of the [leads](optimizations/future-work.md#leads) | - | - |

## Keeping the documents apart

- **Current-state files hold current values only.** A PR that changes a number overwrites it in
  `status.md` or `kernels.md` and adds the old value, with the PR that replaced it, to
  [Superseded numbers](optimizations/history.md#superseded-numbers).
- **Closing a candidate is three edits:** its full record moves from `candidates.md` (or
  `future-work.md`) to `history.md`, and its row in the table above changes status and link.
- **A lead becomes a candidate** with the next free number once its stage 0 measures a stake
  worth having.
- **A lesson that applies beyond its candidate** goes in `lessons.md`, pointing at the history
  entry that established it.
