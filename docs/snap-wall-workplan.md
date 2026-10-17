# Workplan: an asymmetric margin on the snap (the snap wall)

**Status: written 2026-10-16, at the owner's request ("a margin term ... asymmetric so that it's
easier to fall into a code boundary than out - making the process stable but dynamic"), after
E2b' showed a certified network losing its proof. D1-D3 settled by the owner, D4-D8 under the
owner's delegation, each with its options' pros and cons. P1 registered here, before its data
(#787). Stage 1 done 2026-10-16: the wall in numpy's codes, the gate, `replay` (#788). **P1 ran
2026-10-16: C1 refuted** (wall on lost the proof in 8 of 10 replays, wall off in 9 of 10). **Paused
by the owner** (2026-10-16: "let's pause investigation of the mechanism for the moment"), the
`p` 11 ladder with the fork going first; stages 3-5 wait, and this plan stays open until resumed
or its leftovers move to next-steps.** A stable-seed replay probe (exploratory, 2026-10-16, stage
2b) found seed 6's collapse is its network's: from 4 other seeds' step-7 networks, wall off, 0 of
20 replays lost the proof. A machine-diff probe (exploratory, 2026-10-17, stage 2c) found every
collapse begins as 1-3 (state, window) pairs flipped by an ordinary update, then cascades under
Adam's amplified updates within 1-13 batches. **The certificate guard works** (stage 2d, registered
in its script before the data, 2026-10-17): undoing each update that makes a right machine wrong
kept the proof in 10 of 10 of the same replays, against 1 of 10 without it.

## Why

E2b' (docs/window-recurrent-report.md, D39) trains every snapped run past its first
certified step. Seed 6 (`curriculum-linear-k1-s16-m16-seed6`) was certified at steps 5-7 and lost
its proof in step 8's first epoch (screen 1.000 to 0.404). A replay from its saved step-7 network
(exact: it ends the epoch at the recorded 0.404) shows the mechanism, batch by batch:

- **The trigger is the snap's discontinuity.** At batch 977 an update no larger than its
  neighbours' (norm 0.041, no parameter moved more than 0.006) took that batch's loss from 0.001
  to 2.2: states crossed a code boundary. Batch 947 did the same (0.0005 to 3.4) and recovered.
  The task's loss is flat up to a boundary, because the snap hides how near `h` is to it, so
  training has no warning of a crossing.
- **Adam amplifies it.** After hundreds of batches at a loss near 1e-4 its second moments are
  small; with batch 978's loss of 10.7 the update norms rose about 6-fold (0.04 to 0.26) and
  stayed at 6-8 times through the epoch's end, knocking the other states off their codes.
- **Not the new length:** an epoch more of step 7's own data collapsed too (between batches 512
  and 576); a fresh Adam's (a different trajectory) didn't, within the epoch.

So a certified snapped network isn't a stable point of further training, and the default rule
ending a run at its first certified step (the factored-state workplan's D7) is protective.
Division's large drops (stage 5's trace, docs/next-steps.md) are probably the same mechanism.

## The wall

For each state `h` snapped to its nearest code `e1`, with `e2` the next nearest, `q1 = ||h - e1||^2`
and `q2 = ||h - e2||^2`, the margin is `s = log((q2 + eps) / (q1 + eps))`, the log of the squared
distance ratio the certificate reads (it asks `sqrt(q2 / q1) >= 1.05`, SNAP_RATIO). The wall's
loss, per state:

    L = wall * weight * gate * max(0, log(ratio^2) - s)^2

with `weight` the network's (1 / its window count, as the commitment loss's), `gate` 1 for a state
of an example whose every digit the network answers right and 0 otherwise, `ratio` the margin
the wall keeps (D4) and `wall` its weight (D5). Its gradient goes to `h` (back through the GRU,
as every state's) and to `e1` and `e2`.

**The asymmetry.** Falling in is free: the commitment loss already pulls `h` to its nearest code,
and the wall exerts nothing on a state deeper than the ratio. Falling out is hard when it would
cost a right answer: the wall pushes a right example's state back from a boundary before it
crosses, the warning the task's loss can't give. A wrong example's states feel no wall, so the
task's gradient moves them across freely: what works is held, what doesn't is free.

## Decisions

- **D1. What the gate reads. Settled by the owner: (a).**
  - (a) *Chosen.* **Per example: all its digits right.** Pros: conservative and simple; a state
    that leads to any wrong digit is left free. Cons: one wrong digit frees every state of the
    example.
  - (b) Per window: the window's digits right. Pros: finer. Cons: a window can be right while its
    state is already wrong for the columns after it.
  - (c) No gate. Pros: simplest. Cons: it would hold wrong states too and block learning.
- **D2. The wall's shape. Settled by the owner: (a).**
  - (a) *Chosen.* **A squared hinge on the log ratio.** Pros: its force starts at zero at the
    ratio, so it adds no discontinuity of its own. Cons: a weak force just inside the ratio.
  - (b) A linear hinge. Pros: a firm force at once. Cons: a kink at the ratio.
- **D3. The codebook. Settled by the owner: (a).**
  - (a) *Chosen.* **The wall's gradient reaches `e1` and `e2` too:** the boundary moves from the
    state as the state moves from the boundary. Cons: one more path to test.
  - (b) States only. Pros: simpler. Cons: the codes drift as now.
- **D4. The ratio. Settled: (b), under the owner's delegation.**
  - (a) 1.05, the certificate's. Pros: trains exactly the certificate's condition. Cons: no
    headroom; a step can still cross from 1.05.
  - (b) *Chosen.* **1.5.** Pros: headroom well above the certificate's, so a held network stays
    robustly certified; a deep state (ratio above 1.5) feels nothing, so the wall is local. Cons:
    a guess, not fitted; states that must sit near a boundary for a while (early in training)
    feel it, but those are mostly wrong examples', ungated.
  - (c) 2 or more. Pros: more headroom. Cons: pushes more states, less "dynamic".
- **D5. The wall's weight. Settled: (a), under the owner's delegation.**
  - (a) *Chosen.* **1.** Pros: at a boundary (`s` 0) the loss is `log(2.25)^2 = 0.66` a state
    times the weight, the order of a wrong digit's cross-entropy, so it can hold against a small
    task gradient but not a large one (the dynamic half). Cons: one setting, not searched.
  - (b) The commitment's 0.25. Pros: matches an existing scale. Cons: weaker than the loss it
    must hold against.
  - (c) A search. Cons: against existence-first; only if (a) fails for a reason a scale explains.
- **D6. Where the probe runs. Settled: (a).**
  - (a) *Chosen.* **The numpy reference implementation,** the saved network converted (the same
    weights, Adam's moments and generator). Pros: no crate change before the idea is shown to work;
    12 ms a batch. Cons: numpy and the crate differ in rounding, so a replay's trajectory isn't the
    run's; P1 counts rates, not one trajectory.
  - (b) The crate first. Pros: the run's implementation. Cons: a crate stage and a bump before
    any evidence.
- **D7. P1's measure. Settled: (a).**
  - (a) *Chosen.* **The certificate after each epoch** (certify_step's, both verdicts recorded),
    with the screen beside it. Pros: the proof is the thing lost; 5 ms a certificate. Cons: none
    at this size.
  - (b) The screen alone. Cons: sampled tests overstate "works".
- **D8. The wall's settings' home in stage 1. Settled: (a).**
  - (a) *Chosen.* **Attributes on `CodesArrayLayer`, off (weight 0) by default, set by the
    replay tool;** the spec, the format and the crate come in stage 3 if P1 works. Pros: the golden
    run untouched (bit-identical, checked); no format change for a probe. Cons: a saved network
    doesn't record its wall until stage 3.
  - (b) A `Factor` field and format change now. Cons: the persistence work before the evidence.

## P1. The wall holds a proof (registered 2026-10-16, before its data)

**Design.** From E2b' seed 6's step-7 network (certified, `jebel`
`data/window-recurrent/e2b-prime/models/curriculum-linear-k1-s16-m16-seed6/step-7.network.json`),
converted to numpy (D6). Two arms on the same 10 replays: **wall off** (weight 0) and **wall on**
(weight 1, ratio 1.5). Replay `i` (0-9) trains 2 epochs of step 8's data (data window counts 1-8,
the run's settings: 65,536 examples an epoch, batch 64, rate 0.003), its generator
`snap-wall-p1-<i>-<epoch>`, the same in both arms. Feasibility: the start is certified, so every
replay can keep its proof; nothing is trained from scratch, so the exclusion bound (18) is not at
issue (the certificate, not the training lengths, decides).

**Measures.** After each epoch: the certificate (D7) and the screen (500 cases a property at
lengths 1-8). A replay **keeps** its proof when it is certified after both epochs. Beside them,
descriptive: each epoch's least snap ratio over the screen's states, the batches whose loss rose
above 1 in one update, the update norms.

**Claim C1.** The wall holds a proof that further training otherwise loses.
- **Works:** wall on keeps the proof in at least 9 of 10 replays, and wall off loses it in at
  least 3 of 10.
- **Refuted:** wall on loses the proof in 3 or more of 10.
- **Can't tell:** wall off loses it in fewer than 3 (the baseline doesn't collapse in numpy at
  this length): then P1', the same with 5 epochs a replay, registered before its data. Anything
  else between: inconclusive, recorded, and the next probe decided on it.

**Cost.** About 15 s an epoch, so about 10-20 minutes on `pyramidon` at 2 workers.

## Stages

1. **The reference implementation and the replay tool** (a PR): the wall in numpy's
   `CodesArrayLayer` (D2, D3), the gate in `_accumulate_gradients` (D1), off by default (D8); a
   gradient test against central differences with the wall on; `window_recurrent_study.py replay`:
   a saved step's network, a step's data for some epochs, per batch the loss before and after and
   the update's norm, per epoch the screen and the certificate, the wall's settings and a
   conversion to numpy as options. Golden run bit-identical.
2. **P1** (this plan): **ran 2026-10-16 13:56-14:10, C1 refuted.** `pyramidon`, code `~/code/wt-p1`
   at b81affa, 2 workers; data `data/window-recurrent/snap-wall-p1/` (the start network,
   35c36dd8...8c141a, the same as `jebel`'s; each replay's JSON; `p1.log`), manifest
   `snap-wall-p1.sha256` (726fe6cc...e8807b, 22 files, run from `data/window-recurrent`).

   | arm | proof kept both epochs | certified after epoch 1 | jumps past loss 1 (20 epochs) |
   | --- | --- | --- | --- |
   | wall off | 1/10 (replay 1) | 3/10 | 49 |
   | wall on (weight 1, ratio 1.5) | 2/10 (replays 3, 7) | 3/10 | 16 |

   - **Refuted, as registered:** wall on lost the proof in 8 of 10 (3 or more). The strict and the
     branched verdicts agree in every epoch.
   - **The baseline loses it in 9 of 10** (from the same certified network, in numpy, within 2
     epochs): the collapse is very reproducible here, a testbed for any later fix.
   - **The wall cuts the boundary jumps about 3-fold but not the loss of the proof;** the largest
     updates in the first epoch are about 0.4 in both arms (the amplification is still there).
   - **Replay 6 is the same in both arms through its first epoch** (least snap ratio 6.6: every
     visited state past 1.5, so the wall exerted nothing) and loses its proof in its second in
     both: a collapse can start where no visited state is near a boundary, which a wall on visited
     states can't anticipate. Wall on, replay 4 collapsed to 1 state (codes merging).
   - **Readings, for when this plan resumes** (each a new registered probe on this testbed; P1's
     settings are not changed): (a) the crossing starts from states training doesn't visit, or from
     the codebook moving; (b) the wall is too weak against Adam at weight 1; (c) the amplifier is the
     target: cap or clip the update, or lower the rate once a step passes. The fork (the collapse-fork
     workplan, retired) already rolls back and halves the rate, two of (c)'s remedies.
2b. **Stable-seed replay probe** (exploratory, 2026-10-16 17:06-17:15, after P1; not a resumption of
   the plan). `jebel`, code `~/code/wt-replay` pinned 7618dfe, 2 workers: P1's replay, wall off, in
   numpy, 2 epochs of step 8, generators `snap-wall-p1-0..4` (seed 6's `off-0..4` data order), from
   the step-7 networks of E2b' seeds 1, 3, 8 and 9 (certified, never lost a proof). Registered in
   the launch script's header before the data: **stable networks** if 3 or more of the 4 keep the
   proof in 4 or more of 5; **generic collapse** if 10 or more of the 20 lose it. Data
   `data/window-recurrent/stable-replay/` (the start networks, each replay's JSON, `replay.log`,
   the launch script, `margins/`), manifest `stable-replay.sha256` (5469465b...99fda6c, 31 files,
   run from `data/window-recurrent`).

   | start network (step 7) | proof kept both epochs | least snap ratio | codes in use | closest pair in use | farthest state |
   | --- | --- | --- | --- | --- | --- |
   | seed 1 | 5/5 | 107.6 | 5 | 1.875 | 0.015 |
   | seed 3 | 5/5 | 2.85 | 3 | 1.730 | 0.536 |
   | seed 8 | 5/5 | 51.7 | 5 | 1.637 | 0.031 |
   | seed 9 | 5/5 | 1.18 | 5 | 0.887 | 0.408 |
   | seed 6 (P1, wall off) | 1/10 | 4.22 | 8 | 0.904 | 0.243 |

   - **Stable networks holds; generic collapse is refuted:** all 20 replays certified in both
     verdicts after both epochs, with no jump past loss 1 (seed 6's wall-off arm: 49 in 20 epochs).
     Seed 6's collapse is its network's, not further training's in general.
   - **The registered margins don't set seed 6 apart** (the start networks' margins, `replay
     --epochs 0`, after the data): seeds 3 and 9 start nearer a boundary (least snap ratio 2.85 and
     1.18 against 4.22) and seed 9's closest codes in use are as close (0.887 against 0.904), yet
     neither loses its proof. So a wall on the margin (C1) aimed at the wrong quantity, as P1's
     replay 6 suggested.
   - **A candidate, read after the data (one network against four):** seed 6 snaps its 3 minimised
     states to 8 codes, the others to 3-5; spare codes in use beside each other could swap or merge
     under an update. A probe of it would need networks with many codes in use, registered first.

2c. **Machine-diff probe** (exploratory, 2026-10-17, after the owner's question whether the
   collapse is a property of the machine: a snapped network's machine changes only in jumps, and
   one wrong transition spreads through the state). `pyramidon`, code `~/code/wt-machdiff` (base
   91bf661, `replay --machines`: after every batch the machine read off by breadth first search, its
   (state, window) pairs changed against the last batch's, and its pairs wrong for the target),
   P1's 10 wall-off replays as run (seed 6's step 7, numpy, 2 epochs of step 8, generators
   `snap-wall-p1-0..9`). Every epoch's screen and certificate reproduce P1's. Data
   `data/window-recurrent/machine-diff/` (each replay's JSON and log), manifest
   `machine-diff.sha256` (40450dcc...3f7ad805, 20 files, run from `data/window-recurrent`).
   - **Every collapse starts as a minimal discrete event:** the first change from the certified
     machine is 1 pair in 5 replays, 2 in 3, 3 in 2, every changed pair wrong, on an ordinary update
     (norm 0.016-0.046, the epoch's median about 0.02). In 8 of 10 the batch's own loss didn't move
     (in replays 0 and 5 it jumped, 0.0012 to 3.1 and 0.0017 to 2.3): the flip is silent to the
     batch that makes it.
   - **Then the cascade is training's:** in 9 of 10 the first update above 0.1 (5 times the median)
     comes 1-13 batches after the first flip, then tens of pairs change a batch and up to 94-164
     pairs are wrong. In replay 7 the first flip was repaired without amplification (the first
     update above 0.1 only 1,512 batches later). Flips and repairs alternate: replays 1, 4, 7 and 9
     are correct in 48-79% of the batches after their first flip, the others in 0-13%.
   - **Reading:** the jump is the machine's (a snapped network changes its machine one transition
     at a time, and one wrong transition is a wrong machine), so no loss on the outputs can warn
     of it; the collapse is the optimizer's response to it, the first batch that exercises the
     wrong pair taking a large loss to Adam's small second moments. A remedy acts on the response
     (roll back, cap the update: reading (c)), not on the flip. The term "glitch" doesn't apply
     (digital design's glitch is an asynchronous transient; these machines are synchronous).
2d. **The certificate guard** (2026-10-17, after stage 2c's reading (c); the owner: "a collapse
   remedy, based on P1's saved replays"). `replay --guard`: after every batch the machine read
   off (as `--machines`), and an update that makes a right machine (no wrong pair) wrong undone,
   the network's weights, its optimizer's state and its generator restored, the batch skipped.
   `pyramidon`, code `~/code/wt-guard` (ad651d8 and the guard, as merged), P1's 10 replays as run
   (seed 6's step 7, numpy, 2 epochs of step 8, generators `snap-wall-p1-0..9`, wall off).
   **Registered** in `~/code/replay-guard.sh`'s header before the data: works if 8 or more of 10
   end certified in both verdicts. Data `data/window-recurrent/replay-guard/`, manifest
   `replay-guard.sha256` (7947cab8...d59d2646, 21 files, run from `data/window-recurrent`).
   - **It works: 10 of 10 end certified in both verdicts** (without the guard 1 of 10), each after
     both epochs, minimised to the carry automaton, screen 1.000, least snap ratio 2.3-12.5 at the
     end (4.2 at the start).
   - **It acts rarely:** 3-48 updates undone a replay, 174 of 20,480 batches (0.85%). With them
     undone Adam never amplifies: the largest update of any epoch is 0.03-0.07 (the cascades of
     stage 2c ran at 0.2-0.4), the median 0.011-0.019 as before.
   - **The loss would miss them:** only 32 of the 174 undone updates took their batch's loss up
     tenfold; the other 142 were silent to it, as stage 2c found. A guard on the loss would have
     let 82% through; it takes reading the machine.
   - **Limits:** one network's replays, an existence test at one cell; its cost is a machine read a
     batch (cheap at `k` 1: every window from every reachable state, `|Σ|^k` times the states, so
     it grows with the window). Next: the guard in real training, registered (division, where
     every run that neared a solution and trained on collapsed).
3. If C1 works: the wall in the crate (`indrajala-math-rust` first, then a bump), `Factor`'s
   settings and format 2's, study configs' field; the golden run unchanged with the wall off.
4. A registered comparison on E2b''s cell with only the wall changed: proofs lost past the first
   step, with and without.
5. Division at `p` 11 (docs/p11-trigger-workplan.md) with the wall, beside or instead of the fork's
   trigger: the trigger catches a drop; the wall would prevent its cause.

## Known limits

- The wall acts only on states training visits; a rare input's state can still sit near a boundary
  unseen. It lowers the risk; the certificate stays the proof.
- One seed's network in P1: a mechanism shown on one run, not yet a rate across runs (stage 4's).
