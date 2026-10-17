# Workplan: the certificate guard in real training (division at `p` 11)

**Status: written 2026-10-17, the next step the snap-wall workplan's stage 2d registered ("the
guard in real training, registered (division): the fork alone against the fork plus the guard, at
the same seeds"). D1-D8 settled under the owner's delegation, each with its options' pros and
cons. G1 registered here, before its data.**

## Why

The guard (`replay --guard`, the snap-wall workplan's stage 2d) reads the network's machine after
every batch and undoes an update that makes a right machine (no wrong (state, window) pair) wrong.
Replaying one certified network's collapse, it kept the proof in 10 of 10 replays against 1 of 10.
That is one network; whether it matters in training is this plan's question.

**Division's runs do lose right machines.** The stage-3 `p` 11 ladder's saved step networks (the
p11-trigger workplan, retired: `git show 28894cf:docs/p11-trigger-workplan.md`), certified
strictly (snap ratio 1.0) against the 121-state machine, 2026-10-17:

- seed 4: right at the ends of steps 17-21, 23-24, 26 and 30, wrong at 22 (123 states), 25
  (137), 27-29 (147, 134, 275); certified in both verdicts only at step 31;
- seed 2: right at step 13, wrong at 14 (238 states) and 15, certified at 16;
- seeds 1 and 5 first right at the step they certified; seeds 0 and 3 never right.

The run stops only on the branched verdict (near ties, snap ratio under 1.05, branched both
ways), so a strictly right machine with near ties trains on into the next step, and there it was
lost again and again: seed 4 spent 14 steps, about 100 epochs, from its first right machine to its
certificate. That is the guard's case: hold the right machine while training resolves its near
ties. The fork can't be relied on for it: it restores the step's best network on a large fall of
the screen, and a machine one or two pairs wrong (seed 4's step 22, 123 states) barely moves it.

## Decisions

- **D1. The cell: stage 3's, `p` 11** (the family, two factors of 32 codes, `s` 16, reading the
  output, the fork with today's trigger, 600 epochs a run, stop at certified).
  - (a) *Chosen.* Pros: the only cell with evidence of right machines lost (above); stage 3's
    settings, so nothing else changes. Cons: the longest runs we have (stage 3: 1-2 h each on
    `jebel`); the crate's `exp` changed since (aarch64 stage 2), so today's runs aren't stage 3's
    and may not repeat its losses.
  - (b) `p` 7, the fork probe's cell. Pros: about half the cost. Cons: its runs saved no step
    networks, so no evidence the guard would act there.
  - (c) `p` 5, 20 seeds a rate. Pros: cheap, a rate. Cons: no evidence either; a rate before
    existence (experiment-design.md, section 2).
- **D2. The arms: the fork alone, and the fork plus the guard (the owner's).** The guard alone,
  a third arm, isn't run. Pros: the question as set; runs are deterministic and the guard changes
  nothing until it first undoes an update, so each seed is a pair, the same run up to that batch.
  Cons: the guard's effect without the fork isn't seen (later, if the guard works).
- **D3. Seeds 0-5, both arms; registered extension: if the guard acts on fewer than 2 of them,
  seeds 6-11 in both arms.** Pros: stage 3's seeds fill `jebel`'s 6 workers; the extension is
  fixed now, before the data, so it isn't a setting changed after it. Cons: a screen, 6-12 pairs,
  no rate.
- **D4. What the guard reads: the quotient (slot 0) against division's machine**, the machine the
  run certifies. Pros: the guard protects exactly what the run is judged by. Cons: the sum and
  difference can still be lost under a held quotient (they are the cover, not the claim).
- **D5. When it reads: armed for an epoch whose screen before it (the step's start, or the last
  epoch's) has every quotient kind at 1.0; armed, the machine read before the epoch's first batch
  and after every batch.**
  - (a) *Chosen.* A right machine answers every quotient right, so an epoch after an imperfect
    quotient screen didn't start right. Pros: every loss seen above was of a machine right at a
    step's end (screen 1.0), all caught; unarmed epochs train exactly as without the guard, at no
    cost. Cons: a machine that becomes right and is lost within one epoch, unarmed, isn't held;
    the guard is armed only from its next perfect screen.
  - (b) Every batch. Pros: the guard of stage 2d exactly. Cons: a read is a forward pass over every
    reachable state's 9 windows, 21 ms for a machine far from right (its first wrong pair early)
    and 130 ms for a right one (measured on stage 3's networks), against 25 ms a batch's update:
    about twice the cost of every run, mostly where nothing is right.
  - (c) A forward pass on each batch first (a right machine gets every quotient of the batch
    right). Pros: exact. Cons: about a third more on every batch, for the case (a) misses.
- **D6. The read: breadth first by levels, every state of a level stepped in one forward pass,
  stopping at the first wrong pair.** Pros: 2.5 times faster than a call a state (130 against
  330 ms at 456 states); a wrong machine costs its levels up to its first wrong pair. Whether a
  machine is right doesn't depend on the order states are reached in. Cons: a second reader beside
  `window_replay.machine_table`, which keeps its call a state (its replays' records were read that
  way; a batched pass may round differently in numpy's products).
- **D7. Where it lives: `indrajala_ml/studies/certificate_guard.py`** (the read, the guard, the
  training state it saves and restores), `window_recurrent_study.train_epoch` taking an optional
  guard (so the harness can use it, as next-steps.md plans for the fork), `modular_family.run`
  taking `guard=True` (`--guard` in `modular_family_probe.py`). Each epoch row logs `rollbacks`
  and `armed` with the guard on. Pros: one guard for the family and the harness; the probe's
  training unchanged with it off. Cons: a small refactor of the replay's training-state helpers
  into the new module.
- **D8. The fork and the guard stay independent:** the guard within an epoch's batches, the fork
  between epochs. A fork restores its snapshot and the guard reads it afresh. Pros: each as
  tested; the owner's "the fork stays". Cons: a fork can restore a network that isn't right
  (its snapshot is the best screen, not the right machine); counted, not prevented.

## The claim (registered before the data)

**G1, the guard helps where it acts.** On each seed where the guard undoes at least one update,
compare the arms: **better** if the guard arm certifies (both verdicts) and the fork alone doesn't,
or both certify and the guard arm in fewer epochs; **worse** the reverse. **Works:** better on at
least one such seed and worse on none. **Refuted:** worse on any. **Undecided:** neither (equal
everywhere). If it acts on no seed after D3's extension, G1 is untested.

Checked, not claimed: on a seed where the guard never acts, the two arms' logs are identical but
for timings and the guard's fields (determinism; a failure there is a bug, found and recorded).

**Reported per seed, both arms:** certified or not, step, epochs, forks, large drops; the guard
arm's armed epochs, rollbacks, its first rollback (step, epoch, batch), and the strict verdict at
each step's end. Every certified network recertified in both verdicts and long-tested (U1 and S1
at 128, 256 and 1,024). Statistics: none beyond the pairs (experiment-design.md, section 3: a
screen).

## Feasibility (experiment-design.md, section 1)

| cell | target states | the model's states | per factor | exclusion bound `m + q - 1` | cap |
| --- | --- | --- | --- | --- | --- |
| family, `p` 11, two factors of 32 | 121 | 1,024 joint | 32 (a counter 11) | 1,144 | 1,144 |

Stage 3's cell unchanged: 4 of its 6 seeds certified within 600 epochs.

## Stages

1. **This plan.**
2. **The build** (D5-D7), one PR, tier 0 (the study code; no timed path): the guard module, its
   tests (the read agrees with `machine_table` on saved networks, right and wrong; an update that
   breaks a right machine is undone with weights, optimizer and generator restored); with the
   guard off, the golden run and a 3-step `p` 5 fork-on run byte-identical before and after.
   **The budget probe:** an armed epoch's time on `jebel` from stage 3's seed-4 step-17 network,
   so the study's estimate is a range before launch.
3. **The run:** `jebel`, 6 workers, the 12 runs, seeds paired; then D3's extension if it applies.
4. **Findings** in `modular_family_probe.py`'s docstring, the snap-wall workplan's stage 2d and
   the papers' notes (P1: the guard); certified networks to indrajala-models; this plan retired.
