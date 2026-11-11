# Workplan: one A/B guards the window program (learn, forward and the long test)

**Opened 2026-11-11.** Infrastructure item 1 of [next-steps.md](../../next-steps.md): "the window
program's timing guard covers learn and forward". It gates every later window-program crate change
(items 40, 41's P4, 44, 45, 46), so it comes first.

## The gap

A crate change to the window program has two timing scripts, and one A/B covers only part of it:

| script | where | its default cases | what they cover |
|---|---|---|---|
| `window_recurrent_timing.py` | indrajala-ml, `scripts/measurement/` | six: `k1-s4-m16-B32-W12`, `k4-s4-m16-B32-W5`, `learn-linear-k1-s16-m16-B64-W19`, `learn-family-p7-f32-B64-W16`, `forward-linear-k1-s16-m16-n1024`, `forward-family-p7-f32-n1024` | the training step (forward and backward) and the long tests' forward pass |
| `window_study_timing.py` | indrajala-studies, `scripts/measurement/` | one: `long-test-family-p7-f32-U1-n1024-c1023` | the long test end to end (forward only) |

The studies script reuses the ml script's parsing, timing and breakdown, and it already runs the
learn and forward cases when they are named (`case_step` falls through to `timing.case_step`). Only
its default leaves them out.

**What it cost.** #1212 (the FiLM state input) needed two A/Bs on `jebel`. One ran the studies
script: [indrajala-benchmarks
`runs/jebel/2026-11-03-film-state-input-cmd`](https://github.com/davidbarkhuizen/indrajala-benchmarks/tree/main/runs/jebel/2026-11-03-film-state-input-cmd),
ONNO, about 65 s a pass. The other ran the ml script: `2026-11-03-film-state-input-window`, ONNO then
NOON, about 33 s a pass. Both had the same commits and crates (`857aff8` against `bb05ca1`).

**Other places with the same gap:**
- `crate_perf/ab_all.sh`'s window guard on stone and klip (`WINDOW`) runs the ml script alone, so
  the guards never time the long test.
- [measurement.md](../../measurement.md) §1's tier-1 table has no row for the window program.
- The working-notes entry ("A crate change to the window program needs ml's
  `window_recurrent_timing.py` cases too") is the rule that stands in for the missing default.

### The control that moved

In the studies script's A/B, the control `matmul 256` came out consistent, -22.9%. That marked the
run "CONTROL MOVED" (changes about that size can't be resolved). In the ml script's A/B the same
crates' control was within noise (+0.2%). Same crates, two scripts, so the cause was the run's
context, not the crate. The control's values, µs a step, five repeats each:

| run | old passes | new passes |
|---|---|---|
| studies script (after the long test) | 593 525 488 474 481; 573 531 512 520 532 | 475 447 376 378 383; 490 422 399 406 391 |
| ml script (after the forward cases) | 591 501 498 498 497; 607 496 493 504 492; ... | 606 500 524 497 501; 636 491 489 504 496; ... |

Two things show:
1. **The first repeat is slow in every pass of both runs,** by 10-25%. The control warms up for 20
   loops, about 10 ms: too short for a threaded product (its pool, the clock).
2. **The studies run's sides sit at different levels throughout.** The control runs straight after a
   10.7-s single-threaded long test there.

**A development probe on `pyramidon`** (2026-11-11, not a timing machine) timed the control in one
process, cold and then straight after one long-test step (11.3 s), µs a step, five repeats:

| when | 20 loops (today) | 200 loops |
|---|---|---|
| cold | 474 475 462 480 532 | 551 514 720 570 412 |
| after the long test | **833 823 672 709 591** | 513 522 658 494 565 |

At 20 loops, the control after a long case measures the case before it, decaying over its five
repeats. At 200 loops it is back in the cold range. **A candidate explanation of #1212's -22.9%,
not a claim:** `jebel` is out of service, and the old trees of #1212 no longer run with today's
studies script, so that run can't be repeated. Stage 3 checks the fix on `stone`.

## Decisions

I settled these myself (the owner delegates workplan decisions). Each lists every option it
considered.

### D1: how one A/B gets every case

- **(a) The studies script's default is ml's `DEFAULT_CASES` plus the long test** (imported, not
  copied). **Chosen.**
  - Pros:
    - One line.
    - It follows ml's list as cases are added.
    - It is already the script ab.py runs for the window program.
  - Cons:
    - The timing module is each side's ml tree's, so an A/B across a change to ml's list measures
      some rows on one side only. `ab.py` reports those ("N rows measured on one side only"), and
      a timed row needs both sides anyway.
- **(b) A literal list of all seven cases in the studies script.**
  - Pros:
    - The case set is the same on both sides whatever ml's tree holds.
  - Cons:
    - It copies ml's list, which drifts.
    - A case whose parser exists only on one side fails there anyway, so the sameness buys little.
- **(c) `ab.py` runs two scripts in one pass.**
  - Pros:
    - Neither script changes.
  - Cons:
    - A new ab.py feature, with its contract, for a case already solved by (a).
    - Two process trees a pass.
- **(d) The long test moves into ml's script.**
  - Pros:
    - None that holds.
  - Cons:
    - It runs the studies' own code (`modular_family`, `window_tasks`), which the research
      separation (its D3) took out of ml.

**The order of the cases:** ml's six first, the long test last. The learn rows then run before any
heavy case, as they do in ml's script. The control after the long test is D2's problem.

### D2: the control's warm-up

- **(a) Leave it.**
  - Pros:
    - No change to any row.
  - Cons:
    - The combined default puts the control after the long test, where the probe shows it
      misreads. Stage 1 alone would make a moved control likely in every window A/B.
- **(b) The control alone: 200 loops, and a warm-up of the same.**
  - Pros:
    - The smallest change that the probe supports.
  - Cons:
    - A special case: the learn cases keep a 20-loop warm-up (60 ms to 0.4 s, long enough there),
      and the next short-step case would need the same care again.
- **(c) One rule for every step: warm up for at least 0.5 s** (`loops` steps, then more until
  0.5 s has passed), and time the control at 200 loops. **Chosen.**
  - Pros:
    - One rule covers the control and any later short step.
    - Every case except the control is already past 0.5 s (a learn case's 20 steps and a
      forward case's one step take 60 ms to 3.4 s), so their timed loops don't change. The learn
      cases' warm-up grows only where it is under 0.5 s.
  - Cons:
    - Each pass lasts a few seconds longer.
    - In an A/B whose old side predates it under the studies script, the warm-ups differ by side,
      because the timing module comes from each side's ml tree. That holds only for A/Bs that
      straddle this PR.
- **(d) Time the control twice, first and last** (two control rows).
  - Pros:
    - Every run would show a context effect directly.
  - Cons:
    - A second row to read.
    - The probe already shows the effect, and (c) removes its cause.
  - **Not now.** If stage 3's A/A still moves the control, (d) is the next probe.
- **(e) A single-threaded control.**
  - Pros:
    - Threading changes would no longer move it (working-notes).
  - Cons:
    - A different control from every archived window run.
    - The context effect is not a threading effect.
  - **Not now.** Kept with the working-notes entry.

### D3: next-steps' numbering when item 1 goes

- **(a) Delete item 1 and its edge, renumber nothing.** **Chosen.**
  - The list then starts at 2, which Markdown renders as 2.
  - Drop "needs 1" from 40, 41, 44, 45 and 46.
  - Pros:
    - Every reference to an item number (the edges, Part 2's "infra 27, 35, 49", handovers)
      stays true.
  - Cons:
    - None while the deleted items are at the top. A deletion in the middle would raise this again.
- **(b) Renumber.**
  - Pros:
    - A tidy list.
  - Cons:
    - It rewrites every reference in the file and breaks those outside it.

### D4: where it is checked

`jebel` is out of service, and nothing here claims a speed, so this needs no claim machine.

- **(a) `stone`, with `ab.py run` on the machine** (working-notes: "`ab.py` on `stone`", by
  bundles), **and `klip` for the guard's own path.** **Chosen.**
  - Pros:
    - Both are free after the SMB runs (about 12:30).
    - `stone` has a profile (`i7-1255u.json`).
    - `klip` is where `ab_all.sh`'s guard runs, under `ulimit -v` and the thermal guard.
  - Cons:
    - Neither is `jebel`, so the -22.9% itself isn't re-run (above).
- **(b) Wait for `jebel`.**
  - Pros:
    - It is the machine that showed the move.
  - Cons:
    - No date for its return.
    - It would block items 40-46's guard on a machine they don't need for a tier-1 A/B.

## Stages

Each stage is one PR, merged before the next.

### Stage 1: the combined default (indrajala-studies)

- `window_study_timing.py`: `DEFAULT_CASES = timing.DEFAULT_CASES + ["long-test-family-p7-f32-U1-n1024-c1023"]` (D1).
- Its docstring: the default is every learn and forward case and the long test, the one A/B for a
  window-program change.
- `tests/scripts/test_window_study_timing.py`: the default is ml's cases followed by the long test.
- The README's Timing paragraph says the same.
- **Tier 0:** a measurement script, no timed path. Tests and lint on `pyramidon`.

### Stage 2: the control and the rule (indrajala-ml)

- `window_recurrent_timing.py`: `timed` warms up for `loops` steps and on until 0.5 s has passed
  (D2 (c)). The control is timed at 200 loops a repeat. A test checks both on a stub step, with
  nothing timed against a machine.
- `crate_perf/ab_all.sh`: `WINDOW` runs the studies script (each host's checkout path: `stone`
  `~/labour/code/indrajala-studies`, `klip` `~/code/indrajala-studies`). Bundle the studies repo
  to `stone` beside ml's.
- `measurement.md` §1, the tier-1 table: a row "the window program (its op, the window-recurrent
  networks' learn and classify paths, or their crate ops)" → indrajala-studies'
  `window_study_timing.py` through `--bench cmd`.
- working-notes: the entry "A crate change to the window program needs ml's ... too" becomes
  "one A/B: the studies script's default covers learn, forward and the long test". Add a new note:
  a control timed straight after a heavy case misreads without a warm-up of its own (this plan's
  probe).
- next-steps: item 1 and its edge deleted, "needs 1" dropped from 40, 41, 44, 45 and 46 (D3).
- **Tier 0.**

### Stage 3: the check, on `stone` and `klip` (after the SMB runs end, about 12:30)

1. **A smoke run on `stone`:** the studies script at its new default, once, outside `ab.py`. It
   checks that all seven cases and the control run, and gives a pass's length.
2. **An A/A on `stone`:** `ab.py run --old main --new main --order ONNO --bench cmd --
   ~/labour/code/indrajala-studies/scripts/measurement/window_study_timing.py`.
   - **Passes if** the control is within noise and no row is consistent beyond the profile's noise
     rules.
   - If the control still moves, try D2 (d) (a first-and-last control) as a probe before anything
     else.
3. **The same A/A on `klip`,** under `ulimit -v 4500000` and the thermal guard (85 °C), as
   `ab_all.sh` runs it. Bring klip's studies checkout up to date first (https; its clone is at
   `db8db90`, 2026-10-31).
4. **Record** the passes' lengths and both reports in this plan.
5. **Retire it** in the same PR (one row in [retired.md](../retired.md); its row out of
   [README.md](../README.md)).

## Status

- Workplan: this PR.
- Stage 1: not started.
- Stage 2: not started.
- Stage 3: waits for `stone` and `klip` (about 12:30).
