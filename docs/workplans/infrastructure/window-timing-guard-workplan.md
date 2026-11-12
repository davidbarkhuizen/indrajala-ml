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

### D2: what to do about the control (decided at stage 3, from data)

**The first choice, (c) below, was withdrawn on 2026-11-11 before it merged.** The probe that
supported it read the control at 200 loops after it had already run at 20, so it was already warm.
Re-run with the warm-up as planned (0.5 s, then 200 loops), the control after the long test read
750-905 µs against a cold 410-570. Then `--settle` (stage 2) timed it a second at a time on
`pyramidon`, µs a step:

| run | cold | after one long-test step |
|---|---|---|
| 1 (8 s) | 509 503 509 513 513 508 511 514 525 | 805 783 782 775 772 **529** 513 499 552 |
| 1, a second long test | | 814 782 808 771 808 827 784 741 **516** |
| 2 (10 s) | 521 520 516 527 551 565 523 523 536 559 | 808 790 771 765 792 768 765 786 786 772 |

After the long test, the control runs 45-60% slow at a flat level. It drops back to the cold level
all at once, after 5-7 s in one run and not within 10 s in another.
- **Not OpenBLAS's spin:** that lasts milliseconds (measurement.md §7).
- **The candidate on a laptop:** its clock or power budget, after 12-15 s of one boosted core.
- **It doesn't explain #1212 on `jebel`:** there the side straight after the long test was faster,
  not slower.

A warm-up of any fixed length would be a guess at a machine's recovery time. The options, with
stone's and klip's readings to choose between them:

- **(a) Leave the control as it is.**
  - Pros:
    - No change.
  - Cons:
    - In the combined default it runs straight after the long test, where it misreads on
      `pyramidon`. A "CONTROL MOVED" there would say nothing about the crate.
- **(b) Time the control first, before any case.**
  - Pros:
    - It reads the machine before this pass's heavy work.
  - Cons:
    - It follows the previous pass's long test, only a process start (a few seconds of imports)
      later.
    - It sees nothing that changes during the pass.
- **(c) A warm-up of at least N seconds for every step.**
  - Pros:
    - One rule.
  - Cons:
    - N is a guess unless the runners' recovery times are measured; on `pyramidon` it would have
      to exceed 10 s.
    - Withdrawn above.
- **(d) Time the control first and last** (two control rows).
  - Pros:
    - Every run shows a context effect directly, whatever its cause.
  - Cons:
    - A second row to read.
    - The last row still misreads where (a) does. It flags the problem, not cures it.
- **(e) Order the cases so the long test runs first,** then the learn and forward cases, then
  the control.
  - Pros:
    - Several seconds of lighter work separate the long test from the control.
  - Cons:
    - The forward cases are heavy too, and the gap is shorter than what `pyramidon` needs.
- **(f) A single-threaded control.**
  - Pros:
    - Threading changes no longer move it (working-notes).
  - Cons:
    - A different control from every archived window run.
    - Not shown to avoid this effect: a clock state touches one core as much as eight.

**Decided at stage 3 (2026-11-12): (a).** Neither runner shows the effect: the control straight after the long
test reads as it does cold, every second of 30, on both (stage 3's record below). The effect is recorded as
`pyramidon`'s, as the rule below says.

**How it is decided:** `--settle long-test-family-p7-f32-U1-n1024-c1023 --settle-seconds 30` on
`stone` and on `klip` (stage 3, step 1).
- The control recovers within a few seconds there, and the A/A's control is within noise: **(a)**,
  and the effect is recorded as `pyramidon`'s.
- It stays off for longer: **(b) with (d)'s last row,** a first and a last control. This is the
  smallest change that both separates the control from the long test and still shows a drift
  within a pass.

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

### Stage 2: the guard's docs and the settle probe (indrajala-ml)

- `window_recurrent_timing.py --settle CASE [--settle-seconds S]`: the control's µs a step, a
  second at a time, cold and then straight after one step of the case. It times no rows, and the
  studies script has it too (it hands `main` its own `case_step`). This is the durable form of
  D2's probe. Tests run it on a small case, with nothing timed against a machine.
- `crate_perf/ab_all.sh`: `WINDOW` runs the studies script, from each host's checkout:
  - `jebel` and `klip`: `~/code/indrajala-studies`, pulled first.
  - `stone`: `~/labour/code/studies-main`, cloned or fast-forwarded from a bundle sent beside ml's.
    stone's `~/labour/code/indrajala-studies` is an empty repository with 79 MB of objects, touched
    on 2026-11-11, so it is left alone.
- `measurement.md` §1, the tier-1 table: a row "the window program (its op, the window-recurrent
  networks' learn and classify paths, or their crate ops)" → indrajala-studies'
  `window_study_timing.py` through `--bench cmd`.
- working-notes: the entry "A crate change to the window program needs ml's ... too" becomes
  "one A/B: the studies script's default covers learn, forward and the long test". Add a new note:
  on `pyramidon` the control misreads for 5 to more than 10 s after the long test (D2).
- next-steps: item 1 and its edge deleted, "needs 1" dropped from 40, 41, 44, 45 and 46 (D3).
- **Tier 0.**

### Stage 3: the check, on `stone` and `klip` (after the SMB runs end, about 12:30)

1. **On `stone` and `klip`, outside `ab.py`:**
   - `--settle long-test-family-p7-f32-U1-n1024-c1023 --settle-seconds 30`. This decides D2.
   - The studies script at its new default, once. It checks that all seven cases and the control
     run, and gives a pass's length.
   - If D2 calls for a change to the control, it lands (ml) before the A/As.
2. **An A/A on `stone`:** `ab.py run --old main --new main --order ONNO --bench cmd --
   /home/david/labour/code/studies-main/scripts/measurement/window_study_timing.py`.
   - **Passes if** the control is within noise and no row is consistent beyond the profile's noise
     rules.
   - If the control still moves, D2 is read again with this run's data.
3. **The same A/A on `klip`,** under `ulimit -v 4500000` and the thermal guard (85 °C), as
   `ab_all.sh` runs it. Bring klip's studies checkout up to date first (https; its clone is at
   `db8db90`, 2026-10-31).
4. **Record** the passes' lengths and both reports in this plan.
5. **Retire it** in the same PR (one row in [retired.md](../retired.md); its row out of
   [README.md](../README.md)).

## Status

- Workplan: this PR.
- Stage 1: done (indrajala-studies #21).
- Stage 2: done (#1300).
- Stage 3: done 2026-11-12; the record below. D2: (a).

### Stage 3's record (2026-11-12)

**`klip`** (ml `de66e13`, its crate rebuilt at `rust/`'s `2c4415b` first, since the pulled tree's Python outran the
built crate; thermal guard at 85 C, `ulimit -v 4500000`):
- **Settle,** µs a step: cold 569-571 for 30 s; after one long-test step (18.4 s), 567-571 for 30 s. No effect.
- **The default pass,** once: every case and the control ran; 167 s. The long test was 18.5 s a step and the
  control 577 µs. Peak 53 C.
- **The A/A** (`2026-11-12-guard3-aa-klip`, ONNO, 17 min, peak 41 C): the profile's identity matches
  (`ryzen7-3700u-klip.json`); the control is within noise; 7 rows within noise, max |Δ| 2.3%. **Passes.**

**`stone`** (ml `de66e13` by bundle, its crate rebuilt at `2c4415b` for the same reason; the studies at
`e3f010e` in `~/labour/code/studies-main`):
- **Settle,** µs a step: cold 713-818 for 30 s; after one long-test step (10.8 s), 705-836. No effect, though
  noisier than `klip`'s.
- **The default pass,** once: every case and the control ran; 101 s. The long test was 11.0 s a step and the
  control 602 µs.
- **The A/A:** refused first. `stone`'s identity no longer matches its profile: its kernel moved from
  7.0.0-38 to 7.0.0-42, and its energy-performance preference from `balance_performance` to `balance_power`.
  It is the owner's desktop, so that may be the owner's setting. This A/A claims no speed, so it ran with
  `--skip-profile` (`2026-11-12-guard3-aa-stone2`, ONNO, 6 min, Chrome open): the control is within noise; 7 rows
  within noise (1 separated inside its spread), max |Δ| 5.6%. **Passes.**
- **Left open for `stone`:** a new profile (tier 3, measurement.md §1) before its next timed claim, or the
  owner's power setting restored. In next-steps.
