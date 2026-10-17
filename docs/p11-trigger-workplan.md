# Workplan: division at `p` 11, then a generic collapse trigger

**Status: written 2026-10-16 at the owner's request: "run the p11 first with the existing trigger,
then test an alternate trigger after that, to see the effect. In general we want a generic or
task-tunable trigger, we don't want to tie it too tightly to the specific task." Decisions D1-D7
settled (the owner delegates a workplan's decisions; each with its options' pros and cons).
Probe-driven ([experiment-design.md](experiment-design.md), section 2). Stage 2 (the build) done
2026-10-16; stage 3 (`p` 11) done 2026-10-17: works, 4 of 6 certified against the 121-state
machine, every one passing all 663,000 long-test cases. Stage 4 (the alternate trigger) next, or
superseded by the certificate guard.**

## Why

Two things were left open by the fork probe (the probe script's findings; the collapse-fork
workplan, retired: `git show 89cf3a3:docs/collapse-fork-workplan.md`):

- **Its registered next step: division at `p` 11** (121 states), the fork on: the largest machine
  so far would be 49 states, and next-steps.md names `p` 11 as `jebel`'s largest run.
- **The trigger.** The fork fires when the screen score (the share of every answer right on the
  step's training lengths) falls below 0.6 of the step's peak. Read against every division log we
  have (72 runs without the fork, 4,298 five-epoch intervals; 2026-10-16, the analysis in this
  plan's stage 2), it catches 16 of the 38 large drops (the worst kind of answer from >= 0.9 down by
  more than 0.3) and 8 of the 11 never recovered, with 1 false alarm. A fall more than 0.2 below the
  peak catches 27 and 11 of 11 (about 1 false alarm once the peak is at least 0.8); the mean over
  answer kinds catches all 38 (3-4 false alarms, measured on the diagnosis's cases, every 5 epochs,
  so only a proxy). The pooled screen hides a fall in one kind of answer (seed 101: the screen 1.00
  to 0.76, the kinds' mean 0.99 to 0.59).

The order is the owner's: `p` 11 first with the trigger as it is, so its result doesn't wait on, or
get confounded with, a trigger change; then the alternate trigger on the same seeds, so its effect
is read run by run.

## What "generic" means here

A trigger is three choices, none of them about a task:

- **the signal:** the screen's share of every answer right (pooled, today), the mean over answer
  kinds of each kind's share, or the worst kind's share. The kinds are whatever the task's
  `accuracy` tallies: division's quotient classes and queries, a modular sum's one answer, the
  harness's properties. A task with one kind of answer has all three equal.
- **the rule:** below a fraction of the step's peak (`ratio`, today: 0.6) or more than an amount
  below it (`fall`).
- **the floor:** fire only once the step's peak is at least this (today: none), so a run still
  learning its step doesn't fork on noise.

Each is a setting with a default that reproduces today's trigger exactly; a task tunes them, never
the code. The trigger lives in its own module (`indrajala_ml/studies/collapse.py`), so the
window-recurrent harness can take it up (next-steps.md: the fork in the harness) without a second
copy.

## Decisions

- **D1. The order: `p` 11 with today's trigger, then the alternate on the same seeds (the owner).**
  - (a) *Chosen.* Pros: `p` 11's existence result stands on the known mechanism; the alternate is
    then a paired comparison (runs are deterministic: the two arms are the same run until the first
    epoch where the triggers disagree). Cons: the alternate's effect is read at one task size first.
  - (b) The alternate first. Cons: `p` 11 waits on an untested change and can't tell which part
    worked.
- **D2. `p` 11's cell: the family (the cover: quotient, sum and difference), two factors of 32
  codes (`s` 16), reading the output, the fork on with today's trigger, seeds 0-5, 600 epochs a run
  (`--budget-epochs`), stop at certified; 64 a factor only if 32 certifies none (the ladder).**
  - (a) *Chosen.* Pros: the cell that certified 5 of 8 at `p` 7 (the fork probe), only `p` changed;
    32 a factor is 2.9 times a counter's 11 and 1,024 joint codes 8.5 times the 121 states (the
    surplus rule); epochs, not minutes, so the machine's load doesn't matter; 6 seeds fill
    `jebel`'s 6 workers. Cons: 600 epochs is a guess (`p` 7's successes took 149-382); a screen,
    not a rate.
  - (b) 64 a factor from the start. Pros: more surplus (5.8 times a counter). Cons: four times the
    joint codes and the certificate's cost; the `p` 7 cell is what worked (existence first).
  - (c) Without-fork arm too. Cons: the fork probe compared them; doubles the cost; not the
    question.
- **D3. What `p` 11 needs built: `p` 11 in `modular_data.PRIMES`** (3 is invertible mod 11, so the
  task's facts hold), its tests, and the certificate checked on it (the existing parametrised tests
  take `PRIMES`). Pros: no new code path. Cons: none known; the build stage finds out.
- **D4. Instrumentation before `p` 11 runs: each epoch's screen per answer kind in the log**, beside
  the pooled score (one dictionary a row; the tallies are already counted). Pros: `p` 11's logs can
  then score every candidate trigger offline at every epoch, not at every fifth on the diagnosis's
  cases; training unchanged (checked as #779 was: a 3-step run byte-identical before and after).
  Cons: larger logs (a few kB a step).
- **D5. The trigger as settings in a shared module (`collapse.Trigger`: `signal`, `rule`,
  `threshold`, `floor`), today's as the default; `modular_family.Fork` takes one.**
  - (a) *Chosen.* Pros: generic (no task in it), tunable per task from the command line or a
    config, one place for the harness to take it from; the default bit-identical. Cons: a small
    refactor of the fork's code.
  - (b) A new flag per alternative. Cons: grows a flag per idea; ties the code to the cases tried.
- **D6. The durable scoring tool: `modular_family_probe.py drops ROOT... --trigger ...`**: every
  logged run's large drops, never-recovered drops and false alarms for a given trigger (this plan's
  analysis, made a tool). Pros: the alternate is chosen by a rerunnable measurement, and any later
  task's logs are scored the same way. Cons: a scorer reads what was logged (five-epoch diagnoses
  before D4).
- **D7. The alternate trigger is chosen after `p` 11 and before its replays, by D6 over every
  log (`p` 5, 7 and 11), and registered in this plan before the replays run.** The candidate from
  today's logs: the mean over answer kinds more than 0.2 below the step's peak, floor 0.8;
  the fallback if `p` 11's per-epoch logs don't support it: the pooled screen more than 0.2
  below, floor 0.8. **The replays:** `p` 11's six seeds and the fork probe's four `p` 7 seeds,
  the alternate in place of today's, everything else equal. Pros: two task sizes, each paired with
  a run already made; a choice made on the most data, still before the replays. Cons: chosen on
  logs that include `p` 11's training runs (not its replays), so the scorer's numbers are
  in-sample; the replays are the out-of-sample test.

## Feasibility (experiment-design.md, section 1)

| cell | target states | the model's states | per factor | exclusion bound `m + q - 1` | cap |
| --- | --- | --- | --- | --- | --- |
| family, `p` 11, two factors of 32 | 121 | 1,024 joint | 32 (a counter 11) | 1,144 | 1,144 |
| family, `p` 11, two factors of 64 (the ladder's second rung) | 121 | 4,096 joint | 64 | 4,216 | 4,216 |
| replays, `p` 7, two factors of 32 | 49 | 1,024 joint | 32 (a counter 7) | 1,072 | 1,072 |

No run reaches its cap in 600 epochs; a probe claims nothing about memorising, and every success is
certified, which is exact at every length.

## Stages

1. **This plan.**
2. **The build:** `p` 11 in `PRIMES` (D3); per-kind screens in the log (D4); `collapse.Trigger` and
   the fork taking it, today's as the default (D5); the `drops` scorer (D6). One PR: the golden run
   and a 3-step probe run byte-identical before and after (the trigger's default unchanged).
3. **`p` 11, today's trigger** (D2), registered before its data. **Works:** at least one network
   certifies against the 121-state machine in both verdicts. **Fails:** none at 32 a factor, then
   none at 64. Reports: each run's certified step, epochs, forks, large drops, its trace. Every
   certified network recertified in both verdicts and long-tested (division's U1 and S1 at 128,
   256 and 1,024). Where: `jebel` (6 workers), after E2b'.
   **Done 2026-10-16:** `p` 11 in `PRIMES` (the certificate's parametrised tests pass at 11);
   each screen row logs `kinds`, every answer kind's share; `collapse.Trigger` and
   `Fork(trigger=...)` (`--trigger-signal`, `-rule`, `-threshold`, `-floor`); `drops`. Checked: a
   3-step fork-on run at `p` 5 and the fork test's always-firing run (6 forks) give the same logs but
   for `seconds` and the new `kinds`, and byte-identical networks, before and after. **The scorer's
   definitions** (the Why's figures came from an inline scoring not kept, with other definitions,
   and this tool's supersede them): a large drop as the Why's; caught when the trigger fires at the
   drop's point (its step's peak over the points before); unrecovered when no later point is back
   within 0.05 of the level before; a false alarm a firing near a solution (the worst kind at 0.9
   before) at no large drop; an early firing one while still learning. Over the same 72 runs (every
   log but the fork arm's; at diagnoses, a proxy): 38 large drops, 14 never recovered.

   | trigger | caught | never recovered caught | false alarms | early |
   | --- | --- | --- | --- | --- |
   | pooled, below 0.6 of the peak (today's) | 7 | 4 of 14 | 0 | 67 |
   | pooled, 0.2 below | 16 | 6 | 1 | 104 |
   | pooled, 0.2 below, floor 0.8 | 16 | 6 | 1 | 49 |
   | mean of kinds, 0.2 below, floor 0.8 (D7's candidate) | 25 | 7 | 5 | 48 |
   | worst kind, 0.3 below, floor 0.9 | 25 | 7 | 0 | 12 |

   Descriptive only: D7 chooses after `p` 11, on every log, its per-epoch kinds included.

   **Stage 3 run 2026-10-16 (jebel, 7618dfe, 21:38 to 23:38): works.** 4 of 6 certified at two
   factors of 32 (seeds 1, 2, 4, 5 at steps 10, 16, 31, 16; 293-543 epochs; 1-72 forks), so the
   ladder's second rung doesn't run. Every one recertified in both verdicts (minimised to 121,
   210-448 codes reached) and passing all 663,000 long-test cases (pyramidon, c4a91e7). No run
   collapsed; 4 large drops, all recovered; the two failures (seeds 0 and 3) never came near a
   solution (worst answer at best 0.75, 0.41). The findings, the table and the manifests are in
   `scripts/modular_family_probe.py`'s docstring; the networks in indrajala-models.
4. **The alternate trigger** (D7): chosen with `drops`, registered here, then the replays (`p` 11's
   seeds and the fork probe's). Reported paired: per seed, both arms' certified step or failure,
   epochs, forks and unrecovered drops. **Better** if the alternate certifies more seeds, or as
   many with fewer epochs, with no seed it loses that today's trigger certified.
5. **Findings,** the paper's notes (P2: `p` 11), and the default trigger decided from stage 4;
   this plan retired, the harness's fork to next-steps.md.

Tier 0 throughout (the study module and its script, pure Python; no timed path), unless the
build touches `classify_rows` (it won't: the tallies exist).
