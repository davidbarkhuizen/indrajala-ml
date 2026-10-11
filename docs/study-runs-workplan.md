# Workplan: studies and timing runs sharing `jebel`

**Status: decisions D1-D9 settled (2026-10-11; the owner: "going with your recommendation on
each decision point"). Stage 1 (this plan) in review.**

Studies run on `jebel` as well as A/Bs, and nothing makes the two share it safely. Neither of
the study harnesses records enough to plan around a study or reuse its results. The addition
study's sweep (stage 5, launched 2026-10-11 04:16) showed five gaps:

1. **Nothing stops an A/B from starting during a study.** Studies don't take `ab.py`'s run lock
   (`RUNS_ROOT/run.lock`), and `ab.py`'s pre-flight records the load average but only flags it
   (`high_load`, 1.5 on `jebel`). Worse, `ab.py remote run` runs `sync_checkout` first. That
   fast-forwards `jebel`'s checkout to `origin/main` and rebuilds the venv's crate when `rust/`
   moved, which is the same checkout and crate a running study imports from. Only the handover
   note's "nothing else on jebel" prevents it.
2. **A short A/B waits hours behind a long study.** #669's tier 1 A/B, well under an hour, waits
   about 5 hours for the addition sweep. The harness resumes, but stopping it means killing six
   runs in flight.
3. **A study's end can't be estimated.** `addition_study.py status` prints counts only. The
   handover's 2.5 runs a minute was the sweep's average from its start. The same hour's rate was
   1.6 (06:18-07:06, 77 runs), so the forecast end moved from 08:15 to about 09:30.
4. **`train_seconds` mixes training with checking.** On a passing run it includes the whole
   catalogue (16 s at `n = 4` for a column transformer, 213 s for the string format), so seconds
   an epoch look 4-8 times the true 2.5-3 s. D7's cost estimates were read from it.
5. **Studies' raw results are lost.** Each study's `--out` is optional and goes wherever it's
   pointed. No raw JSON of the sequence study's sweep survives, so the rotary study (step 8,
   stage 8) was planned against recorded means ± sd with no per-seed differences. The
   attention-dropout sweep survives only by chance, as `~/attention-dropout-sweep.json` on
   `jebel` (sha256 `54f86f23...2a51291`). It holds per-seed results of the sequence study's
   2-layer arm (its control, which reproduced the sequence study's rows to every digit) on three
   corpora.

## Decisions

Settled by building for extensibility and speed, with the alternatives and why they lose:

- **D1. Keeping a study and an A/B apart. Settled: (c).**
  - (a) Studies take `ab.py`'s run lock. Pros: one mechanism, and the refusal names what holds
    it and since when. It also stops `sync_checkout` from moving the checkout under a study,
    because `run --detach` syncs under the lock. Cons: every study's command line takes it.
  - (b) `ab.py`'s pre-flight refuses a high load. Pros: catches anything, including stray jobs
    no lock knows of. Cons: says only that the machine is busy, not why; doesn't protect the
    checkout.
  - (c) *Chosen.* Both. The lock names the cause and guards the checkout; the load check catches
    the rest.
- **D2. Where a study takes the lock. Settled: (a).**
  - (a) *Chosen.* In each study's command line, around the stages that train (`sweep`, `tune`,
    `time`, the addition harness's `run`), through one helper. Pros: explicit; `status`,
    `report` and the tests never take it, so `./cli test` on `pyramidon` can't collide with a
    local `ab.py run`. Cons: a new study must remember it. A test per command line checks that a
    held lock refuses it.
  - (b) Inside `run_parameter_sweep` and the harness's `run`. Pros: one place, can't be
    forgotten. Cons: the tests and `demo_conv_depth_uci_digits_comparison.py` take it too, and a
    test run would fail beside a real run.
- **D3. The load rule. Settled: (a).**
  - (a) *Chosen.* The pre-flight refuses when the 1-minute load average is above the profile's
    `high_load` (1.5 on `jebel`; `DEFAULT_RULES` without a profile). First it waits up to 5
    minutes, checking every 15 s, because the 1-minute average lags: after a study stops, a load
    of 6 takes about 1.5 minutes to fall under 1.5. The refusal lists `busy_processes()`.
    `--allow-load` overrides it, and the manifest records that. Pros: reuses the profile's
    measured rule. Cons: a short spike near the start can refuse a run, though the wait covers
    most.
  - (b) A separate refusal threshold. Cons: a second number to measure and keep, with no data
    for it.
- **D4. When an A/B meets a study. Settled: (a) by default, (b) on request.**
  - (a) *Chosen, the default.* Refuse, naming the study, its pid, its start and its resume
    command (from the lock's record, D1). Pros: nothing is stopped without asking. Cons: the A/B
    waits.
  - (b) *Chosen, on request.* `ab.py run --drain` (and `remote run --drain`) works only when the
    holder is drainable (D5). It writes the study's stop file and starts detached. The detached
    process waits for the lock, so the wait survives a dropped ssh session; `status` shows the
    run as waiting on the study. It then syncs, checks and runs as now, and prints the study's
    resume command after the report. Pros: an A/B goes in at the next gap, at the cost of the
    slowest run in flight (minutes for the column transformers) and no lost runs. Cons: someone
    must resume the study.
  - (c) (b), with `ab.py` resuming the study itself. Pros: nothing to remember. Cons: `ab.py`
    starting commands read from a lock file; a failed A/B leaves the question of whether to
    resume; the agent is already waiting on the A/B and resumes in one command.
- **D5. Which studies drain. Settled: (a).**
  - (a) *Chosen.* The addition harness, which already resumes. A stop file `drain` in the sweep
    directory makes `run` submit nothing new, finish its runs in flight, delete the file and exit
    with "drained: N runs finished; resume with ...". `scripts/addition_study.py drain CONFIG`
    writes the file. Pros: small, and builds on the resume and `.running` markers. Cons: the
    `run_parameter_sweep` studies (sequence, attention dropout, rotary, patch, residual depth,
    batch size) have no resume, so they refuse under D4 (a) only.
  - (b) Also make `run_parameter_sweep` resumable (a result file per job, as the harness).
    Pros: every study drains. Cons: a change to six studies' result handling for sweeps of 1-3
    hours; moved to next-steps.
- **D6. The addition harness's progress. Settled: (a).**
  - (a) *Chosen.* `status` adds the elapsed time, runs a minute over the last hour (from each
    result's new `finished` time, or its file's mtime for results from before), the runs still
    possible (an upper bound from each unresolved ladder's bisection bracket, via `wanted`), and
    an end time as a range: the runs pending now at the recent rate, to the bound at it. Pros:
    cheap; a handover can quote a range instead of a guess. Cons: under bisection the total is
    a bound, so the range can be wide early on.
  - (b) A cost model per run from size and `n`. Cons: fitted on the sweep it forecasts; more code
    for an estimate.
- **D7. The harness's timing fields. Settled: (a).**
  - (a) *Chosen.* Add `fit_seconds` (`train_epoch` only), `screen_seconds` and `check_seconds`
    (the whole-catalogue checks after a passing screen, and the final evaluation). Keep
    `train_seconds` and `seconds` with their current meaning. `report` adds seconds an epoch,
    `(fit + screen) / epochs`. Pros: additive, so the calibration's and this sweep's results stay
    comparable and readable; the next budget is costed from the true per-epoch time. Cons: three
    more fields a result.
  - (b) Redefine `train_seconds` as training only. Cons: the same name means two things across
    the stored results.
- **D8. Keeping studies' raw results. Settled: (a).**
  - (a) *Chosen.* Every study's `--out` defaults to
    `data/studies/<study>/<stage>-<YYYY-MM-DD-HHMM>.json` (gitignored). The file is an object
    `{"provenance": {...}, "results": [...]}` (the commit, crate hash, host, arguments and start
    time, as the addition harness's `_provenance`). Readers (`sequence_study`'s `--tuned`) accept
    the old bare list too. A findings PR records the file's path, host and sha256 in the study's
    docstring. Pros: nothing to remember at launch; a later study finds the file from the
    docstring. Cons: one copy, on one machine.
  - (b) Also archive raw study results in `indrajala-benchmarks`. Pros: durable, off the machine.
    Cons: widens the archive past timing, which next-steps rules out without the owner; left
    there.
  - The attention-dropout file moves to `data/studies/attention-dropout/sweep-2026-10-10.json` on
    `jebel`, and its docstring gains the path and sha256. The addition harness keeps its own
    `data/addition/` directories, already gitignored and already provenanced per result.
- **D9. The rotary study's baseline. Settled: (c); amends the rotary workplan's D8.**
  - (a) The recorded means ± sd (the rotary workplan's D8 (a), as built on `rotary-study`).
    Cons: no per-seed differences, which matter here: rotary and learned share seeds and
    shuffles, so their paired differences are tighter than either arm's spread.
  - (b) Rerun both learned arms beside the rotary ones (the rotary workplan's D8 (c)). Cons: 40
    more runs, the rotary sweep's cost again.
  - (c) *Chosen.* Take the 2-layer learned arm's per-seed results on Tiny Shakespeare, Herodotus
    and Euclid from the attention-dropout file (its contiguous `control`, the same model, rate,
    seeds and 10 epochs). Rerun only the 25 runs it lacks: the 1-layer arm on all four corpora,
    and the 2-layer on the *Muqaddimah*. The study checks that the file matches its recorded
    sha256 and that every learned row reproduces RECORDED's means, then reports per-seed
    differences. Pros: paired comparisons on every corpus and arm. The cost is about half the
    rotary sweep's (RECORDED's seconds an epoch: 299.9 of 601.1), about 1.25 h on the handover's
    2.5 h estimate. Cons: the study reads a file another study wrote, pinned by hash.
  - Built in the rotary workplan's stage 8 PR, not here (`--control FILE`, so it doesn't wait on
    stage 5 below); that PR rewrites the rotary workplan's D8 to match.

## Timing and checks

Every stage is tier 0. `ab.py` and the study harnesses aren't timed paths, and none of this can
reach training results, so no stage needs the golden run. `ab.py`'s changes are checked by its
tests, and stage 2's and 4's end to end on `jebel` as their stages say. **Don't pull on `jebel`
while a study runs** until stage 2 is there: these stages merge to `main` any time, but reach
`jebel`'s checkout only between studies.

## Stages

1. **This workplan.**
2. **The machine lock (D1 (a), D2).** `run_lock` moves from `scripts/ab.py` to
   `indrajala_ml/measurement/machine_lock.py`, with its record as one JSON line (`what`, `kind`
   run or study, `pid`, `since`, and for a study its `resume` command and its `drain` path or
   none). `ab.py` imports it, and its refusal prints the record. Every study command line that
   trains takes it, through `machine_lock.study("<study> <stage>", resume=..., drain=...)`. Tests:
   a held lock refuses `ab.py run` and each study command line (a temporary `AB_RUNS_ROOT`); a
   killed holder frees it; the lock's file isn't inherited by the forkserver workers. On
   `jebel`: a tiny addition config running, `ab.py remote --host jebel run` is refused naming it.
   Docs: measurement.md §2, §4 and §11, and CLAUDE.md's "Check `jebel`, then go".
3. **The load refusal (D3).** In `_preflight`, for `run` and `extend`: wait, then refuse, with
   `--allow-load` and its manifest record. Tests with `os.getloadavg` patched: under the rule
   passes, over it waits then refuses with the busy list, a fall during the wait passes, and
   `--allow-load` passes and is recorded.
4. **The drain (D4 (b), D5).** The harness's stop file and `drain` command; `ab.py run --drain`
   and `remote run --drain`. The detached process waits on the lock, with a run directory whose
   status is "waiting on <study>", and prints the resume command after the report. Tests: a
   drained `run` finishes its jobs in flight, starts none, exits, and resumes to the same
   results as an undrained run; `--drain` against a non-drainable holder is refused. On `jebel`:
   one real drain of a tiny config under a short `cmd` A/B, then the resume.
5. **The harness's progress and timings (D6, D7).** `finished` time, the three timing fields,
   `status`'s rate, bound and range, and `report`'s seconds-an-epoch column. Tests: the bound
   on a hand-built ladder; the range from synthetic results; the fields add up to `seconds`
   within the bookkeeping.
6. **Studies' result files (D8).** The default `--out` for every study, the provenance object,
   `--tuned` accepting both forms, `data/studies/` gitignored, the attention-dropout file moved
   on `jebel` and its path and sha256 written into its docstring. Tests: the default path, the
   object, the old list read back.
7. **Docs and retirement.** measurement.md's studies paragraph (the lock, the drain, result
   files); next-steps: D5 (b) (resumable `run_parameter_sweep`), D8 (b), and the run queue's
   entry updated (the drain covers a study in the way, the queue still A/Bs back to back);
   retire this workplan.

## Not in this plan

- A run queue on `jebel` (From the benchmark archive, next-steps): unchanged.
- Studies on `pyramidon` beside development: they take the lock there too, but nothing times on
  `pyramidon` except diagnostics.
- Pausing a study mid-run (SIGSTOP and the like): its memory stays resident and its caches go
  cold for the A/B; a drain leaves the machine clean.
