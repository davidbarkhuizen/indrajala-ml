# Workplan: a benchmark archive, tiered benchmarking, and a remote benchmark machine

**Status: D1-D10 settled (owner, 2026-10-08). Stages 1-4 done (2026-10-09): the archive exists
and is seeded, with CI reproducing every report; tier 1 runs 4 passes. Stage 5's code and docs
are done (#596-#598): a tier 1 A/B started from `pyramidon` ran on `jebel`, survived its ssh
session being killed, and reported back while `pyramidon` built and passed its tests. Left for
the owner: `sudo scripts/install_benchmark_setup.sh` on `jebel`, and Claude Code and `gh` on
`pyramidon`. Then stage 6.**

Three changes that belong together, because each makes benchmarking cheaper to do and its results
last longer:

1. **A separate public repository, `indrajala-benchmarks`,** that accumulates golden runs, raw
   A/B runs with their reports, and the machine profiles they ran on, for every change that
   matters. Results stop living only on one machine's disk.
2. **Benchmarking tiers:** how much timing a change needs, from none to a full baseline, so the
   timing protocol stops growing with every new feature.
3. **The Ryzen laptop (`pyramidon`) as the development harness host, and the i7 (`jebel`) as a
   benchmark machine administered from it over ssh.** Development no longer has to stop while a
   benchmark runs.

## Why

- **Results are lost with the machine.** The laptop's golden run went when the plan assumed the
  laptop was gone, so the benchmark machine workplan's D5 (do the two machines compute the same
  bits?) couldn't be answered until the owner found it still existed. Today every golden file
  lives in an ignored `data/refactoring/` and every raw A/B in `~/code/ab-runs/` on the machine
  that ran it.
- **Raw runs outlive their reports.** PR bodies keep only the rendered tables. On 2026-10-08 the
  noise rules became per machine (#582), and only raw runs can be re-reported under new rules.
  `tests/fixtures/ab/` already keeps four archived runs whose published tables the tests
  reproduce: the archive is that, for every run worth keeping.
- **Profiles beside results** let a number be compared across machines and over time, which the
  crate's planned tuning on CPU attributes needs (several machines, each with its attributes and
  its tuned values).
- **Benchmarking is starting to slow development.** The six A/As of the i7's baseline took about
  100 minutes of quiet machine, during which no build, test or lint could run. Every new feature
  adds paths that could be timed. Without a rule for how much is enough, the cost keeps growing.
- **The benchmark machine is also the development machine.** Claude Code, the terminal and the
  desktop run on `jebel` during every pass (benchmark machine workplan, Pitfalls), and the
  checkout `ab.py` reads is the one being edited. Splitting the roles removes both problems.

## Where things are now

| | size | where |
| --- | --- | --- |
| a golden run (106 networks, JSON) | 1.4 MB, 270 KB gzipped | `data/refactoring/golden_run.json`, ignored, one per machine |
| an A/A or A/B run directory | 136 KB-1.3 MB | `~/code/ab-runs/<date>-<name>/` on the machine that ran it |
| the i7's six-run baseline | about 2.3 MB | `~/code/ab-runs/2026-10-08-baseline-aa-*` |
| machine profiles | a few KB each | `docs/machine_profiles/`, with each machine's noise rules since #582 |
| archived runs used as test fixtures | | `tests/fixtures/ab/` (#477, #479, #480, the pyo3 bump) |

- `ab.py` has no archive command. `golden_training_run.py` records and checks one file and keeps
  no history: "a golden file is only valid on the machine that recorded it".
- CLAUDE.md's Timing rules exempt a PR that touches no `learn*` or `classify_rows` path, and one
  confined to pure Python. Every other PR that touches the training path gets an A/B, with no
  distinction between "show it didn't get slower" and "show it got faster".
- `jebel` runs everything: Claude Code, development, and benchmarks. `pyramidon` is reachable
  over ssh both ways (set up 2026-10-08) and ran the benchmark machine workplan's laptop columns
  (D11), but hosts no development yet.
- The setup script needs sudo after every boot, and `sudo` only works in the owner's own terminal
  (no TTY for the agent).

## Decisions

Settled with the owner on 2026-10-08.

- **D1. A separate public repository: `davidbarkhuizen/indrajala-benchmarks`.** Public, like
  indrajala-ml, so the numbers behind any speedup claim can be checked. It holds hostnames, kernel
  and library versions and CPU details: nothing secret. Not a directory in indrajala-ml: it would
  grow a library's clone forever with data no user needs.
- **D2. What it holds:**
  - **golden runs,** gzipped, each with its machine, commit pair, date and reason;
  - **`ab.py` run directories,** raw (manifest, each pass's output and logs, not the `data`
    symlink), with `report --brief` and `report --md` (and `--pooled` for an A/A) rendered at
    archive time;
  - **a copy of the machine profile each record ran under,** so a record stays readable after the
    profile is re-recorded.

  Not: crate wheels, worktrees, datasets, or anything outside a run directory. The baseline
  documents (`docs/machine_profiles/<machine>.md`) stay in indrajala-ml, where the docs link them;
  they cite archived records by path.
- **D3. When something is archived (the tiers, D7):** tier 2 and 3 runs always; a golden run when
  it is recorded for a material change or new functionality (D10); nothing for tiers 0 and 1. A
  record is immutable once merged. A correction is a new record that names the one it replaces.
- **D4. Records arrive by PR, auto-merged.** The archive command opens one PR per batch of
  records (typically one indrajala-ml PR's worth). The archive repo's CI validates them, and the
  command squash-merges when CI is green, the same flow as the other repositories with no extra
  step for the owner.
- **D5. Records stay readable.** Each record names the format versions it was written with (the
  `ab.py` manifest, the profile `schema_version`). The archive repo's CI re-renders every
  archived run's report with indrajala-ml's current `ab.py` and compares it with the stored one,
  so a change to `ab.py` that can no longer read old runs fails there. indrajala-ml's tests keep a
  few records as fixtures, as now.
- **D6. The layout:**

  ```
  machines/<hostname>/<captured_at>-profile.json      profile snapshots, by capture time
  golden/<hostname>/<date>-<commit7>.json.gz           golden runs
  golden/<hostname>/<date>-<commit7>.md                its commit pair, reason, what changed
  runs/<hostname>/<date>-<name>/                       ab.py run directories, raw
  runs/<hostname>/<date>-<name>/report.md, brief.txt   rendered at archive time
  INDEX.md                                             one line per record, newest first
  ```

  By hostname, because a profile is chosen that way (#582). A record also names its profile
  snapshot, so a host whose hardware changed is told apart by profile, not name.
- **D7. The benchmarking tiers.**

  | tier | when | what | archived |
  | --- | --- | --- | --- |
  | 0 | every PR | tests, lint; the golden-run check where the change could reach training results (not for docs or machine profiles) | no |
  | 1 | a PR that changes a timed path and claims no speedup | one A/B of the single most relevant benchmark, to show it got no slower | no |
  | 2 | a PR claiming a speedup | the full protocol (measurement.md) on the affected benchmarks | yes |
  | 3 | milestones: a release; a toolchain, numpy/BLAS, kernel or BIOS change; a new benchmark machine | the full baseline (the benchmark machine workplan's stages 4-5) | yes |

  New functionality that adds code paths without changing existing timed ones is tier 0: it
  adds golden entries (D10), not timings. Tier 1 runs 4 passes (`ONNO`) instead of 6 if stage 4
  shows that gives the same verdicts on the i7's A/As. **Stage 4: it does** (43 of 591 rows
  consistent over passes 1-4, 7.3%, against 8-9% by chance at 2 passes a side; measurement.md §1).
- **D8. Roles: `pyramidon` is the development harness host, `jebel` the benchmark machine,
  administered from it over ssh.** Claude Code and all development (edits, builds, tests, lint)
  run on the Ryzen. `jebel` runs benchmarks and nothing else, and is driven over ssh. The Ryzen
  still times things occasionally, as a second machine (the benchmark machine workplan's D11),
  under its own profile and noise rules.
- **D9. Remote sudo: a sudoers rule for exactly one root-owned script.** The setup script is
  installed as a root-owned copy, `/usr/local/sbin/indrajala-benchmark-setup`, with a `NOPASSWD`
  sudoers entry for that path alone, so the agent can re-apply the settings after a reboot over
  ssh. The repository's copy is never run as root directly: anyone who can edit the checkout could
  otherwise run anything as root. The owner re-installs the copy when the script changes (stage 5
  gives the one command).
- **D10. Golden runs are versioned, not replaced.** A golden run is archived when a material
  change is approved (CLAUDE.md: re-recorded only for an owner-approved correctness improvement)
  or when new functionality adds entries (as the residual and attention entries were added: no
  earlier entry moves, which the check proves). Each version records which of the two it is and
  what moved. The machine's working file in `data/refactoring/` stays as the check's input.

## Pitfalls to design around

- **What a run directory may contain.** Pass stdout and stderr are whatever the benchmark
  printed. The archive command copies only the files `ab.py` wrote (the manifest's stems), and
  refuses a run directory with anything else in it. `ab.py` records only the thread variables
  of the environment, never the whole of it.
- **Hostnames outlive hardware.** A new CPU in `jebel` keeps the name. Records name their profile
  snapshot, and the archive command refuses a run whose profile identity no longer matches the
  snapshot it would cite.
- **A remote run must survive the ssh session.** The dev host may sleep, or the connection drop,
  mid-run. The remote command starts `ab.py` detached (`setsid`, output to the run directory) and
  returns; the dev host waits on a separate ssh `wait` that can be re-attached. A dropped
  connection never kills a run.
- **Both sides must be on GitHub before a remote run.** `jebel` checks out commits by fetching
  them, so an A/B's branch is pushed first (it is anyway, for its PR). Commits that exist only on
  the dev host can't be timed.
- **Two runs at once.** With the agent no longer on the benchmark machine, nothing stops a second
  `ab.py run` starting while one runs. `ab.py` takes a lock for the length of a run and refuses a
  second.
- **The quiet-machine rule changes meaning.** Today it forbids builds, tests and lint during a
  run because they share the machine. After D8 it applies to `jebel` only: development on the
  Ryzen carries on during a run. CLAUDE.md's Timing section changes with stage 5, not before.
- **Auto-merge without a reviewer.** CI is the only gate on archive PRs, so it has to check
  everything that can be checked: schemas, file lists, report reproduction (D5), and that a
  record doesn't overwrite one already merged (D3).

## Stages

Each stage is one PR (or a pair: the archive repo's, then indrajala-ml's), merged before the next,
with `./cli test`, `./cli lint` and, where the change could reach training results, the golden
run green. ssh between the machines already works both ways (key-only, set up for the benchmark
machine workplan's D11), so stage 5 here builds on it.

### Stage 1: the archive repository

1. Create `davidbarkhuizen/indrajala-benchmarks` (public, D1) with the layout (D6), a README (what
   it holds, how records arrive, how to read one), `INDEX.md`, and the format notes.
2. Its CI: validate each record's files against D2 and D6, each profile snapshot against the
   schema version it names (the schema is fetched from indrajala-ml at that version), and that no
   merged record changed. Report reproduction (D5) comes in stage 3, once there are runs.
3. Branch protection on `main`: PRs only, CI required (D4).

Done when an empty archive repo exists with green CI on a no-op PR.

### Stage 2: the archive commands in indrajala-ml

1. `ab.py archive [RUN] [--reason TEXT]`: checks the run is complete and its files are the ones
   it wrote (Pitfalls), renders `brief.txt`, `report.md` and, for an A/A, `pooled.md`, copies the
   run and its profile snapshot into a clone of the archive (`AB_ARCHIVE_REPO`, default
   `~/code/indrajala-benchmarks`), adds the `INDEX.md` line, and opens and auto-merges the PR
   (D4). `--no-pr` stops after the commit, for tests.
2. `golden_training_run.py archive FILE --reason {material,new-functionality} --note TEXT`: the
   same for a golden file, with its commit pair and what moved (D10).
3. Tests against a temporary git repository standing in for the archive, with no network.
4. measurement.md: how and when to archive.

Done when both commands archive into a test repository and `--no-pr` leaves the expected
commit.

### Stage 3: seed the archive, and reproduction in CI

1. Archive the i7's baseline: the six A/As of 2026-10-08 (`baseline-aa-*`), the golden run
   recorded on `jebel` at stage 0, and the i7 profile.
2. Archive the laptop-era fixtures in `tests/fixtures/ab/` as `pyramidon` history, with their
   PRs as the reason.
3. Archive the failed run `-pl1-reset` with its reason: the first record of `ab.py`'s per-pass
   policy check catching a reset.
4. The archive CI's report reproduction (D5): check out indrajala-ml `main`, re-render every
   archived run's report, and compare with the stored one.

Done when the archive holds the i7 baseline and its golden run, and CI reproduces every report.

### Stage 4: the tiers

1. **The 4-pass check (D7):** `ab.py report --passes 1-4`, a report over a subset of a run's
   passes (an `ab.py` option, not a script). Re-report the i7's six A/As over passes 1-4 and
   compare the verdicts with the 6-pass ones. Tier 1 takes 4 passes if no verdict changes beyond
   what chance gives at 2 passes a side; otherwise 6.
2. **CLAUDE.md's Timing section and measurement.md §1, §9:** the four tiers, what decides a PR's
   tier, and which benchmark is "the most relevant" for a tier 1 PR (a short table: changed path
   to benchmark).

Done when CLAUDE.md states the tiers and tier 1's pass count is set from the check.

### Stage 5: the Ryzen as the harness host, `jebel` as the benchmark machine (D8, D9)

1. **Owner:** ssh keys from `pyramidon` to `jebel` (and back, for the benchmark machine
   workplan's D11); install the root-owned setup copy and its sudoers entry with one command this
   stage provides (`scripts/install_benchmark_setup.sh`, run once with sudo).
2. **The Ryzen set up for development:** Claude Code, the checkout and submodule, `.venv`,
   `./cli build-rust`, `./cli test`, `./cli lint`, its golden run recorded and archived; its
   profile re-recorded at schema version 4 with its noise rules kept.
3. **Remote runs:** `ab.py remote run|status|report|archive --host jebel ...`. It fetches the
   commits on `jebel` (both repositories), starts `ab.py run` there detached, waits on it over a
   re-attachable ssh session, and copies the reports back. `ab.py run` takes the run lock
   (Pitfalls). A failed machine check on `jebel` (a reboot) re-applies the settings through the
   sudoers entry and runs the check again, once.
4. **The setup script's machine check after a reboot:** the agent runs
   `sudo /usr/local/sbin/indrajala-benchmark-setup` over ssh; the setup script itself is
   unchanged.
5. **Docs:** CLAUDE.md's Timing section (runs go to `jebel` through `ab.py remote`; the quiet
   rule applies to `jebel` only), measurement.md §2 and §4, and the session handover's "sudo
   needs the owner's own terminal" becomes "after a reboot, `ab.py remote` re-applies the setup".

Done when a tier 1 A/B started from the Ryzen runs on `jebel`, survives a dropped ssh session,
and reports back, while the Ryzen builds and tests.

### Stage 6: retire the plan

The README links the archive repository where it describes measurement. Leftovers go to
next-steps.md.

Done when this file is deleted and next-steps.md lists it.

## After this plan

- **Results from other people's machines,** if the project is used widely: submissions by PR
  with a recorded profile, a clean tree and a named commit. The formats here allow it; the review
  process for untrusted submissions is its own plan.
- **A run queue on `jebel`** (several A/Bs queued from the Ryzen, run back to back), if one at a
  time with the lock turns out to be the bottleneck.
- **Crate tuning on CPU attributes** ([i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md))
  uses the archive's profiles and runs from both machines.

## Out of scope

- Changing the timing protocol itself (the verdict rule, the noise rules per machine): only tier
  1's pass count, from data (D7).
- Hosting results anywhere but the archive repository (no dashboards, no release assets).
- Any change to training code or the crate.
- CI-run benchmarks: GitHub's runners are shared and noisy, so no timing is ever taken there.
