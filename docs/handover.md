# Session handover: benchmark-machine workplan, 2026-10-08 (after stage 2)

**Plan:** [benchmark-machine-workplan.md](benchmark-machine-workplan.md). Its status line is up to
date: D1-D10 are settled, stages 0-2 are done and stages 3-7 are planned.

## Done

- **Stage 0 (machine setup, no PR):** `jebel` (the i7-9700K) has Rust 1.98.1, a `.venv` on Python
  3.14.4, maturin 1.15.0, numpy 2.5.3 with scipy-openblas 0.3.34 (Haswell kernels), MNIST and a
  golden run file (`data/refactoring/golden_run.json`, 106 networks).
- **Stage 1: [PR #576](https://github.com/davidbarkhuizen/indrajala-ml/pull/576), merged as
  `04c90c0`.** The profile reads Intel's turbo, EPP and `intel_pstate` mode. It also added
  `scripts/benchmark_machine_setup.sh`, made `docs/machine_profiles/i7-9700k.json` the
  `PROFILE_REFERENCE` and rewrote measurement.md §2 for this machine.
- **Stage 2: [PR #577](https://github.com/davidbarkhuizen/indrajala-ml/pull/577), merged as
  `6f0e9de`; local `main` is synced.**
  - **D1 is (a), the defaults:** governor `powersave`, EPP `balance_performance`, turbo on. Both
    A/As (`prepared_dataset_timing`) were clean and close. (a) had the smaller pass-to-pass spread:
    a median of 3.2% and a max of 5.6% over the rows, against 3.5% and 7.2% under EPP
    `performance`. The runs are `~/code/ab-runs/2026-10-08-stage2-aa-a-balance-performance` and
    `-aa-b-performance`.
  - **New D10: PL1 is 65 W, set by the setup script on each boot.** At the stock 95 W, ten minutes
    of all eight cores reached 100 C and throttled 460 times, and the cooling can't be improved.
    At 65 W the same load ran at 3.69 GHz and about 78 C with no throttling, about 8% slower.
    Single-core work (about 37 W) and the threaded conv case (about 40 W) are unaffected.
  - **Profile schema version 3** adds `identity.cpu.power_limits` (`long_term_w`,
    `short_term_w`), so the machine check catches a reboot that resets PL1. The Ryzen profile has
    it as null, and the i7 profile is re-recorded with 65 W.
  - **Findings for later stages:**
    - A core reaches full clock about 3 ms after idle under (a); the laptop took hundreds of ms.
    - Conv mini-batch 512 keeps only about 1.3 CPUs busy.
    - Eight single-threaded copies at once each ran 2.7x slower than one alone. The clock explains
      about 1.2x of that; the rest is most likely shared L3 and memory bandwidth. That matters for
      stage 6's threading re-check.
    - All of these are in the workplan's stage 2 section.
- **The machine is set up right now.** The owner re-ran the setup script after stage 2, and
  `machine_profile.py compare` reports "identity matches".

## Next: stage 3 (the baseline commit)

1. Tag `main` (`6f0e9de`, or whatever `main` is when stage 3 starts) as `baseline-i7-9700k` here.
   Tag the submodule's commit (`rust/`, currently `66ecc0d`) with the same name in
   `indrajala-math-rust` (D6).
2. Create `docs/machine_profiles/i7-9700k.md` (D8):
   - the machine, with a link to its profile, the D1 policy and D10's PL1
   - the commit pair, and how to rerun it (stages 4-5's commands)
   - empty sections for stages 4-6

Stage 3 needs no measurement and no sudo. Stage 4 (A/As of every `ab.py` adapter) is the next long
run.

## Things to know

- **sudo needs the owner's own terminal.** There's no passwordless sudo, and `! sudo ...` fails
  because no terminal is available for the password prompt. Ask the owner to run sudo commands in
  their own terminal and paste the output.
- **The setup state is lost on reboot.** The governor, EPP, PL1 and `perf_event_paranoid` all
  reset, and the machine check refuses a run until `sudo scripts/benchmark_machine_setup.sh` is
  re-run. It prints `PL1, PL2: 65 W, 120 W` when PL1 has been applied.
- **The snap hold expires 2026-10-09 09:41 SAST.** Re-run the script before any run after that.
- **Before a run:** check `uptime` and `ps`, and confirm that `systemctl is-active
  apt-daily.service apt-daily-upgrade.service` prints `inactive` twice. Don't use `pgrep` to look
  for package jobs: the daemons `unattended-upgrade-shutdown` and `snap userd` always run, and
  `pgrep -f` matches its own shell.
- **Readings that change the machine's policy** need `ab.py run --allow-profile-change`, as stage
  2's (b) A/A did. Put the policy back afterwards.
- **`machine_profile.py compare` reports `rustc: null`** in a shell without `~/.cargo/bin` on
  PATH; prefix `PATH=$HOME/.cargo/bin:$PATH`. `ab.py` sets PATH itself.
- **Re-recording a profile:** run `machine_profile.py profile --out ...` on a committed, clean tree.
  An untracked file such as this note makes `repo_dirty` true; stage 2 moved the note aside for
  the capture.
- **Creating a GitHub branch:** a plain `git push` of a new branch fails with HTTP 500. Create the
  remote branch with `gh api .../git/refs` first; the PR-flow memory has the steps.
- **Stage 2's throwaway probes** were the load loop, the turbostat driver and the clock-up timer.
  They were in this session's scratchpad and are not in the repo. If a stage needs them again, they
  belong in `ab.py` or `scripts/`, per CLAUDE.md.
