# Working notes: gotchas for this repository and its machines

Hard-won operational knowledge, gathered from the session handovers (2026-10-12 to 10-14) so it
outlives them. Each note is something that went wrong once. The rules that govern the work are
elsewhere: CLAUDE.md, [experiment-design.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/experiment-design.md) (studies) and
[measurement.md](measurement.md) (timing); this file is the "how not to trip" companion. Add a note
where you meet a new trap; delete one that no longer holds.

## Git and GitHub

- **Push over ssh; https pushes fail:** `git push git@github.com:davidbarkhuizen/indrajala-ml.git
  <branch>` (the crate: `indrajala-math-rust.git`; the archive and dataset repos likewise).
  `gh pr create --head <branch>` works; for the archive, `-R davidbarkhuizen/indrajala-benchmarks`.
- **`gh pr merge --squash --delete-branch` deletes the local branch too** and switches to `main`
  when it can; check `git branch` and `git ls-remote --heads` after. It fails when the main
  checkout has local changes to files `main` touches: merge with plain `gh pr merge N --squash`,
  `git fetch origin main:main`, and delete branches by hand.
- **`gh pr checks --watch` can exit with a check still pending.** Merge only when every printed
  line says `pass`. A push to an open PR restarts CI: a watch started before the push watches the
  old run. A short `sleep 15;` before `--watch` avoids "no checks reported" just after a push.
  Gate the merge in the command, never by reading: `gh pr checks N | awk -F'\t' '$2!="pass"{bad=1}
  END{exit !(NR>0 && !bad)}' && gh pr merge N --squash --delete-branch` merges only when there are
  checks and every one is `pass` (#829 merged with `tests/` failing when a merge followed a `cat`
  of the checks). `awk`, not `grep -v`: an agent's shell may wrap `grep`.
- **One watcher per PR.** Check for a running `gh pr checks` before starting another.
- **CI checks out only HEAD** (no fetch depth): a test can't use past commits.
- **A PR body is limited to 65 KB.** Write PR comments with a quoted heredoc (`cat <<'EOF'`): a `%`
  in `printf` text breaks it; fix one with `gh pr comment --edit-last -F FILE`.
- **Don't write a result into a PR body before reading it** (a test count typed from expectation
  was off by one).
- **Bumping a submodule:** `git -C sub checkout X`, `git add sub`, then `git -C sub submodule
  update --recursive`. A top-level `git submodule update` first resets `sub` to the recorded commit.
- **`git update-index --cacheinfo 160000,<sha>,rust` (a pin without a checkout) accepts any
  SHA:** take it whole from `git -C <crate> rev-parse <ref>` and check `cat-file -t` first. A
  mistyped pin pushes a commit no checkout can fetch (2026-10-18, `litshe`: "upload-pack: not our
  ref").
- **Stage files by name, never `git add -A` over a directory:** the untracked `docs/handover.md`
  was committed that way once (#582, untracked again in #583).
- **Never `git commit -a` or `git add -A` on `pyramidon`; add paths by name:** its `rust/` checkout sits at the crate its `.venv` was
  built from (647f23c, while `main` pins 10dfec9), so `-a` commits that pointer too, and CI then
  builds the old crate: every shard fails (`WindowProgram has no op "merge"`, #1036 and #1037,
  2026-10-27, fixed by a commit setting `rust/` back). `git diff --stat origin/main` before pushing
  shows it. Again 2026-10-29: #1086 (`add -A`) and #1090 (`commit -a`) set main's `rust/` to two
  unmerged crate branches' commits, so A/Bs whose old side resolved through `main` timed the wrong
  crate until #1092 set it back; name the crates (`--old-crate`, `--new-crate`) in a crate A/B.

## Worktrees

- **A second PR beside uncommitted work goes in a worktree:** `git worktree add -q -b <branch>
  <path> <start>` (path before start point; a swapped order fails and a following `cd` lands in
  the wrong checkout).
- **A fresh worktree has no `.venv`, no data and no `rust/` checkout.** Link `.venv` (`ln -s
  ../indrajala-ml/.venv .venv`) and every corpus: `for d in $(ls ../indrajala-ml/data); do [ -e
  data/$d ] || ln -s ../../indrajala-ml/data/$d data/$d; done`, which skips `data/refactoring`: it
  holds a tracked file (`golden_run_rust.json`), so it is a directory in every checkout; link its
  untracked `golden_run.json` alone (`ln -s ../../../indrajala-ml/data/refactoring/golden_run.json
  data/refactoring/`). Through a link to the main checkout's directory, a `rm` or a rebase in the
  worktree reaches the main checkout's files (2026-10-17: a rebase refused, and an `rm` deleted the
  main checkout's tracked copy). A loop over `git check-ignore` skips directories that don't exist
  yet (`data/<d>/` matches directories only). Without the corpora the suites fail on MNIST and other
  files; without `rust/`, `./cli test` stops at once ("no tests ran", exit 0): run `PYTHONPATH=.
  .venv/bin/python -m pytest tests/ -q -p no:cacheprovider` for a package-only PR. `data/mnist`'s
  link isn't ignored, so stage files by name. `git worktree remove --force` deletes only the links.
- **Which package a worktree imports:** pytest imports the worktree's (rootdir); `python -c` and
  scripts import the main checkout's unless `PYTHONPATH=.` (the script's directory, not the cwd,
  is on `sys.path`; `indrajala_ml` is a namespace package with the editable finder). This holds
  for `scripts/measurement/golden_training_run.py` too. To prove a test fails without a fix, revert the fix in
  the worktree (`git diff > patch; git checkout file; ...; git apply patch`).
- **A config's paths are relative to the cwd:** to run a study command from a worktree on the main
  checkout's data, link `data/` or write a config copy with absolute paths.
- **Stacked local branches** (avoid them; if met): `git rebase --onto main <old base> <branch>`
  after each merge, then `git diff --quiet <tested> <new>` to confirm the tree is what was tested.

## Tests and lint

- **`./cli test` takes 5-10 minutes** (no pytest-xdist): run it in the background and let its own
  completion notify you. Read the summary line (`= N failed, M passed =`), never the exit code of
  `cmd > log; tail log`.
- **`./cli lint` stops at ruff's formatter** when a file would be reformatted, before pyright: run
  `.venv/bin/ruff format` first. Run pyright through `./cli lint`, not directly (that ignores the
  project config). Lint before pushing, not only the tests' files.
- **Registry walks find every `ArrayNetworkBase` subclass:** a new network class needs an entry in
  `tests/data/test_prepared_dataset.py` (`CONSTRUCTORS`; `_is_sequence` decides per-token labels,
  and a class without `predict_probabilities` is taken for binary), `test_seeded_init_parity.py`
  (`NUMPY_ONLY`), `test_legacy_saved_models.py` (`FIXTURE_LATER`) and `test_format2.py`
  (`SEQUENTIAL`). Grep for them before the full run. A shape mixin must not subclass
  `ArrayNetworkBase` at runtime (use the `if TYPE_CHECKING: _ShapeBase = ArrayNetworkBase`
  pattern).
- **Fixture names are prefix-filtered** (`startswith("Attention")`, `"Ensemble"`): a new prefix
  must not start with an existing one; a fixture's weights are seeded by its name, so renaming
  means regenerating. Untrained fixtures hold an entropy-seeded `rng`: compare with it stripped.
- **A test's fake network must follow the network's API** (the study tests' `AddingNetwork`
  returns tuple states since #739). Keep its check lengths short (`k` 1, test lengths 10-12) or a
  test takes 20 s.
- **`test_ab.py::test_a_host_without_a_profile_is_refused` fails on a loaded machine:** deselect it
  locally under load and say so in the PR; CI runs it.
- **Parity tests patch three numpy modules' `exp`** (`attention_array_layer`, `token_array_layer`,
  `recurrent_array_layer`). BLAS order is machine-dependent: never assert a numpy product by bits
  against a left fold.
- **Crate stubtest reads the installed stub:** after editing `indrajala_math_rust.pyi`, run
  `maturin develop --release` again. mypy isn't in the venv: make a scratch venv with
  `mypy==2.3.1` and run stubtest with `PYTHONPATH=.venv/lib/python3.14/site-packages` from a
  neutral directory. The crate's lint tools: `../.venv/bin/ruff` and `../.venv/bin/pyright
  --pythonpath ../.venv/bin/python` from `rust/`. The crate's ops take `heads` as a keyword after a
  `*list` unpack; `causal` is keyword-only.
- **A constant added to every entry of a weight read after a LayerNorm moves nothing:** a token's
  features sum to zero there (gamma 1, beta 0 at initialisation), so `W + 1` adds zero to each row's
  product. A test that perturbs a layer to show what reads it uses a random change (the
  recombining window's test, 2026-10-27, first passed `Wq + 1` as a no-op).
- **Pyright on `jebel` against a branch's crate:** `jebel`'s main `.venv` holds an older crate
  stub, so `./cli lint` there reports the newer ops' parameters (`straight`) as unknown. Give the
  copy a temporary `pyrightconfig.json` with the branch wheel's SITE in `extraPaths` (root and
  `tests`), never committed.
- **Python 3.13+ dataclasses have `__replace__`**, so pyright infers a PEP 695 dataclass's type
  parameter invariant; disjoint `@overload`s avoid needing covariance.
- **`load_network` returns `Any`**: pyright won't flag a `None` attribute on a loaded network;
  grep for it when touching code that loads networks.
- **The venv has no scipy:** Fisher's exact test by hand (`math.comb`), or add the dependency in
  its own PR.

## Studies

The studies' notes (running and reading them, the batch runner, the window study's API, their
background jobs) moved to indrajala-studies' `docs/working-notes.md` with the research code
(2026-10-31).

## Code internals

- **`WindowRecurrence(factors, cross, cell, pooling)`:** `cell` is the third field; pass `cell=` by
  keyword. Machine states are tuples (one code per factor) everywhere, a single factor included.
- **Window networks have `columns`, not `dimension`;** their labels are a class per column of the
  whole `W * k` state.
- **A pure-Python window keeps per-example state in lanes:** `StateInputLayer`'s `example_fields`
  (`_example`, `_state`, `_token_sum`) are captured after each example's forward and backward; its
  `state_deltas` list is reset in `begin_example(0)`, so read it after the window's backward pass,
  before the next window's forward.
- **The Python network's parts aren't `TrainableLayer`s** (no `forward`): the optimiser needs only
  `weight_sets()`, and `snapshot()`/`restore()` append the parts after `trainable_layers`.
- **A new window layer kind needs a window-program op:** the Rust window network builds a
  `WindowProgram` from its layers (`_program_op`), and a layer with no op refuses to build;
  `test_every_layer_class_a_window_holds_has_an_op` fails first. Add the op to the crate's
  `program.rs` (forward, its cache, its downstream and gradients) with a parity test, as stages 2
  and 4 did, then the descriptor here.
- **The program's training step keeps every window's forward pass:** past `memory_budget` (2 GiB)
  `window_program_learn` refuses with that name; its own masks aren't the layers', so a test that
  reads a dropout layer's last mask reads it through `per_layer_path`.
- **Developing the crate beside running jobs:** never `maturin develop` into the main `.venv` while a
  study or long test runs from it (or from a worktree whose `.venv` links to it). Build in the
  crate worktree's own venv, then `maturin build --release -o DIR` and `pip install --no-deps
  --target SITE` the wheel, and run indrajala-ml's tests with `PYTHONPATH=SITE:.`; pyright needs
  SITE in a temporary `pyrightconfig.json`'s `extraPaths` to see the new stub. Don't replace SITE
  while a suite runs: tests that start subprocesses import it again.

## Machines

- **`sudo` needs a password on `jebel`, `stone` and `klip`; `litshe` alone has passwordless sudo.**
  An agent can't sudo on the first three, and the owner's `! ssh -t host sudo ...` fails too (a `!`
  command has no tty: "a terminal is required to authenticate"); the owner runs those in a terminal
  of their own.
- **Development on `pyramidon`, timing and long studies on `jebel`** (measurement.md section 4).
  Before a run on `jebel`: `uptime`, its processes, `pgrep -f '^/usr/bin/python3
  /usr/bin/unattended-upgrade'` and `pgrep -x dpkg`.
- **`jebel` at or below 65 W.** After a reboot PL1 comes back at 95 W: `ssh jebel 'cd
  ~/code/indrajala-ml && .venv/bin/python scripts/measurement/ab.py check --apply-setup'`, then check
  `/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw` is 65000000.
- **A study run on `jebel` or `stone` needs the straight-through crate on its path:** the continuous
  cell's `--continuous` calls the crate with `straight` (the rust/ bump fb0f1be), which `jebel`'s
  `.venv` (its checkout behind `main`) and `stone`'s lack. Put the built crate first:
  `PY="env PYTHONPATH=$HOME/code/site-straight:WORKTREE .venv/bin/python"` (`jebel`;
  `$LABOUR/site-straight` on `stone`). Without it every run ends in a second with
  `window_program_learn() got an unexpected keyword argument 'straight'` (D67's first launch,
  2026-10-23). A launcher's `ended` lines in its first minute are the sign: read one `.out`.
- **`jebel` shares its power supply with the wifi access point:** a power cut (2026-10-18, about
  45 minutes) takes both down, so `stone` loses its network too and moves to another wifi. After one,
  re-apply `jebel`'s setup (PL1) and relaunch its study; `stone`'s runs go on.
- **`stone` can drop off the wifi with no power cut** (2026-10-21: it jumped to another network;
  the owner moves it back to hlatikulu). Its runs survive (launched with `setsid`); only ssh
  waiters on `pyramidon` die. Wait on `stone` by polling `timeout 30 ssh stone 'pgrep -f ...'`,
  reading an ssh failure as "still running", never as "ended"; wrap a launching ssh in `timeout
  20` and confirm the launch by its log.
- **`pgrep -f`/`pkill -f` over ssh match the remote shell's own command line** when the ssh command
  has more than one part (`ssh host 'for p in $(pgrep -f PAT) ...'`): the `bash -c` carrying it
  contains PAT, so killing its process group kills the ssh session (exit 255) halfway through
  (2026-10-27, twice). List the targets first (`ps -eo pid,pgid,args`), then kill those process
  groups by number in a second call; a single-command `ssh host 'pgrep -f PAT'` (a waiter) is safe.
- **`jebel.local` sometimes doesn't resolve** (mDNS): try its address (it was `192.168.1.88`). A
  connection that opens but times out "during banner exchange" means `jebel` is thrashing, not
  down.
- **`jebel`'s non-interactive shell lacks cargo:** `PATH=$HOME/.cargo/bin:$PATH ./cli build-rust`.
  **`gh` isn't authenticated on `jebel`:** open PRs for its branches from `pyramidon`.
- **`ab.py remote run` refuses when `jebel`'s main checkout shows `M rust`** ("must be clean"):
  its `rust/` was left at another crate commit (2026-10-19: 13ccf90 against main's 40a2761). With
  nothing running from that checkout, `git submodule update rust`, then `./cli build-rust` (cargo
  on the PATH, above) so its venv's crate matches the pin, then relaunch.
- **Studies and `ab.py` share `jebel`'s machine lock:** an A/B is refused while a study runs,
  naming it. The lock's record is cleared when the hold ends; a line left in `~/data/indrajala-ml/runs/ab/run.lock` by a killed job means nothing (it's an `flock`; `machine_lock.holder()` checks the pid).
- **The permission classifier has blocked `pkill` and `kill` over ssh on `jebel`** at times; a kill
  of a named PID or process group the owner asked for went through. If blocked, ask the owner.
- **`stone`, the third machine (since 2026-10-18): studies only, never timing.** `ssh stone`
  (192.168.1.101, user david); i7-1255U laptop (2 performance and 8 efficiency cores, 12 threads,
  15 W), 30 GB. **Available weekdays 18:00-07:00 and all weekend**: a study must fit the window or
  split by whole runs. Lid and idle sleep are off (logind drop-in, GNOME), and it should be on AC.
- **`stone`'s sandbox (the owner's rules):** work only in `~/labour` and below, and **no authed
  service on or from it** (no `gh`, no GitHub remotes, no tokens). Code goes as git bundles by scp
  from `pyramidon` (`git bundle create X.bundle main`; a bundle of one branch has no `HEAD`, so clone
  with `-b main`; the submodule from its own bundle with `git -c protocol.file.allow=always
  submodule update --init`), data and results by rsync from `pyramidon`, with manifests. Anonymous
  downloads are allowed (rustup); pip installs offline from `~/labour/wheels` (fetched on
  `pyramidon` with `pip download --only-binary=:all:`, leaving out `requirements.txt`'s `-e .`,
  which zips the whole repo, data and all). `source ~/labour/env.sh` first: caches, config, pip and
  `AB_RUNS_ROOT` (the machine lock) all under `~/labour`. Checkout `~/labour/code/indrajala-ml`.
- **`stone`'s throughput:** at 8 workers it matches `jebel` at 6 (the ladder's `p` 13 cell, 30
  epochs, seeds bit-identical with `jebel`'s round 2, screens equal at every matched epoch). A run
  takes 0.89x `jebel`'s time at 4 workers, 1.34x at 8 and 1.93x at 12, so 12 adds about 4% over 8:
  use 8. Measured only on early steps and 3.5-minute bursts; a long run's sustained clocks may be
  lower on a 15 W laptop.
- **`klip`, the fourth machine: studies only; off for now (the owner, 2026-10-30): leave it out of `--hosts`.** `ssh klip` (192.168.1.104); Ryzen 7 3700U laptop
  (4 cores, 8 threads), 5 GB, so about 3 small runs at once. Always on, all cores; the owner's
  limit (2026-10-21): <= 85 C, ideally <= 80 C. **CPU boost is off** (`cpu-boost-off.service`;
  `/sys/devices/system/cpu/cpufreq/boost` reads 0): boost overheats it. **Run every job under the
  thermal guard** `scripts/runner/thermal_guard.sh LOG [STOP] -- COMMAND...`: it starts the command in its own
  process group, logs the k10temp sensor every 5 s to LOG and stops the group at STOP C (85) or
  above, with an "ended, status N, peak T C" line. A waiter there watches the probe's "probe ended" line
  or the guard's process gone (`pgrep -f "[t]hermal_guard.sh <its LOG's directory>"`).
- **Local `ab.py run` on `pyramidon`** needs `AB_CRATE_REPO=/home/david/code/indrajala-ml/rust` and
  `--skip-profile`; diagnostics only. `ab.py remote run` names the run after the laptop's branch:
  pass `--name`.
- **Archiving a golden version:** `golden_training_run.py archive data/refactoring/golden_run.json
  --reason new-functionality --note "..." --commit <squash-merge commit>` on each machine. On
  `jebel` it pushes its branch and fails at `gh pr create`: open the PR from `pyramidon`. The
  archive repo needs the branch up to date with `main` (`gh pr update-branch`) and doesn't allow
  auto-merge; a 502 on merge can leave "merge already in progress", so re-check before retrying.
  The archive branch on `jebel` is `archive-YYYYMMDD-HHMMSS`; delete it locally after the merge.
- **The addition harness's `crate` field hashes the crate's `__init__.py`**, not the `.so`: it
  doesn't identify the build (fixed in the length-generalisation workplan's stage 7). To check a
  venv's crate, hash the extension module.
- **The golden run** (`.venv/bin/python scripts/measurement/golden_training_run.py check
  data/refactoring/golden_run.json`; untracked, per machine) has 124 networks on both machines
  since #705, each version archived in `indrajala-benchmarks`; earlier versions are kept beside it
  as `golden_run.before-<change>.json`. The one tracked file there is `golden_run_rust.json`
  (the Rust entries, the same on every machine; CI checks it): after an approved re-record of the
  Rust bits, re-record it too (`record --backend rust`), or CI fails.
- **A test module's basename must be unique across `tests/` and `rust/tests/`:** neither is a
  package, so `./cli test` (both in one session) stops at collection on a clash, which CI's
  per-suite jobs can't see. `tests/test_suite_layout.py` checks it.
- **CI splits `tests/` into 8 shards by test, not by file** (`--shard I/N` in `tests/conftest.py`:
  every Nth test in collection order). A test must not rely on another test of its file having run
  first, and a module- or session-scoped fixture is built once per shard, not once in all. A shard
  run locally: `./cli test tests/ --shard 3/8`.
- **Crate builds:** `./cli build-rust`; after switching branches whose `rust/` differs, `git
  submodule update rust`, then rebuild.
- **A worktree's `.venv` may be a symlink to the main checkout's** (`ls -la .venv`): building a
  branch's crate into it replaces main's, and its editable install loads main's Python, so a
  golden run or suite there tests main's code against the branch's crate. A branch with a crate
  change gets a venv of its own (`python3 -m venv .venv`, `pip install -r requirements.txt
  maturin==1.15.0`, `pip install --no-deps -e .`, `maturin develop --release` in `rust/`; ruff
  and pyright at `./cli`'s pins for `./cli lint`); check with a one-line script in its
  `scripts/` that prints a module's `__file__`.
- **`litshe` fetches from GitHub anonymously** (the repos are public): its checkout's `rust/`
  submodule URL and its crate repo `~/code/indrajala-math-rust` (`ab.py`'s `CRATE_REPO`) both point
  at GitHub, so `ab.py remote --host litshe run --sync` and a crate A/B work unchanged. Cargo there
  runs one job (`~/.cargo/config.toml`), a single-core load under the owner's thermal limits.
- **Over a non-interactive ssh, `cargo` isn't on the PATH:** `maturin develop` on `jebel` fails
  with "rustc, the rust compiler, is not installed". `source ~/.cargo/env` first (`stone`'s
  `env.sh` adds it).
- **A machine's A/B data from before the root moved** (2026-10-23, from `~/code/ab-runs` and
  `~/code/ab-worktrees` to `~/data/indrajala-ml/runs/ab`) is invisible to the new `ab.py` until moved,
  with no A/B running: `R=~/data/indrajala-ml/runs/ab; mkdir -p $R/runs && mv ~/code/ab-runs/20*-* $R/runs/
  && mv ~/code/ab-runs/* $R/ && mv ~/code/ab-worktrees $R/worktrees && git -C ~/code/indrajala-ml
  worktree repair $R/worktrees/*`, then `rmdir ~/code/ab-runs`. Moved: `pyramidon`, `jebel` (2026-10-29,
  54 runs and 49 worktrees), `litshe` (2026-10-29); none left.
- **A host's first `ab.py remote run` from a checkout older than #1076** (its `scripts/ab.py`, before
  `scripts/` was grouped) fails after `--sync` with "No such file or directory:
  .../scripts/benchmark_machine_setup.sh": the old script syncs the checkout to main, then looks for
  the setup at its old path. The checkout is now at main, so run the same command again (jebel,
  2026-10-29).
- **Archiving a `litshe` run:** `ab.py remote --host litshe archive` fails (no archive clone there,
  and no GitHub credentials). Copy it here and archive it from `pyramidon`: `rsync -a
  litshe:data/indrajala-ml/runs/ab/runs/<run> ~/data/indrajala-ml/runs/ab/runs/`, then
  `ab.py archive <run>` (the run's profile is read
  from this checkout, its identity checked). On `jebel` the archive pushes its branch and fails at
  `gh pr create`: open the PR here with `-R davidbarkhuizen/indrajala-benchmarks --head <branch>`.
- **On `litshe` a worktree's `git submodule update --init rust` fails** with "transport 'file' not
  allowed" (its submodule URL is the local crate repo): `git -c protocol.file.allow=always
  submodule update --init rust`.
- **`scripts/data/fetch_datasets.py --help` fetches** (no argument parser); harmless when the files are
  present.

## Background jobs and the shell (for agents)

- **One waiter per job; trust only real notifications** (a system notification with a task id and
  output file). Check `ps` before calling a waiter gone or broken; an old session's waiter can
  notify a new session, and its notification is real too (read it, check what it watched).
- **A waiter's `pgrep -f` pattern matches its own command line**, and other waiters' too: anchor
  on the python process, `pgrep -f "^.venv/bin/python scripts/studies/window_recurrent_study.py run"` (in indrajala-studies).
- **An ssh that launches a nohup job holds on while the job runs** unless stdin is `/dev/null`;
  used on purpose it is the job's waiter (`setsid nohup ... < /dev/null`). Stopping a local
  background task doesn't kill its remote job.
- **A local `setsid nohup ... < /dev/null` returns at once** when the calling shell leads its
  process group (`setsid` forks): it is no waiter. Wait on the study's pid (`while kill -0 PID`).
  Over ssh it holds, as above. An ssh that starts a detached job with `&` holds on too: run that
  ssh in the background and kill only the local ssh (the remote job, `setsid`, survives).
- **After a local `setsid nohup cmd &`, `$!` is `setsid`'s short-lived parent, not the run**
  (`pyramidon`, from the Bash tool: the waiters on it exited at once); find the run with `pgrep -af`
  after launch. Over ssh to `jebel`, `$!` was the run.
- **`cd X && ... && cmd &` backgrounds the whole chain in a subshell:** variables set in it don't
  reach the next command, and over ssh the session stays open until the chain ends.
- **`pkill -f PATTERN` and `pgrep -f PATTERN` match their own shell** (its command line holds the
  pattern): 2026-10-20, a `pkill -f "probe-4a2.sh word-problem"` killed the Bash tool's shell and a waiter
  with it. Bracket one character (`pgrep -f "[p]robe-4a2.sh"`), which the pattern matches but its own
  text doesn't; or kill by the PIDs a bracketed `pgrep` lists.
- **A scheduled stop kills itself the same way:** `stone`'s stop for D55 ran `pkill -f
  "option1/probe-[d]55"` inside a `bash -c` whose own command line named `.../probe-d55/launch.log`
  (to write the "stopped" line), so the pattern matched its own shell: the runs stopped, but no
  "stopped" line was written. Put a scheduled stop in a script file (as `docs/probe-launches/stone/stop-d64.sh` at 40d66d9 was), so the
  command line that runs it holds no pattern.
- **Never chain a local launch after a remote one in one command:** when the launching ssh hangs
  (the line above), the Bash tool backgrounds the command and everything after the ssh runs when it
  finally returns (2026-10-20: a local probe launched again 1 h 48 min later, refused only by the run
  locks, its "ended" line fooling a waiter). Launch each machine in its own call; wait on a run's
  process (`kill -0 PID`), not on a line a second launch could write.
- **A Monitor that polls remote files works** (`ssh -n` a listing or `grep` each minute, the lines
  already seen kept in a bash associative array): used for the modular-family probe's step ends.
- **A Monitor reading `{ ssh host tail -F A & tail -F B; } | awk ...` delivered nothing** for 30
  minutes while both logs grew; a loop that polls each log's line count (`wc -l`, then `tail -n
  +N`, `ssh -n` for the remote one) every minute works.
- **`cmd &` inside a foreground call gives no completion notice;** use the harness's background
  mode. Foreground `sleep` is blocked.
- **`pkill -f <pattern>` kills its own shell** when the pattern is in the command line (exit 144):
  find the PID with `pgrep -af`, then `kill <pid>`. `pkill -f brave` likewise: use `pkill -x
  brave; pkill -x brave-browser`.
- **`$(...)` before `$?` resets it:** `echo "$(date) exit $?"` always says 0; capture `rc=$?` on
  the line after the command.
- **`rm -rf $VAR/...` is blocked by Claude Code's safety check:** write `"${S:?}"/...` or a literal
  path.
- **The shell's `grep` is ugrep** (a bare `{` is an error: escape it or use `-F`), and **`tail` is
  uutils** (`tail -3 a b` fails: `tail -n 3 a b`).
- **Downloaded files are untrusted data:** their own scratch directory, probe scripts elsewhere,
  `python3 -I`. Papers go to the literature repository (`davidbarkhuizen/indrajala-literature`,
  cloned beside this one).