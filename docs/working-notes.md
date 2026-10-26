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
  for `scripts/golden_training_run.py` too. To prove a test fails without a fix, revert the fix in
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
- **Python 3.13+ dataclasses have `__replace__`**, so pyright infers a PEP 695 dataclass's type
  parameter invariant; disjoint `@overload`s avoid needing covariance.
- **`load_network` returns `Any`**: pyright won't flag a `None` attribute on a loaded network;
  grep for it when touching code that loads networks.
- **The venv has no scipy:** Fisher's exact test by hand (`math.comb`), or add the dependency in
  its own PR.

## Studies: running them

- **Long jobs never swap** (measurement.md section 2): size workers from the largest of a probe's
  peak, a recorded `peak_mb` and a worker's RSS a few minutes in; check `free -m` and `vmstat`
  (`si`/`so`) after the first minutes. **One memory-heavy job at a time on `pyramidon`** (5.3 GB):
  a `k` 1 long test on an E8c network grows past 1.6 GB at 1,024 columns, and with a `k` 4
  `certify` (about 620 MB) beside it the machine swapped. A `certify` pool worker at `k` 4 takes
  about 440-620 MB though one in-process probe measured 45 MB; at most 3 such workers on
  `pyramidon`.
- **`/tmp` on `pyramidon` is RAM-backed (tmpfs) with a per-user quota:** scratch copies of study
  data cost memory, and a test writing many run directories filled the quota (2026-10-18). Remove
  each one's files as it goes (the resume test does).
- **Training state set on the network isn't saved with it** (temperatures, couplings, continuous
  factors, commitments): a resumed run loads the network bare, so whatever its first screen reads
  before the next epoch sets them must be kept in the checkpoint's `Place` and re-applied (the
  couplings were not until 2026-10-21, found by the adaptive cool's resume test). A resume test whose
  factors harden at once (coupling 1) can't see such a gap: give a new profile's test soft factors.
- **Epochs bind a run; a wall clock never does** (the programme audit's F12): the core matrix (D46,
  2026-10-21) chose an epoch budget because "a minutes budget would give a run's outcome its
  machine"; the probes registered that evening onward (D54-D64) still used "1,600 epochs or 690
  minutes", and on a loaded `stone` the clock cut ten runs short (D67 re-ran them). `--budget-epochs`
  is required now; set `--budget-minutes` only as a runaway guard far above any normal run, and
  register its stop as censored.
- **A run its wall-clock guard stopped resumes:** `modular_family_probe.py run` with a "deadline"
  stop keeps the last epoch's checkpoint (not "ended"), so the same command with `--resume` and a
  larger `--budget-minutes` goes on to its epoch budget as the uninterrupted run (the deadline's
  rows and the cut step's files dropped); the guard isn't part of the run's identity. Runs cut
  before this change (D67's nine) ended their checkpoints and can't: re-run them from their seeds.
- **Background Python writing to a log is block-buffered:** a killed job leaves nothing. Use
  `python -u` or `PYTHONUNBUFFERED=1` for any log you may need mid-run.
- **The harness log prints nothing within a step;** progress shows as `models/<key>/step-<j>.json`
  files, and `status` (safe over ssh while a study runs) counts them. **To see whether a run is
  going anywhere, `diagnose CONFIG --run KEY --follow`** beside it (one core, about a second a
  step): each step's accuracy per answer class and each factor's share of the target's
  coordinates, as each step is saved, until the study ends. Write the stop rule into the study's
  log before the first diagnosis, not after (C9' seed 5).
- **A stopped run doesn't resume:** its last step is saved with stop "stopped" and its result
  written, so `run` skips it and `train_run` would end at that step again. To resume one run (within
  its study's budget), move its `results/<key>.json` and its stopped step's `step-<j>.json` and
  `.network.json` aside (not delete), and the stop file; `run` then trains only it, from step
  `j - 1`'s network.
- **A registered stop check runs at its time, never behind a question to the owner:** C9''s 2 h
  rule waited on an unanswered question and both machines ran 2 h more. Apply the rule, then ask.
- **Don't `git pull` on a machine while a study runs there:** its forkserver may import changed
  modules.
- **Stopping a study** (only with the owner's say): kill its Python process's whole process group,
  `kill -TERM -- -<pgid>` (`ps -eo pid,pgid,args`); the python, its resource tracker and forkserver
  workers share it. Killing a bash wrapper's group leaves the workers training after the lock is
  freed. `status` then shows stale "running" entries until the next `run`.
- **`certify` and `long-test` don't skip work already done:** to resume, pass only the missing
  networks.
- **An existence study's certified count is "at least":** parallel runs can certify in the same
  epochs before the stop reaches them.
- **Training past a certified step can destroy the network** (E8b): existence tests use
  `certify_steps` and `stop_at_certified`.
- **A state can collapse to one code at long training lengths:** a fresh `m` 16 network starts on
  1-2 codes; lengths 1-3 or 1-6 split it, 1-10 or 1-20 don't. A 2-epoch probe (about 20 s an epoch
  at 20 columns) shows it before a long launch.
- **A hand-built training set needs the harness's final zero window** (`window_recurrent_study.
  example`): its answer needs the state the data left, so even a one-column step trains the state.
  Without it the modular-family probe's step 1 passed on the column alone and its state stayed on
  one code in every arm.
- **Every probe stops at its first certified step** (certify each step's end: under a second at
  `k` 1): all four of the modular-family runs that solved or nearly solved division collapsed when
  trained on (E8b's warning, seen again). Save the network before it is overwritten.
- **`modular_family_probe.py run` with one `--task` writes `DIR/seed<s>`**, not
  `DIR/<task>/seed<s>` as several tasks do: read and `readout` such a run as `--arms .`.
- **To probe beside a running study on `jebel`, use a worktree there** (`git worktree add --detach
  ~/code/wt-X origin/<branch>`, `.venv` and `data/window-recurrent` linked, `PYTHONPATH=$PWD`); don't
  update its files while a harness study (forkserver workers) runs from it. A single-process probe
  has its modules loaded and doesn't care.
- **When trimming a stage, check which later experiments compare against the dropped cells**
  (D34 removed E2's and E6's `k` 1 comparators).
- **Copying results between machines:** `sha256sum` on the source, `rsync`, `sha256sum -c` on
  the destination; record the manifest's own sha256 in the findings. **Don't rename or re-root on
  the copy:** a host's manifest lists its own paths (`klip`'s `data-d64/probe-d64/...`), so a file
  renamed on arrival (`launch.log` to `launch-klip.log`) or a different root fails `sha256sum -c`
  on the archive (the audit's F31). Keep each host's files under its own directory as listed, or
  write an archive-side manifest after the copy (`HOST-option1-dNN.archive.sha256`).
- **Manifests' base directories differ:** `X.sha256` beside a directory `X` lists paths inside it
  (`e8-q4.sha256`, run from its parent); `certificates.sha256` inside `e8-q4/` lists bare names
  (run from `certificates/`); `control/results.sha256` and `matrix/results.sha256` list
  `results/...` (run from `control/`); `e8c-q*.sha256` list `e8c-q4/...` (run from
  `data/window-recurrent`).

- **Two runs at once on one machine each need their own lock:** `run` holds `$AB_RUNS_ROOT/run.lock`,
  so a second run started with the same `AB_RUNS_ROOT` is refused ("not started: ... run.lock is held by
  study ...") before it writes data. Give each concurrent script its own `AB_RUNS_ROOT`.
- **`certify --out` is the certificate's file path, not a directory:** a second verdict written to the
  same path overwrites the first. Name them `certify/<task>-seed<s>-certified-step-<n>[.strict].certificate.json`
  (the strict verdict with `--snap-ratio 1.05`), with `certify.log` beside them.

### The batch runner (`scripts/batch.py`, `indrajala_ml/batch.py`)

- **The ledger's commit must be the full sha:** workers report theirs in full and the controller
  compares exactly, so a short `--commit` refused every worker (2026-10-23, T1's first start; nothing
  ran). `register` now expands it.
- **Stop a fleet by its process group** (`kill -- -PGID`; it runs under `setsid`): killing the fleet's
  pid alone leaves its `ssh -R` tunnels holding the remote port, and the next fleet's tunnels fail with
  "remote port forwarding failed" (harmless while the old ones forward to the same port, but they
  outlive it).
- **A worker can connect to itself** (Linux's TCP self-connect): while its tunnel is down, a retry to
  127.0.0.1:PORT given PORT as its own ephemeral port connects to itself, and its messages go nowhere.
  T10 (2026-10-25/26): `stone` off the network for about 20 hours, its worker self-connected, its 11
  ends reported into the void (dropped from its memory), and the tunnel, the IPv4 port taken, bound
  `[::1]` alone; the controller held them as running. Recovered by stopping that worker, reopening the
  tunnel and replaying each `ended` from its `ended.json` (the controller's own pull and check). Now
  the worker refuses a connection to itself, the tunnel binds `127.0.0.1` (or fails and retries), and
  the controller acks each `ended`: a worker resends the unacknowledged ones (`ROOT/acked/`) on its
  next connection, listed in its hello so they aren't lost, and leaves a drain only once every end is
  acked. A worker at this commit needs a controller at it (an older one never acks: no drain ends).
- **The crate is pinned too** (D6): `register` records the crate extension's SHA-256 (`--crate`, default
  the registering machine's) and the controller refuses a worker that imports another. Builds of one
  source differ by machine, so every host imports one build: `b.CRATE` (`crate-647f23c`), the package
  directories copied from `pyramidon`'s venv into `~/code/crate-647f23c` (`~/labour/` on `stone`) and put
  first on PYTHONPATH by the host table. A new crate: a new directory on every host, `CRATE` changed. T1 ran before the check: `pyramidon`'s crate (rust 647f23c, whose one change is
  aarch64-only NEON) differed from `klip`'s and `stone`'s (45d6a37); `same-prefix` on `d5-sum-304` gave
  31 identical rows across them (2026-10-23).
- **A lost attempt re-run on its host appended to its old log** (T1: `d5-sum-300`, `d5-sum-302`,
  `d5-suffix-302`, on `pyramidon`; the ledger summed both runs, 662, 681 and 596 epochs for 400). Now
  the worker sets the old directory aside (`ROOT/lost/<attempt>-<n>`, never adopted) and `summary`
  reads a log's last run only (a step end not after its predecessor's starts one), so re-summarising an
  older ledger's logs corrects it. Ledgers written before the fix still hold the summed summaries.
- **An attempt's memory is the worker's sample, not `/usr/bin/time`:** a run's pool workers are a
  forkserver's children, outside the attempt's rusage (`time -v` saw 51 MB of a division run). The
  worker sums PSS over the attempt's session at each look (every 2 s) and puts the peak in its summary
  (`peak_pss_mb`); a spike shorter than a look can be missed. **`klip`'s peaks before the fix are
  wrong** (3.8-6.5 MB on T10's probe against `pyramidon`'s 184-221 MB): its thermal guard `setsid`s the
  run, so the session held only the wrapper. Now the sum follows every descendant of the session's
  processes too (`attempt_pss`); use another host's peaks for `klip`'s earlier attempts.
- **Several cells in one batch:** `register --cell A --cell B ...` runs the spec under each cell in
  turn (list the longest first: the fleet takes attempts in order), the ids `<cell>.<backdrop>-<seed>`
  so the cells' runs never share a directory; `find` counts `k` per cell and backdrop. A ledger from
  before (one `cell` for the batch) is read as each attempt's, and the attempt message is unchanged,
  so a controller at a newer commit serves workers pinned at an older one.
- **A host given outside its hours:** `controller --open HOST` (the owner's grant, logged in the
  ledger as `opened`); the host table is unchanged.
- **The controller's own host's attempts weren't copied into the batch** (T2/T5, T10's probe): only
  remote hosts were pulled, so `pyramidon`'s attempts stayed in `~/code/batch-attempts/` with no
  `pulled` event, and a reading of `attempts/` missed them (T2/T5's 15, copied by hand). Now every
  host's ended attempt is pulled, the local one by rsync without ssh, and checked against its
  manifest. A controller started before the fix still needs the hand copy (excluding `ab-runs`).

## Studies: reading them

- **Test window networks at full catalogue sizes:** reduced samples passed lengths the full sizes
  fail on a near-working network. `check` uses the full default catalogue; to time other sizes,
  call `wr._evaluate(model, n, wr.test_properties(...))`.
- **`tests_past_break: false` leaves lengths untested** (the report shows `-`), and codes used then
  count only the lengths tested: at break 0 it is the initial state's code alone. To see collapse,
  run `state_codes` on real examples (`wr.example`).
- **The checks are exhaustive at short lengths** (B1 at `n` 4 is all 531,441 triples); a check at
  12-20 columns costs minutes (187 s at 12 in E8b's probe): budget it.
- **`certify` branches near ties by default** ("robustly certified"); `--snap-ratio 1` gives the
  strict verdict ("certified"). `wrong_ties` counts only wrong carries on a tie's own branch: a code
  reached only through a tie can carry wrong on its own windows, so read the strict certificate
  for "wrong as computed".
- **`certified` = no wrong output, no wrong state, no output near tie;** `correct` without
  `certified` means an output near tie only. A carry counterexample shows as a wrong sum only
  through the next window and only at alignment 0, which the harness's hash picks for few cases:
  feed it with `wr.example(triple, n - k, k, 0)`, not `WindowAdder`.
- **The certificate is fast at `k` 1** (about 2 s a 40-network pass) and slow at `k` 4 (minutes a
  network).
- **The long-test JSON:** `lengths -> n -> properties -> id -> {cases, passed, failures}`;
  `modular_family_probe.py long-summary DIR` reads a study's (`long-test --out DIR/<run>`): a line
  per run, its failures and the network's sha256, and one for all.
- **`WindowRecurrence.codes` is None for a continuous network:** `state_codes`, `first_step` and
  `window_step` assert "no codes" on one.
- **Results carry `window`** (#717) and may carry `m: null`: use `comparable`'s defaults (no
  `window` = transformer; no `task` = addition).
- **`addition_data.catalogue()` returns new Property objects:** `screen_cases` caches per Property,
  so build the catalogue once per step.
- **A modular-family step stopped by a budget logs one epoch too many:** its end row's `epochs` is
  the loop's index at the stop (`spent`, `deadline`), not trained, so a run of `--budget-epochs
  400` sums to 401. `modular_family.outcome` (the `readout`) corrects it; sum the rows by hand and
  you must too.
- **A study's end can't be read from its average rate:** later ladders are slower; quote a range.
- **Check every figure against the data before writing it into a doc.**
- **The addition study's harness:** `train_seconds` include the whole catalogue whenever a screen
  passed, so a short passing run's seconds per epoch look several times the true rate; its worker
  processes run the module's `CANDIDATES` only; `patch_study.parameter_count` is numpy-only; don't
  test its catalogue's generalisation at `n` 2; a screen passing isn't a pass.

## Code internals

- **The window study's API** (`indrajala_ml/studies/window_recurrent_study.py` as `wr`):
  `wr.example(task, case, n, k, a)`, `wr.alignment(task, case, n, k)`, `wr.build(task, k, s, m,
  seed)`, `wr.window_specs(task)`, `wr.WindowModel(network, k, task)` with `.evaluate(n,
  properties[, limit])` and `.predict(n)`; `wr.check_properties(task, n, cat)`,
  `wr.test_properties(task, cat)`; `settings.catalogue(task)`.
- **The certificate's JSON names:** `wrong_outputs`, `wrong_states`, kinds `outputs`/`state`,
  `state_in`, `case`, `after`, `task`. E1's files keep the older `wrong_digits`, `wrong_carries`,
  `digits`/`carry`, `carry_in`, `addition`.
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
- **Never `x // BASE**t` (or `d * BASE**t`) for every digit t of a long number:** each power and
  division is a big-integer operation the size of the number, so a conversion is quadratic in the
  length. At the long tests' 1,024 columns it was a quarter of a long test (the window-program
  workplan's stage 1). Use `addition_data.digits` and `from_digits` (DIGIT_CHUNK digits a
  big-integer step, the rest from a table or small ints).

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

- **Development on `pyramidon`, timing and long studies on `jebel`** (measurement.md section 4).
  Before a run on `jebel`: `uptime`, its processes, `pgrep -f '^/usr/bin/python3
  /usr/bin/unattended-upgrade'` and `pgrep -x dpkg`.
- **`jebel` at or below 65 W.** After a reboot PL1 comes back at 95 W: `ssh jebel 'cd
  ~/code/indrajala-ml && .venv/bin/python scripts/ab.py check --apply-setup'`, then check
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
- **`klip`, the fourth machine: studies only.** `ssh klip` (192.168.1.104); Ryzen 7 3700U laptop
  (4 cores, 8 threads), 5 GB, so about 3 small runs at once. Always on, all cores; the owner's
  limit (2026-10-21): <= 85 C, ideally <= 80 C. **CPU boost is off** (`cpu-boost-off.service`;
  `/sys/devices/system/cpu/cpufreq/boost` reads 0): boost overheats it. **Run every job under the
  thermal guard** `~/code/thermal-guard.sh LOG [STOP] -- COMMAND...`: it starts the command in its own
  process group, logs the k10temp sensor every 5 s to LOG and stops the group at STOP C (85) or
  above, with an "ended, status N, peak T C" line. A waiter there watches the probe's "probe ended" line
  or the guard's process gone (`pgrep -f "[t]hermal-guard.sh <its LOG's directory>"`).
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
- **The golden run** (`.venv/bin/python scripts/golden_training_run.py check
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
  worktree repair $R/worktrees/*`, then `rmdir ~/code/ab-runs`. Moved: `pyramidon`; `jebel` and
  `litshe` still to move (`jebel` after D67's runs end, with its checkout's pull).
- **Archiving a `litshe` run:** `ab.py remote --host litshe archive` fails (no archive clone there,
  and no GitHub credentials). Copy it here and archive it from `pyramidon`: `rsync -a
  litshe:data/indrajala-ml/runs/ab/runs/<run> ~/data/indrajala-ml/runs/ab/runs/`, then
  `ab.py archive <run>` (the run's profile is read
  from this checkout, its identity checked). On `jebel` the archive pushes its branch and fails at
  `gh pr create`: open the PR here with `-R davidbarkhuizen/indrajala-benchmarks --head <branch>`.
- **On `litshe` a worktree's `git submodule update --init rust` fails** with "transport 'file' not
  allowed" (its submodule URL is the local crate repo): `git -c protocol.file.allow=always
  submodule update --init rust`.
- **`scripts/fetch_datasets.py --help` fetches** (no argument parser); harmless when the files are
  present.

## Background jobs and the shell (for agents)

- **One waiter per job; trust only real notifications** (a system notification with a task id and
  output file). Check `ps` before calling a waiter gone or broken; an old session's waiter can
  notify a new session, and its notification is real too (read it, check what it watched).
- **A waiter's `pgrep -f` pattern matches its own command line**, and other waiters' too: anchor
  on the python process, `pgrep -f "^.venv/bin/python scripts/window_recurrent_study.py run"`.
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
  "stopped" line was written. Put a scheduled stop in a script file (`~/labour/stop-d64.sh`), so the
  command line that runs it holds no pattern.
- **Never chain a local launch after a remote one in one command:** when the launching ssh hangs
  (the line above), the Bash tool backgrounds the command and everything after the ssh runs when it
  finally returns (2026-10-20: a local probe launched again 1 h 48 min later, refused only by the run
  locks, its "ended" line fooling a waiter). Launch each machine in its own call; wait on a run's
  process (`kill -0 PID`), not on a line a second launch could write.
- **To run a study from a worktree's code into the main checkout's data** (so `main` can be pulled
  while it runs): `cd` to the main checkout, `PYTHONPATH=<worktree>`, and the script and config by
  the worktree's path. `indrajala_ml` is a namespace package, so check which copy loads (a
  one-line script in the worktree's `scripts/` printing a module's `__file__`): run as a script,
  the worktree's loads; `python -c` from the main checkout loads the main checkout's. E2 ran this
  way (`~/code/wt-e2`, launched by `~/code/e2-launch.sh` on `jebel`).
- **Never develop in a worktree a queued or running study imports from:** a launcher that waits
  imports the worktree as it is when the study starts (and records its commit then). Keep a separate
  worktree for development (`~/code/wt-dev`); `~/code/wt-e2b` on `pyramidon` was nearly launched on a
  development commit.
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
- **A `mkdir` claim over ssh isn't a perfect lock:** stage 5's shared long tests
  (`~/code/stage5-long-tests.sh`, each worker claiming a network by `ssh pyramidon mkdir ...`) had
  two of `jebel`'s workers start the same network in the same second (2026-10-16), cause not found.
  Harmless for an idempotent test (one result overwrote the other); for anything that isn't, have
  each worker take a fixed share instead.
  Worse, a queued study whose script waits on `pgrep -f <the previous script's name>` also waits
  on a waiter watching that name. On 2026-10-18 that held `jebel` idle for hours after round 2
  ended (the waiter's `while pgrep -f algebras-round2.sh` never ended, so neither did the
  diagnostic's). Wait on a log line instead (`until grep -q "ended" .../launch.log`), and have each
  launch script write one.
- **A second batch beside a running one on the same host needs its own code, root and port**
  (2026-10-25, T11 on `klip` beside T10): `prepare` would move the host's `wt-batch`, which the running
  batch's worker loads code from for every new attempt; and a worker adopts every running attempt
  in its root from their pid files (`Worker.adopt`), so a second worker in the same root takes the
  first batch's. So: a worktree of its own at the batch's commit (`~/code/wt-batch-<name>`), a root
  of its own (`~/code/batch-attempts-<name>`), its controller on another port with the tunnel by hand
  (as `batch.tunnel`; it doesn't reopen itself), and the worker started by hand (as
  `batch.worker_line`). A controller without `--hosts` assigns to any host and **pulls nothing**
  (`Controller.fits` and `ended` skip a host not in it): rsync the ended attempts in by hand.
- **A direct `modular_family_probe.py run` takes the machine's run lock:** a second one started beside
  it refuses ("not started: ... run.lock is held"); run several as one `--runs` pool, or one after
  another. Batch attempts don't contend for it.
- **A long test or study run from a worktree with `PYTHONPATH` still takes the main checkout's
  `.venv`, its compiled crate included:** check `git diff --stat <pinned> <main's> -- rust/` is empty
  before trusting the crate hash the results record.
