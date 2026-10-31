# Workplan: the studies' clean-moving code to indrajala-studies

**Status: planned 2026-10-30; decisions D1-D11 settled (D10, D11 and D5's third helper found in
stages 3 and 4, 2026-10-31); stages 2 and 3 done (indrajala-studies #1-#3), stage 4 in ml's move-out PR. Started after paper 1's stand-alone preprint
was published (the paper workplan's R9, in indrajala-research, done 2026-10-31). The scope is the
owner's: the clean moves only (2026-10-30).**

The goal: the research code that nothing left in indrajala-ml needs moves to a new repository,
[indrajala-studies](https://github.com/davidbarkhuizen/indrajala-studies), with its tests, so ml
carries less of what a reuser of the library never runs (the owner's long-run aim: the research
out of indrajala-ml). It is a first step, not the split: most of the window-recurrent research
code stays (below).

## Why only these

A file can leave ml only if nothing that stays imports it, by `import` or by path. Two pieces that
stay tie most of the research code to ml, and the owner left them out of this plan:

- `scripts/measurement/window_recurrent_timing.py`, an `ab.py` benchmark, imports the harness, its
  tasks, the modular family and `addition_data`;
- `tests/model/networks/test_window_program_network.py`, the network's test, imports the harness,
  its tasks, the certificate and the modular family.

Through their imports they hold 12 research modules in ml: `window_recurrent_study`,
`window_tasks`, `window_recurrent_certificate`, `window_diagnostics`, `modular_family`,
`machine_reading`, `inference`, `collapse`, `certificate_guard`, `indrajala_ml.batch`,
`data/addition_data` and `data/modular_data`. The batch runner also runs
`scripts/studies/modular_family_probe.py` from an ml worktree by path, so that script stays too.
Moved code may import ml freely (indrajala-studies depends on ml); only the reverse is forbidden.

## What moves

| from (ml) | to (indrajala-studies) |
| --- | --- |
| `indrajala_ml/studies/addition_study.py`, `window_drift.py` (D11) | `indrajala_studies/` (same file names) |
| `scripts/paper/` (all four: `paper_tables.py`, `paper_reading.py`, `datapack.py`, `zenodo_deposit.py`) | `scripts/paper/` |
| `scripts/studies/addition_study.py`, `window_recurrent_study.py`, `window_program_check.py`, `profile_page.py` and `profile_page.html` | `scripts/studies/` |
| `scripts/studies/configs/addition_calibration.json` and `addition_sweep.json` (D10) | `scripts/studies/configs/` |
| `scripts/data/migrate_factored_state.py` | `scripts/data/` |
| `tests/studies/test_addition_study.py`, `test_window_drift.py` | `tests/` |
| `tests/scripts/test_paper_tables_script.py`, `test_paper_reading.py`, `test_datapack_script.py`, `test_zenodo_deposit_script.py`, `test_window_recurrent_study_script.py`, `test_migrate_factored_state.py` | `tests/scripts/` |
| `tests/measurement/test_machine_lock.py`'s "addition run" row | a lock test of its own in `tests/` |

About 4,700 lines of code, plus the configs and their tests. Two test helpers are used on both
sides and are copied, not moved (D5): `tests/data/addition_adders.py` (also used by ml's
`test_addition_data.py`), and `tests/studies/test_window_recurrent_certificate.py`'s
`CarryMachine` and `TaskMachine` (used by the moved `test_migrate_factored_state.py`; the
certificate's test stays). `CarryTable` was copied too until `test_window_replay.py` stayed (D11).

**Stays in ml:** the 12 modules above and their tests; the batch runner (`indrajala_ml/batch.py`,
`scripts/runner/`) and `modular_family_probe.py`; the primitive-era studies (batch-size scaling,
sequence, rotary, patch, residual depth, attention dropout), which validate library features and
which ml's timing scripts and lock test import; the window-recurrent network and the crate's
window ops (the architecture); the 26 window-recurrent configs (D10); `chow` and `window_replay` and their tests (D11).

## Decisions

| ID | decision | settled | why |
| --- | --- | --- | --- |
| D1 | the repository | `davidbarkhuizen/indrajala-studies`, MIT, public from creation | the code it receives is already public under MIT in ml; MIT, not indrajala-research's CC BY 4.0, keeps it code-licensed; a separate repository keeps research's CI LaTeX-only |
| D2 | how it depends on ml | ml as a git submodule at `ml/` (recursive, so the crate comes with it), installed editable into the studies' venv; the crate built from `ml/rust` | the crate's own pattern: an exact pin in the tree. `common.provenance` records the working directory's commit and `dirty` (which a moved submodule sets), so a studies commit pins both. A pip git pin isn't editable and hides the pin from `git status` |
| D3 | layout and names | package `indrajala_studies/` (flat modules), `scripts/{paper,studies,data}/`, `tests/` (with `tests/scripts/`), a `cli` with `setup`, `test`, `lint` like ml's | the same shape as ml, so its habits and tools carry over; imports become `from indrajala_studies import addition_study` |
| D4 | history | copied, not filtered: the move's commit names ml's source commit; the files' history stays in ml | fast and simple; `git log` on ml at that commit reads the history |
| D5 | shared test helpers | copied to studies (`addition_adders.py`, `CarryTable`) | both sides need them; a test helper isn't in ml's installed package, and importing ml's `tests` package would collide with the studies' own |
| D6 | the tools, pinned the same | ruff, pyright and maturin at ml's `./cli` pins; `pyproject.toml`'s ruff and strict pyright settings mirrored | the same rules on both sides; a bump in ml is mirrored in studies' next submodule bump |
| D7 | CI | one lint job and one test job, x86_64 only, caching `.venv` and the crate's wheel keyed on `ml/rust`'s commit, as ml's CI does | the crate's cross-architecture parity is ml's and the crate's job; the moved tests need only one architecture |
| D8 | changes that need new ml code | merge in ml first, then a "Bump ml/" PR in studies | the crate's rule, one level up; no stacked PRs |
| D9 | paper 1's reproducibility | unchanged: the tags `paper1-v1` and `paper1-v1.1` hold the old paths for good, and the datapack's verify page names them | a published record never moves; a later paper version names studies' paths and tag |
| D10 | the window-recurrent configs | stay in ml (`scripts/studies/configs/window_recurrent_*.json`, 26); the moved `window_recurrent_study.py` runs them from `ml/scripts/studies/configs/`; the two addition configs move | ml's `tests/studies/test_window_recurrent_study.py` (the harness's `load_config` tests, which stay) reads them by path, so the plan's own rule keeps them; found in stage 3 (2026-10-31) |
| D11 | `chow` and `window_replay` | stay in ml with their tests; the moved scripts import them from `indrajala_ml.studies` | ml's `modular_family_probe.py` (the batch runner's, which stays) imports `chow`, and ml's `test_certificate_guard.py` imports `window_replay`; the stage 1 survey missed both. Found in stage 4 (2026-10-31); their copies left studies in #3 before ml's deletions |

## Stages

Each stage is one PR, merged before the next.

### Stage 1: this plan (ml)

### Stage 2: the repository (studies)

Create `indrajala-studies` (public, MIT, D1) with `README.md` (what it holds, setup, its relation
to ml and research), `LICENSE`, `pyproject.toml` (D3, D6), the `cli` (D3), `ml/` as a submodule at
ml's main (D2), `.gitignore`, the CI workflow (D7) and one smoke test: `indrajala_ml`,
`indrajala_math_rust` and `indrajala_studies` import, the crate from `ml/rust`. Done when its CI
passes and `./cli setup && ./cli test` passes on `pyramidon`.

### Stage 3: the move in (studies)

Copy every file in "What moves" at one ml commit, named in the commit message (D4); rewrite the
moved modules' imports (`indrajala_ml.studies.X` to `indrajala_studies.X`);
copy the helpers (D5); add the addition script's lock test. Each script's `sys.path` and
repository-root logic is checked from the new layout. Done when:

- every moved test passes in studies' CI, the same tests as in ml (counts compared);
- `paper_tables.py check` from studies on indrajala-research's manuscript reports all the same;
- `datapack.py` and `zenodo_deposit.py` still find indrajala-research as the sibling clone;
- one short `window_recurrent_study.py` run from a config matches the same run from ml by bits.

### Stage 4: the move out (ml)

Delete the moved files; drop the lock test's "addition run" row; point the docstrings and docs
that name the moved scripts at indrajala-studies (`window_recurrent_study.py` and
`window_recurrent_certificate.py`'s docstrings, `README.md`, `scripts/README.md`,
`docs/working-notes.md`, `docs/next-steps.md`). Tier 0 (nothing timed changes); the golden run
passes, since the package changed. Done when ml's CI passes and nothing in ml names a moved path
except history and the paper tags' records.

### Stage 5: the bump and the references (studies, research)

In studies, bump `ml/` to stage 4's commit: its tests still pass, so nothing moved depends on a
deleted file. In indrajala-research, the workplans', datapack docs' and README's live commands that
run moved scripts (`paper_tables.py check`, `datapack.py`, `zenodo_deposit.py`) name the studies
clone; records of what was run stay as they are. The handover and the owner's notes follow.
Then this plan retires to next-steps.md.

## After this plan

- The two pieces that tie the rest to ml (above), if the owner takes them up: a task fixture for
  the timing benchmark, and the network's test split into core checks and study checks. With
  both done, the 12 modules, the batch runner and `modular_family_probe.py` can follow.

## Out of scope

- Anything that stays (above), and any change to the batch runner.
- Rewriting the paper tags, the published datapack or its verify page (D9).
