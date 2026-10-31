# Workplan: the rest of the research code to indrajala-studies

**Status: planned 2026-10-31; decisions D1-D9 settled. The owner asked for the full separation the
studies-repo plan left (2026-10-31; `git show 6475d18:docs/workplans/infrastructure/studies-repo-workplan.md`,
"After this plan").**

The goal: ml keeps the library and what validates it; every piece of the window-recurrent research
moves to [indrajala-studies](https://github.com/davidbarkhuizen/indrajala-studies), which has ml as
its submodule at `ml/`. Moved code may import ml; ml never imports studies.

## What ties the research to ml

A survey of every import of the research modules from outside them (2026-10-31) finds two ties,
the ones the studies-repo plan named:

- `tests/model/networks/test_window_program_network.py`, the network's test: four of its 16 tests
  build the study's networks (`window_recurrent_study.build`, `modular_family.network`) and one
  certifies through `window_recurrent_certificate`;
- `scripts/measurement/window_recurrent_timing.py`, an `ab.py` benchmark: its learn and forward
  cases build the same networks; two cases (`long-test-family-*`, `study-family-*`) run the study's
  own long test and `modular_family.run`.

What both need from the study is mostly the network builders, and those builders are architecture:
`transformer_window(tokens, classes, width)` (W11), `linear_window` (E6's baseline), the FFN
window, `branched_window`, `recombined_window`, `answer_branch_window`, all functions of a token
count and a class count, not of a task. The family network is one of them with a factored
`WindowRecurrence` that reads the window's output.

## Decisions

| ID | decision | settled | options and why |
| --- | --- | --- | --- |
| D1 | where the window builders live | the library: `indrajala_ml/model/specs/window_presets.py`, functions of `(tokens, classes, ...)` with W11's constants (`D`, `BLOCKS`, `HEADS`, `FFN`); `window_recurrent_study` calls them | (a) **library presets**: one definition, the ties rebuilt from it, and the architecture where a reuser finds it ("one architecture for every machine"); cost: named presets (W11, E6) in the library. (b) copies in ml's test and benchmark: no library change, but two definitions that drift, and a bit check between them forever. (c) drop the study's cells from ml's test and benchmark: simplest, but ml loses its guard on the networks the research trains. (a) for extensibility |
| D2 | the network test's study tests | re-expressed in ml from the presets (the same cells: k 1, s 16, m 16 for each window; the family's shape, p 7, two factors of 32 codes, the output read, its token and class counts as constants); the originals, through the study's builders, move to studies with a check that `wr.build` and `mf.network` build the presets' networks by bits | (a) **both**: ml keeps the guard; studies proves its builders are the presets. (b) move them only: ml loses the cells. (c) keep them only in ml: the tie stays. The certificate test is study code (the certificate moves), so it moves |
| D3 | the timing benchmark | ml keeps every learn and forward case, built from the presets with random tokens of the same shapes (the time doesn't depend on them, as the benchmark already says), case names unchanged; the two end-to-end cases move to studies' `scripts/measurement/window_study_timing.py`, which ml's `ab.py` runs by absolute path (`--bench cmd -- /home/david/code/indrajala-studies/scripts/measurement/window_study_timing.py`, `_probe_path` takes one) | (a) **split**: ml's rows continue, studies keeps the end-to-end Python cost (the window-program workplan's D10 gate) under ab.py. (b) move the whole benchmark: ml's crate changes lose their most relevant window benchmark. (c) keep it whole: the tie stays. One tier-1 A/B on jebel checks that no kept row moves |
| D4 | what moves | `indrajala_ml/studies/`: `window_recurrent_study`, `window_tasks`, `window_recurrent_certificate`, `window_diagnostics`, `modular_family`, `machine_reading`, `inference`, `collapse`, `certificate_guard`, `chow`, `window_replay`; `indrajala_ml/batch.py`; `indrajala_ml/data/addition_data.py`, `modular_data.py`; `scripts/studies/modular_family_probe.py`; `scripts/runner/`; the 26 window-recurrent configs; their tests and helpers | everything the survey finds only research importing. Stays: `studies/common.py` (the primitive-era studies use it), the primitive-era studies (they validate library features; ml's timing scripts and lock test import them), `machine_lock`, `ab.py` |
| D5 | names in studies | flat modules in `indrajala_studies/` (`indrajala_studies.window_tasks`, `indrajala_studies.batch`, `indrajala_studies.addition_data`); scripts and tests at their ml paths under studies' `scripts/` and `tests/` | the studies-repo plan's D3 layout. A `data/` or `studies/` subpackage would mirror ml's split, which only made sense inside the library |
| D6 | the batch runner | runs from a studies worktree at a studies commit (which pins ml through `ml/`), `PYTHONPATH` the crate, the worktree, then `worktree/ml`; the hosts keep their venvs (the same dependencies); a batch records the studies commit and the submodule's | (a) **studies worktrees**: one commit pins both. (b) two worktrees per host (studies and ml): two commits to keep in step. (c) keep the runner in ml, running studies' probe: the tie reversed, ml naming studies' paths. Checked by a short real batch on jebel before ml's copy goes |
| D7 | history | copied, not filtered: the move's commit names ml's source commit | the studies-repo plan's D4 |
| D8 | pyramidon's study data | moved from `~/code/indrajala-ml/data/` (`window-recurrent/`, `addition/`, `studies/` where they are window-recurrent's) to `~/code/indrajala-studies/data/` (gitignored there too), the manifests checked after; one `mv` on one disk | the study scripts write and read cwd-relative `data/`, so runs from studies find their data without `--data`. The alternative, a symlink, leaves two names for one tree |
| D9 | paper 1 | unchanged: its tags hold the old paths for good (the studies-repo plan's D9) | a published record never moves |

## Stages

Each stage is one PR, merged before the next.

### Stage 1: this plan (ml)

### Stage 2: the window presets (ml)

`indrajala_ml/model/specs/window_presets.py` holds the window builders and W11's constants, moved
from `window_recurrent_study` (D1), which imports them back under its old names so nothing else
changes. A test builds each preset's network beside the old builder's at one commit and compares
their snapshots by bits; the golden run passes. Tier 0: no timed path changes.

### Stage 3: the ties cut (ml)

- The network test's study tests re-expressed from the presets (D2); the originals moved to
  `tests/studies/test_window_program_studies.py`, which moves with the rest in stage 4.
- The timing benchmark's learn and forward cases built from the presets (D3); the end-to-end cases
  moved to `scripts/measurement/window_study_timing.py` beside it, which moves in stage 4.
- One `ab.py` A/B of `window_recurrent_timing.py` on jebel, old against new (tier 1, `--order ONNO`):
  no kept row moves.

Done when nothing outside the D4 set imports it (the survey's command, rerun).

### Stage 4: the move in (studies)

Copy the D4 set at one ml commit (D7), with stage 3's moved test and benchmark, into D5's names;
rewrite the imports; replace the studies-repo plan's copied helpers with the originals; adapt the
batch runner to studies worktrees (D6); shard studies' test job if it passes 10 minutes (ml's
`--shard`). Done when:

- every moved test passes in studies' CI, the same tests as in ml (counts compared);
- `paper_tables.py check` reports the manuscript all the same;
- one short `modular_family_probe.py` run from studies matches the same run from ml by bytes;
- one short batch on jebel from a studies worktree completes and records its commits (D6);
- `ab.py` runs `window_study_timing.py` by absolute path on jebel.

### Stage 5: the move out (ml)

Delete the D4 set and stage 3's moved files; point the docs, docstrings and working notes at
studies (the runner's notes move to studies' own); the golden run passes. Done when ml's CI passes
and the survey finds nothing of D4 in ml but history.

### Stage 6: the bump, the data and the references (studies, research, pyramidon)

Bump studies' `ml/`; its tests pass. Move pyramidon's study data (D8) and check the manifests.
indrajala-research's live pointers name studies. The handover and the owner's notes follow. Then
this plan retires to next-steps.md.

## Out of scope

- The primitive-era studies (they validate library features).
- Rewriting the paper tags, the published datapack or its verify page (D9).
