# Workplans

The open workplans, grouped. A workplan is retired (deleted) once every stage and decision is
resolved; its leftovers then move to [next-steps.md](../next-steps.md), whose "Retired workplans"
table says how to read each retired one from git history.

## `research/`: the research programme

Studies, probes and the paper. The owner's long-run aim is to split the research out of
indrajala-ml (it bloats the core for reuse); this group is what would move.

| workplan | state |
| --- | --- |
| [paper-workplan.md](research/paper-workplan.md) | paper 1: released (tag `paper1-v1.1`, its datapack on Zenodo), out for verification |
| [post-paper-retests-workplan.md](research/post-paper-retests-workplan.md) | after paper 1: crystallisation (X), then branching and depth |
| [incremental-reheat-workplan.md](research/incremental-reheat-workplan.md) | stage 0 read; the build and its probe after paper 1 (D9), X's second arm declined (D8) |
| [state-formation-workplan.md](research/state-formation-workplan.md) | option 1 worked to a decision (the method); options 2-4 shelved (D31) |
| [general-algebras-workplan.md](research/general-algebras-workplan.md) | rounds 1-2 done; the fork's trigger and the guard to tune here (stages 5-7); paused since 2026-10-20 |
| [arbitrary-length-addition-workplan.md](research/arbitrary-length-addition-workplan.md) | a high-level plan; paused since 2026-10-20 |
| [factorisation-machine-workplan.md](research/factorisation-machine-workplan.md) | a high-level plan; paused since 2026-10-20 |

## `library/`: the ML primitives

The [primitives roadmap](../primitives-roadmap.md)'s steps, each planned (stage 1) and not yet built.

| workplan | roadmap |
| --- | --- |
| [generation-workplan.md](library/generation-workplan.md) | step 9: generation with a key/value cache |
| [training-recipe-workplan.md](library/training-recipe-workplan.md) | step 10: schedules, AdamW and gradient clipping |
| [segments-workplan.md](library/segments-workplan.md) | step 11: packing, document masks and padding |
| [context-length-workplan.md](library/context-length-workplan.md) | step 12: context length |
| [ffn-activations-workplan.md](library/ffn-activations-workplan.md) | step 13: GELU and SwiGLU |
| [grouped-query-attention-workplan.md](library/grouped-query-attention-workplan.md) | step 14: grouped- and multi-query attention |
| [cross-attention-workplan.md](library/cross-attention-workplan.md) | step 15: cross-attention and an encoder-decoder model |

Beside the roadmap:

| workplan | state |
| --- | --- |
| [dry-third-pass-workplan.md](library/dry-third-pass-workplan.md) | the third DRY pass (2026-10-30): audit done, D1-D10 settled, stages 1-4 done, stages 5-8 open |

## `infrastructure/`: machines and release

| workplan | state |
| --- | --- |
| [pypi-release-workplan.md](infrastructure/pypi-release-workplan.md) | deferred to the 1.0 release |
