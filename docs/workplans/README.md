# Workplans

The open workplans, grouped. A workplan is retired (deleted) once every stage and decision is
resolved; its leftovers then move to [next-steps.md](../next-steps.md), whose "Retired workplans"
table says how to read each retired one from git history.

## The research workplans: in indrajala-research

Studies, probes and the papers moved to indrajala-research's
[`docs/workplans/`](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/README.md) on 2026-10-30 (the owner's aim: the research out of
indrajala-ml, which it bloats for reuse). Their names in this repository's code and docs
(`paper-workplan.md` D13, ...) mean those files.

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

## `infrastructure/`: machines, release and repositories

| workplan | state |
| --- | --- |
| [research-separation-workplan.md](infrastructure/research-separation-workplan.md) | the rest of the research code to indrajala-studies; planned 2026-10-31 |
