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
| [ffn-activations-workplan.md](library/ffn-activations-workplan.md) | step 13: GELU and SwiGLU |
| [grouped-query-attention-workplan.md](library/grouped-query-attention-workplan.md) | step 14: grouped- and multi-query attention |
| [cross-attention-workplan.md](library/cross-attention-workplan.md) | step 15: cross-attention and an encoder-decoder model |
| [audit-remedies-workplan.md](library/audit-remedies-workplan.md) | the implementation audit's remedies (§16): the codebook against collapse, an explicit output loss, a selection set, He initialisation, a FiLM state input, MRI |

## `infrastructure/`: machines, release and repositories

| workplan | scope |
| --- | --- |
| [multithreading-workplan.md](infrastructure/multithreading-workplan.md) | the crate's outstanding kernel work (stage −1), then a thread pool, one thread budget a process, and threaded attention, conv and the window program |

Retired plans are listed in [next-steps.md](../next-steps.md).
