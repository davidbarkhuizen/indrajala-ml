# Workplans

The open workplans, grouped. A workplan is retired (deleted) once every stage and decision is
resolved; its leftovers then move to [next-steps.md](../next-steps.md), and its row to
[retired.md](retired.md), which says how to read each retired one from git history.

## The research workplans: in indrajala-research

Studies, probes and the papers moved to indrajala-research's
[`docs/workplans/`](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/README.md) on 2026-10-30 (the owner's aim: the research out of
indrajala-ml, which it bloats for reuse). Their names in this repository's code and docs
(`paper-workplan.md` D13, ...) mean those files.

## `library/`: the ML primitives

Workplans for ML primitives, each planned (stage 1) and not yet built; the candidates are
[next-steps.md](../next-steps.md)'s infrastructure items.

| workplan | scope |
| --- | --- |
| [export-and-hub-workplan.md](library/export-and-hub-workplan.md) | **open, 2026-11-11:** format 3 (a directory: `config.json` and safetensors), the Hugging Face Hub (save, push, load by revision) and ONNX export of every shape, the window-recurrent network as its step; nothing built, the owner's O1-O3 open |

The roadmap's steps 1-15 and the array window (the machine-policy plan's M1) are done, their
workplans retired ([retired.md](retired.md)).

## `infrastructure/`: machines, release and repositories

| workplan | scope |
| --- | --- |
| [window-timing-guard-workplan.md](infrastructure/window-timing-guard-workplan.md) | **open, 2026-11-11:** next-steps' infrastructure item 1, one A/B for the window program (learn, forward and the long test) and a control that holds after a heavy case |
| [jebel-parked-workplan.md](infrastructure/jebel-parked-workplan.md) | **parked, 2026-11-05:** what's left of the crate performance plan, waiting for a healthy claim machine: the threshold re-sweep, the column split, new threaded sites and the closing baseline |

Retired plans are listed in [retired.md](retired.md).
