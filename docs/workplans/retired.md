# Retired workplans

A workplan is retired (deleted) once every stage and decision is resolved; what it left open moves
to [next-steps.md](../next-steps.md). Each one is still in git history, with its decisions (D1, D2,
...), design and stages, and code and docs cite them by section. Read one with the `git show` in its
row. The "roadmap step N" names are the steps 1-15 of the primitives roadmap, deleted
2026-11-11 once every step was built; the steps and the case for each are `git show
989819a:docs/primitives-roadmap.md`.

Until 2026-11-11, next-steps.md also carried a "Done" summary of each retired plan beside its
leftovers; that text is `git show 2490bdf:docs/next-steps.md`.

| Workplan | Plan PR | Stages | Read it with |
| --- | --- | --- | --- |
| Composable layers and optimizers (roadmap step 1) | #475 | #476-#485 | `git show 3a5d179:docs/composable-layers-workplan.md` |
| The A/B harness and a stand-alone measurement guide | #487 | #488-#491 | `git show 3a5d179:docs/ab-harness-workplan.md` |
| Batch normalization (roadmap step 2) | #486 | #493-#503 | `git show 189921c:docs/batch-norm-workplan.md` |
| Removing duplicated code (DRY) | #506 | #507-#514 | `git show e066333:docs/dry-workplan.md` |
| The conv batch-size study with batch norm | #516 | #517-#519 | `git show acbc49c:docs/conv-batch-norm-scaling-workplan.md` |
| Explicit generator objects (RNG generators) | #521 | crate #46, #522-#529 | `git show aaf8bb2:docs/rng-generators-workplan.md` |
| Residual connections (roadmap step 3) | #531 | crate #47, #532-#538 | `git show 365a173:docs/residual-connections-workplan.md` |
| Layer norm and single-head attention (roadmap step 4) | #540 | crate #48, #542-#548 | `git show 7d3bd0a:docs/layer-norm-attention-workplan.md` |
| Removing duplicated code, second pass (DRY rerun) | #550 | #551-#557 | `git show 88e1815:docs/dry-rerun-workplan.md` |
| Presets for the one-change combinations | #559 | #560-#564 | `git show 2a799bf:docs/presets-workplan.md` |
| The source layout | #566 | #567-#573 | `git show b362621:docs/source-layout-workplan.md` |
| A baseline on the new benchmark machine (i7-9700K) | #575 | #576-#590, #592 | `git show 480164d:docs/benchmark-machine-workplan.md` |
| A benchmark archive, tiered benchmarking, and a remote benchmark machine | #584 | #593-#598, archive #1-#11 | `git show 056b808:docs/benchmark-archive-workplan.md` |
| Multi-head attention and a transformer block (roadmap step 5) | #602 | crate #51-#52, #603-#613, archive #12-#13 | `git show d70cf0f:docs/multi-head-attention-workplan.md` |
| A sequence task with causal masking (roadmap step 6) | #620 | crate #53, #621-#634, archive #14-#17 | `git show f6e843b:docs/sequence-task-workplan.md` |
| Pure Python's random draws in numpy's order (RNG draw order) | #637 | #649-#652, archive #18-#19 | `git show 7a80858:docs/rng-draw-order-workplan.md` |
| Dropout in attention (roadmap step 7) | #638 | crate #54, #654-#660, #666, archive #20-#21 | `git show 8d2019e:docs/attention-dropout-workplan.md` |
| Length generalisation for the column adders (terminated by the owner before its existence study) | #683 | #684 | `git show c3d619b:docs/length-generalisation-workplan.md` |
| Rotary position embedding (roadmap step 8) | #639 | crate #55, #668-#669, #678-#682, #688, archive #22-#23 | `git show 9b27175:docs/rotary-positions-workplan.md` |
| Studies and timing runs sharing `jebel` (study-runs) | #670 | #673-#676, #695 | `git show 868838e:docs/study-runs-workplan.md` |
| The smallest network that adds (the addition study) | #661 | #662-#665, #672, #677, #696, models `7748416` | `git show 8990f74:docs/addition-study-workplan.md` |
| The demos in their own repository (the demo split) | #720 | #722-#724, indrajala-demos #1-#2 | `git show 9c50fe6:docs/demos-split-workplan.md` |
| The window loop in the crate (the window program) | #792 | #799-#802, crate #58-#59, archive #28-#30, #804 | `git show ad651d8:docs/window-program-workplan.md` |
| Substantiating the window-recurrent adder's findings (the claims workplan, C0-C8, E1-E8) | #701 | #703-#808 (findings #712-#808), models #1; report [window-recurrent-report.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/window-recurrent-report.md) | `git show fea9de1:docs/window-recurrent-claims-workplan.md` |
| The window-recurrent adder (milestone A; its claims in the row above) | #685 | #686-#702, #809 | `git show e45944e:docs/window-recurrent-adder-workplan.md` |
| A factored state for the window-recurrent network (C9, C9', C9'') | #737 | #738-#764, crate #57 | `git show e45944e:docs/factored-state-workplan.md` |
| A family of modular operations on one state (option 2) | #765 | #766 | `git show e45944e:docs/modular-family-workplan.md` |
| Modular division with the family (`p` 5 and 7, the registered claim) | #766 | #767-#782 | `git show e45944e:docs/division-workplan.md` |
| Where division fails in `p` (the division ladder, L1-L3) | #817 | #824, #856, #859, #861 | `git show f9101d8:docs/division-ladder-workplan.md` |
| A path through the window for each factor and for the answers (window branches; W1, W2, B1-B3) | #863 | #864-#868, crate #66, #884 | `git show e7f1ff6:docs/window-branches-workplan.md` |
| Consolidating codes (A) and longer words (B) for extra codes (P1, P2) | #822 | #829, #832-#836, #845, #849, #854, #857 | `git show df1dcac:docs/code-consolidation-workplan.md` |
| Controlling the codes' commitment (the diagnostic, the thermostat, annealing) | #765 | #851, #852, #871, #876, #881, #885, crate #67, #886 | `git show df1dcac:docs/commitment-control-workplan.md` |
| One code, one state (R1-R4, C0, C2) | #873 | #874, #875, #878, #879, #883 | `git show df1dcac:docs/one-code-one-state-workplan.md` |
| A population of runs selected by their machines (level 2, not built) | #882 | - | `git show df1dcac:docs/population-workplan.md` |
| An audit of the research programme (2026-10-23) | #953 | indrajala-research #48-#50, #954 | `git show 4ea65cc:docs/workplans/programme-audit-workplan.md`; the four reviews at indrajala-research's top level (`review-2026-10-23-*.md`) |
| The batch runner (restarts at a cutoff; a controller and a worker on each machine) | #935 | #943, #944, #946, #948, #952, #956 | `git show 53d7fef:docs/workplans/batch-runner-workplan.md` |
| The snap wall (an asymmetric margin on the snap; P1, C1 refuted) | #787 | #788, #789, #793, #797, #803 | `git show 53d7fef:docs/workplans/snap-wall-workplan.md` |
| The rest of the research code to indrajala-studies (D1-D9) | #1134 | #1135-#1138, studies #5, #6, research #131, this PR | `git show bbe51b4:docs/workplans/infrastructure/research-separation-workplan.md` |
| The studies' clean-moving code to indrajala-studies (D1-D11) | #1116 | studies #1-#4, #1131, #1132, research #130, this PR | `git show 6475d18:docs/workplans/infrastructure/studies-repo-workplan.md` |
| Publishing the crate on PyPI and crates.io, the first release 0.1.0 (D1-D11) | #1118 | crate #78-#84, #1119-#1121, #1127, #1129, this PR | `git show 995f0bd:docs/workplans/infrastructure/release-workplan.md` |
| The crate's kernel loose ends: the vector `exp`, the build pinned (D1-D7) | #1084 | #1089, #1093, #1095-#1097, crate #74, #76, bump #1088 | `git show 7046c98:docs/workplans/infrastructure/crate-kernels-workplan.md` |
| aarch64 on a Raspberry Pi 5, the crate bit-identical across architectures (D1-D10) | #721 | #722, #806, #810, #813, #842, #843, #846, #869, #880, #902, #926, crate #62-#65, #71 | `git show 6b9c6fe:docs/workplans/infrastructure/aarch64-workplan.md` |
| Generation with a key/value cache (roadmap step 9, D1-D9) | #640 | #1143-#1148, crate #85, #1149 | `git show 836e9a3:docs/workplans/library/generation-workplan.md` |
| The training recipe: schedules, AdamW, clipping (roadmap step 10, D1-D11) | #641 | #1150-#1156, crate #86, #1160 | `git show b446c31:docs/workplans/library/training-recipe-workplan.md` |
| Segments: packing, document masks and padding (roadmap step 11, D1-D11) | #642 | #1157, #1158, #1162-#1166, crate #87, #1159, this PR | `git show 69e2e93:docs/workplans/library/segments-workplan.md` |
| Context length: tiled attention, windows cut anew every epoch, the study (roadmap step 12, D1-D13) | #643 | #647, #1168-#1175, crate #88, #1178, this PR | `git show ec20e8c:docs/workplans/library/context-length-workplan.md` |
| FFN activations: GELU and SwiGLU (roadmap step 13, D1-D11; its study abandoned) | #644 | #1176, #1177, #1182-#1187, crate #90, #92, this PR | `git show dcedf28:docs/workplans/library/ffn-activations-workplan.md` |
| Grouped- and multi-query attention (roadmap step 14, D1-D13; its study abandoned for the literature) | #645 | #1189-#1197, crate #93, this PR | `git show 3a09724:docs/workplans/library/grouped-query-attention-workplan.md` |
| Paper 1: learning finite-state transducers through their approximations, with proof (D1-D39; retired 2026-11-03 after the preprint, doi:10.5281/zenodo.23826655) | indrajala-research's | T1-T12, R1-R9; datapack doi:10.5281/zenodo.23733575; tags `paper1-v1` to `paper1-v1.2` | `git show 0d1aaef:docs/workplans/paper-workplan.md` in indrajala-research |
| The implementation audit's remedies (§16; D1-D11) | #1202, #1203 | #1204-#1212, crate #94, studies #15, research #139, this PR | `git show 85d1705:docs/workplans/library/audit-remedies-workplan.md` |
| The crate's module structure (D1-D6) | #1238 | crate #97, #98, #1240 | `git show d79074a:docs/workplans/infrastructure/crate-structure-workplan.md` |
| Cross-attention: the encoder-decoder on Greek-English translation (roadmap step 15, D1-D18; its study decided from the literature) | #646 | #1219-#1236, crate #95, #96, this PR | `git show 2a512fb:docs/workplans/library/cross-attention-workplan.md` |
| Removing duplicated code, third pass (DRY, 2026-10-30) | #1100 | #1101-#1106, crate #77, #1108, #1109 | `git show d07a9e3:docs/workplans/library/dry-third-pass-workplan.md` |
| The crate's speed: kernels and threads as one plan (its jebel-dependent work parked) | #1243 | crate #100-#116, #1244-#1273 | `git show 4176d10:docs/workplans/infrastructure/crate-performance-workplan.md` |
| A window of array layers, a given gradient, stored states (the array window; indrajala-research's machine-policy plan, M1) | #1282 | #1283-#1285, crate #119 | `git show 32d4d23:docs/workplans/library/array-window-workplan.md` |
| One A/B guards the window program (learn, forward and the long test; next-steps' infrastructure item 1) | #1291 | indrajala-studies #21, #1292, #1304 | `git show ffea64a:docs/workplans/infrastructure/window-timing-guard-workplan.md` (its status names stage 2 as #1300: it is #1292) |

The optimization docs (the Rust-against-numpy baseline, and the implemented, rejected and
candidate optimizations) were retired the same way: `git show 0a04977:docs/optimizations.md` and
`git show 0a04977:docs/optimizations/<name>.md`.

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization, Residual connections, Layer norm and attention, The sequence task, Dropout among the
tokens, Rotary positions, Generation, Activations, The addition study; grouped key/value heads in Layer norm and attention), [measurement.md](../measurement.md)
and [rng-audit.md](../rng-audit.md). The batch-size studies' findings are in
`indrajala_ml/studies/batch_size_scaling.py`'s docstring, the depth study's in
`scripts/studies/residual_depth_study.py`'s, the patch-attention study's in
`scripts/studies/patch_attention_study.py`'s, the multi-head attention study's in
`scripts/studies/multi_head_attention_study.py`'s, the sequence study's in `scripts/studies/sequence_study.py`'s,
the attention dropout study's in `scripts/studies/attention_dropout_study.py`'s, the rotary study's in
`scripts/studies/rotary_study.py`'s, the context-length study's in `scripts/studies/context_length_study.py`'s, the training recipe's in `scripts/studies/lr_schedule_study.py`'s, the translation probe's in `scripts/studies/translation_probe.py`'s, the addition study's in indrajala-studies' `scripts/studies/addition_study.py`'s.

The crate performance workplan's stage 12 readings (threads, `litshe` with its active cooler, numpy
beside the crate) are in indrajala-benchmarks' `readings/2026-11-05-crate-threads/`.
