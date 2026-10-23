# Workplan: an audit of the research programme (2026-10-23)

**Status: written 2026-10-23, at the owner's request: "conduct a detailed audit of our research
programme and findings so far. Out of this I would like 1. a summary document(s) covering the
evolution of the research programme 2. a summary of logical inconsistencies, gaps and bad
assumptions 3. a list of audit findings"; and, mid-planning, "also re-assess the existing reviews,
once you are done with this one - in light of any audit findings". D1-D6 settled under the owner's
delegation, each with its options' pros and cons.** Offline work: it runs beside D67's completions
and D63's moved runs and reads none of their partial data (the registered-probe rule).

## What is audited

The research programme: the line of work from the addition study (2026-10-08) to the state-formation
workplan and the batch runner (2026-10-23): its questions, claims, evidence, rules and decisions, and
how they changed. Not audited: the library's engineering workplans (the primitives roadmap's steps,
generation, segments, the aarch64 parity work, PyPI), except where a study's result depends on them
(determinism, bit parity, the RNG audit).

## Sources, in tiers

| tier | what | read |
| --- | --- | --- |
| A | indrajala-research: both papers, the four `.tex` research notes, `contribution.md`, the dated reviews, `docs/protocols/`, `docs/research-notes/` | in full |
| B | indrajala-ml: `docs/workplans/state-formation-workplan.md` (D1-D67), `batch-runner-workplan.md`, `general-algebras-workplan.md`, `snap-wall-workplan.md`, `factorisation-machine-workplan.md`, `arbitrary-length-addition-workplan.md`; `next-steps.md`, `working-notes.md`, `measurement.md` (its study sections); `docs/audits/` | in full |
| C | retired workplans through `git show` (the claims workplan at fea9de1, one-code-one-state and commitment control at df1dcac, the window-recurrent adder at e45944e, division and the factored state), the study scripts' findings docstrings (`scripts/window_recurrent_study.py`, `scripts/modular_family_probe.py`) | the parts the claims cite |
| D | git history (PR titles and dates), data manifests, the code that computes headline numbers (`inference.py`, the certificate) | spot checks |

## Method

1. **Timeline.** PR titles by date and each document's status lines, into phases: what question was
   asked, what changed it, which decision or result marks the turn.
2. **Claim ledger.** Every claim in `contribution.md`, the papers' abstracts and results, and the
   research notes: its evidence, registered or screen, `n`, status. The same figure is checked across
   every document that states it.
3. **Rule ledger.** Every rule in `experiment-design.md`, `information-flow.md` and the owner's
   conventions, against what later studies did (feasibility, the exclusion bound, 20 seeds a claim,
   one change between experiments, claims before data, epochs as the time axis).
4. **Decision ledger.** The state-formation workplan's D1-D67 and the earlier plans' decisions:
   superseded, contradicted, or left open; the 2026-10-21 reviews' items: closed or not.
5. **Spot checks.** Headline statistics recomputed with `inference.py` (Fisher, Wilson, Holm);
   a sample of manifests checked with `sha256sum -c`; paper figures against the docstrings.
6. **Assumptions.** The implicit premises the reasoning rests on, each against the evidence for it.
7. **The reviews re-assessed** (the owner's addition): `review-2026-10-15-contribution.md`,
   `review-2026-10-21-state-formation.md`, `reasoning-review-2026-10-21.md`,
   `consistency-review-2026-10-21.md` and `rng-audit.md`, each item read against the audit's findings:
   still right, overtaken, closed, or wrong.

## Deliverables

1. `review-2026-10-23-programme-evolution.md`: the programme's evolution, phase by phase.
2. `review-2026-10-23-programme-reasoning.md`: logical inconsistencies, gaps and bad assumptions.
3. `review-2026-10-23-programme-findings.md`: the audit findings, numbered, each with severity,
   evidence (file and line, or commit), and a recommended action.
4. `review-2026-10-23-reviews-reassessed.md`: the earlier reviews re-assessed (method step 7).

**Severity:** S1, a registered claim or a paper's statement is wrong or unsupported; S2, a rule broken
or a claim weaker than stated; S3, documents disagree or a statement is stale; S4, records and
hygiene.

## Decisions

**D1. Where the deliverables live.**
- (a) indrajala-research's top level, as `review-<date>-<topic>.md` (its README's convention).
  Pros: beside `contribution.md` and the earlier reviews of the papers; the research repo is where
  research docs moved (#952); private until publication. Cons: four files at the top level.
- (b) indrajala-ml `docs/audits/`. Pros: beside the two 2026-10-21 reviews. Cons: the research docs
  have just moved out of indrajala-ml; splits the research record again.
- (c) a shared doc or page. Pros: comments, easy to read. Cons: outside both repos' history and CI;
  not where the next session looks.
- **Settled: (a).** The plan stays here, with the other workplans.

**D2. Are the audited documents corrected in the audit's PR?**
- (a) Yes, as the 2026-10-21 reviews did. Pros: no stale statement survives. Cons: the audit stops
  being a snapshot; a reviewer can't tell the finding from the fix.
- (b) No: the audit records; a follow-up PR corrects S3 and S4 misstatements (wrong figures, stale
  paths); S1 and S2 go to the owner. Pros: reviewable; the owner decides what changes a claim.
  Cons: one more PR.
- **Settled: (b).**

**D3. Who reads.**
- (a) One reader, this session. Pros: one consistent ledger; no re-derived context. Cons: slower;
  the reader wrote much of what is audited.
- (b) Parallel subagents per tier. Pros: faster. Cons: each starts cold; cross-document
  inconsistencies, the audit's main target, fall between readers. Not asked for by the owner.
- **Settled: (a).** The limitation (the auditor is an author) is stated in the findings.

**D4. How deep the spot checks go.**
- (a) Recompute every statistic and checksum. Pros: complete. Cons: hours for little; most are
  computed by tested code.
- (b) The headline claims (`contribution.md`'s rows with a p-value or rate), and one manifest per
  phase. Pros: covers what a reader of the papers would rely on. Cons: an error in a minor figure can
  pass.
- **Settled: (b).**

**D5. The running probes (D63, D67).**
- (a) Leave out of the audit. Cons: the current state is the most relevant.
- (b) Audit their registrations, designs and the readings already made (D64's fresh-seed rate), not
  their partial data. Pros: the registered-probe rule holds; the design is auditable now.
- **Settled: (b).**

**D6. The earlier reviews' re-assessment: after or with the audit.**
- (a) After the four deliverables' first three are written (the owner's "once you are done with this
  one"). Pros: judged against settled findings. Cons: none worth stating.
- **Settled: (a).**

## Stages

0. This plan (one PR).
1. Read tiers A and B; the timeline and the ledgers in a scratch file.
2. Tier C and D spot checks.
3. Write deliverables 1-3; one PR in indrajala-research.
4. Re-assess the reviews (deliverable 4); one PR.
5. The follow-up corrections (D2 (b)); one PR per repo touched.
6. Retire this plan (a line in `next-steps.md` naming the four reviews).
