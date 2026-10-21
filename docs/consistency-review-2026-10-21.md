# Review of the programme's internal consistency (2026-10-21, afternoon)

**Status: a dated note, written at the owner's request ("do a fresh review of our research
programme for internal consistency"), after the program's restructuring (D35-D39) and before the
panel's (4b) registration.** Read against [state-formation-workplan.md](state-formation-workplan.md)
(D1-D39), [reasoning-review-2026-10-21.md](reasoning-review-2026-10-21.md) (the morning's review),
[failure-kinds.md](failure-kinds.md), [machine-discovery.md](machine-discovery.md),
[information-flow.md](information-flow.md), [experiment-design.md](experiment-design.md), the
probes' headers (`~/code/probe-d33.sh`, `probe-d34.sh`, `probe-d38.sh`), the harness
(`indrajala_ml/studies/modular_family.py`) and the status note
(`~/code/indrajala-papers/state-formation-status.tex`). The adaptive cool's existence probe (D33's
probe) and the handover probe's S3 arm (4a'') were still running, and the coupling probe (D38)
waiting for 19:00. Nothing here is planned work: items 1-6 go to the panel's registration or to a
decision; the stale statements (item 9) are corrected in the same PRs.

## Rules the programme states, against what it does

1. **The accept rule (D12) and the panel's registered harm expectation (D37) contradict each
   other.** D12 still stands (the rebuild's preamble: "D10-D12 stand"): an option is rejected when
   it loses a harm certificate. D37 registers the expectation that at least one harm cell loses its
   certificate in each arm. If D37's expectation holds, D12 rejects option 1 on the panel it was
   rebuilt for, though D31 (diagnose before a remedy) already treats the panel as a diagnosis, not a
   verdict. D12 also compares failure-A runs "against their controls", and D37's new cells (S3
   seeds 8-15, the random machines) have none. **At registration:** say what decides option 1 (D12
   as written, or D12 replaced: the harm losses read for their mechanism and the decision taken
   after the diagnosis), and name each new cell's control or say it has none.

2. **The exclusion bound is met by the configs, not by the certified runs.** The register's row 2
   and experiment-design.md require training to `m + q - 1` so a wrong machine can't fit; the
   crucible (D26) claims it through growth ("growth trains to the cap, so the run's training meets
   the exclusion bound"). But every certificate of the quasi-static crucible came inside the
   crucible (4a'''': "none reached growth"), trained to length 8 against S3's bound of 69; seed 2
   certified in the heat. Nothing is wrong (a certificate is a proof), but the stated reason the
   cells are sound is not what happened, and under the answers-only stop (D35), with no certificate
   in the loop, the training lengths are what stands between a run and a memorised machine. D35's
   bound on the machine read off does change the arithmetic: a machine of at most `N` states that
   agrees with a `q`-state target on every word to `N + q - 1` is the target (S3 at `N = q`: 11),
   but D35 reads sampled answers (lengths 8-48), not every word. **At registration:** row 2 restated
   for the crucible (what the stop's lengths exclude, with D35's bound), not left to growth.

3. **The crucible's cap is a fixed constant whose sufficiency is read from the target.** Row 15
   checks the cap (8) against each backdrop's deepest state plus Chow's `k`, both the target's. For a
   feasibility check that is right; for the general method (one architecture, tasks as backdrops)
   and for discovery, the cap must come from what is known, the state bound, whose worst case (depth
   and `k` each up to `N - 1`) is far above 8 at division's 169. The owner's reason for dropping the
   curriculum (D22: "i don't think we could build an ideal curriculum without already knowing too much about
   the target machine") applies to the crucible's cap the same way. The panel's crucible-against-whole
   arms (D36) measure the crucible's harm, not this. **For after the panel:** the cap set from the
   state bound, or the cap's dependence stated as a known-structure assumption.

4. **The answers-only stop (D35) moves the target's size into the loop, not the target out of it.**
   The morning's item 1 (the target in the training loop) is answered for the certificate; the stop
   now reads `N` set to the target's size, and the re-reading showed the slack is what costs (1.1
   times the size: 25 false stops, 1 false discovery). machine-discovery.md has a bound `m` known,
   not tight. So the panel measures discovery with an exact state bound, the best case. Under a
   slack bound a population (D27) also needs a rule for choosing among runs that stop on different
   machines; the certificate is the only selector the programme has. **At registration:** the claim
   worded as "under an exact state bound"; the slack-bound study and the population's selector noted
   for stage C.

5. **What counts as a success under D35 isn't settled.** D35 records two verdicts (the default
   snap ratio, 1.05, and 1.0) and accepts a network exact only at 1.0 as a stop; D12 counts an
   option's certificates only when they "pass both verdicts"; `population` counts a run by its saved
   network's certificate. Of the re-reading's 8 earlier stops, the four read were exact at 1.0 and
   failed at 1.05 on near ties. **At registration:** one definition of a certified run for the
   panel's counts, the other verdict reported beside it.

## The coupling probe (D38)

6. **At coupling cap 0 the quasi-static gate guards the continuous network, so D38 changes two
   things, not one.** The harness reads D29's gate from the epoch's screen with the couplings set
   (`gate_screen = score`, after `net.set_couplings`), as the re-reading found for the logged
   answers. At cap 0 every phase's forward path is the cells' outputs, so the gate reads the liquid
   and never the codes; the snapped path gets no gradient (`l dq` with `l` 0). The header says "one
   change here". A failure then can't be laid on the coupling alone: the gate that fixed 4a''''s
   losses no longer sees the codes. Its registered refutation reading (where the snapped and coupled
   screens part) covers part of it; **at reading:** note whether any run's gate held, and read the
   snapped answers at each step's network (`answers-stop --snapped`) against the logged screen.

7. **D38's registered expectation omits the evidence against it.** The probe expects at least 6 of
   8 from codes annealed on the liquid and never coupled. The workplan's separation findings say an
   untrained snap of a formed S3 state fell to chance (k-means codes, joint 0.41 at length 16) and
   put the loss "at the snap itself" (D25's reason: the coupling lets the network shape its states
   into clusters the codes can hold). For it: S3 seed 2 certified at coupling 0 in the heat, and the
   re-reading's certificates at couplings from 0.06. Either outcome rewrites something: if D38 holds,
   D25's stated mechanism and the status note's "conversion succeeds as a crystallisation with a
   gradual handover" are wrong as written; if it fails, the re-reading's inference (item 3 of the
   morning's review) was. **At reading:** both written up, whichever holds.

8. **The bounded handover's probe (D34's probe) tests a refinement of a part D38 may shelve.** D36
   took the adaptive cool out of the panel; D38, if it holds, shelves the handover (D25, D30), and
   with it the coupling's rate that D34 bounds. The handover runs D34's probe on `pyramidon` before
   D38 is read. D38's S3 runs take 4-28 minutes, so its S3 verdict is in by about 20:00. **Order:**
   D34's probe after D38's S3 reading, and only if D38 is refuted (then the adaptive cool's
   refinement is still a live question); if D38 holds, it is set aside unrun, recorded.

## Statements the documents now contradict

9. **Stale against later decisions (corrected in the same PRs):**
   - The status note's "agreed direction" has the panel's two arms as the quasi-static crucible and
     adaptive cooling; D36 replaced the second with `--whole`. Its last open item keeps "the
     certificate ... the final proof, never a guide during training", the misstatement the morning's
     item 1 named; corrected there to D35's stop. (The state-formation note of 2026-10-20 says the
     same; it is a dated note, so it is left as written, with this review naming it.)
   - The workplan's stages 5-7 list options 2-4 in turn; D31 shelved them. Its "Machines" has
     `jebel` down until 2026-10-22; it is up (06:00-18:00).
   - failure-kinds.md's 2026-10-19 table (its "plan" and "next" rows) points to options 2 and 3 and
     to the plan's stage 2; a log line now records that D31 replaced that route.
   - The register's row 8 (the commitment: "required, open") predates option 1, which runs with the
     commitment never on; it now says so. There is still no row for the coupling (the morning's item
     8); one is written once D38 is read, since D38 decides whether there is a coupling.

10. **A claim broader than its evidence (not corrected; for the status note's next revision).** The
    status note's abstract and its Krohn-Rhodes paragraph draw the class line ("reset-like machines
    form their state from the whole distribution; groups need it seeded") from one machine of each
    kind, one seed (move-to-front-4 seed 3) on the reset side. Move-to-front-5, reset-like, failed
    by failure A (formation or conversion, not then told apart) in 4 of 6 runs in round 2, with the
    curriculum. The morning's item 2 also found the
    synchronising-word mechanism explains conversion, not formation. D36's panel is the test; until
    it reads, "on the two machines tried".

## What holds up

The registration discipline (every probe since 4a with its expectations in its header before its
data, and the flaw in 4a'' fixed by resumption, not by a changed setting); the feasibility checks
written beside every cell; the failure kinds and their measurements; the morning's corrections
applied; D35-D39 each recorded with its options' pros and cons; the population view and its caveat
(8 of 8 measured on the seeds the fixes came from).

## Recommended at the panel's registration

- The decision rule (1) and each cell's control; one success definition (5).
- Row 2 restated for the crucible under D35 (2); the claim under an exact state bound (4).
- After D38's reading: the coupling's row, and D25's or the re-reading's account revised (7);
  D34's probe run or set aside (8).
- Items 3 and 10 are for after the panel.
