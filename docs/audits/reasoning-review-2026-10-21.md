# Review of the reasoning: gaps and inconsistencies (2026-10-21)

**Status: a dated note, written at the owner's request ("review our internal findings and papers,
and specifically our line of reasoning, in the light of what we now know ... are there obvious
things we have missed, or obvious inconsistencies in our reasoning"), with no further literature
review.** Read against [state-formation-workplan.md](../workplans/state-formation-workplan.md) (D13-D34),
[failure-kinds.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/failure-kinds.md), [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md),
[information-flow.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/information-flow.md), the status note
(`~/code/indrajala-research/research-notes/state-formation-status.tex`) and `contribution.md`, with the adaptive
cool's existence probe (D33's probe) and the handover probe's S3 arm (4a'') still running. Nothing
here is planned work: what it recommends goes to the panel's (4b) registration or to a decision.
Two misstatements it found are corrected in the same PR (item 6).

## Inconsistencies in the reasoning

1. **The target is in the training loop, against the discovery aim.** We say the certificate "is the
   final proof, never a guide during training" (the status note, the state-formation note), yet
   every run stops at its first certified step (`stop at certified`, the probe script; the register's
   row 13), and a population's selector is the certificate (D27). Every success rate we hold, P1's
   and P2's included, is "certified at some step, the target checked every step". In discovery there
   is no target, and training past a solution destroys it (P1, 3b), so an answers-only stop could
   lose most of these solutions. Right for the papers' framing (the target known); for discovery the
   largest unmeasured gap: no study measures the rate under an answers-only stop (the machine read
   off consistent with the answers, minimal, stable over steps). The saved step networks allow it
   offline.

2. **The mechanism given for "groups need a seed" explains conversion, but the failure was
   formation.** No synchronising word, so a wrong snap persists: that explains S3's curriculum state
   destroyed by an abrupt k-means snap (conversion). Without a curriculum S3 failed earlier: its
   continuous state never formed (chance under `state-separation --live`), with no snap involved.
   Formation needs its own account; a candidate: an S3 answer depends on every token of the word, a
   move-to-front answer mostly on the last few, so long S3 words carry little usable gradient until
   short ones are solved. The two accounts predict different remedies; the crucible addresses both,
   so its success doesn't say which reasoning held.

3. **The coupling window rests on a curve now known to mispredict, yet it works.** The gradual
   handover (D25) ties the coupling to the measured melting curve, and the 3-octave window (D30)
   extends it; but training's splits come 2-6 octaves hotter than that curve predicts (D32's
   understanding step, "the splits come early"). So the handover doesn't track the material's
   transitions, and the working ingredient is likely a slow, gradual coupling. The bounded handover
   (D34) points the same way: what broke was the coupling's rate, and the fix bounds it directly.
   The untested control: the coupling on its own clock (D34's option (c), set aside for the owner's
   link to temperature).

4. **Two methods, not one general architecture.** The crucible made S3 work; move-to-front-4 seed 3
   hasn't certified in the crucible under either cool, only with no curriculum (`--whole`, 4a''). Both
   panel arms are crucible arms, so if the crucible hurts reset machines the panel shows it as harm
   nobody registered. D33's probe's move-to-front-4 crucible runs are the early reading.

5. **The bounded handover (D34) erodes the panel's two-arm rationale (D32).** The arms were to let "a
   cooling failure in one arm and not the other name itself". Bounded to the quasi-static cool's rate
   inside the window, where the failures came, the adaptive arm now differs mainly outside it. Its
   other gain is speed, and time isn't the constraint (the owner). Its remaining advantage is
   generality (no shape, octave or cool-length constants); if that is the reason to keep it, it
   should be the stated reason, or the second arm doubles the panel for little diagnosis.

6. **Two misstatements, corrected here.** `failure-kinds.md`'s 2026-10-21 log line had the
   quasi-static crucible curing move-to-front-4; that certificate came from the handover probe (4a''),
   with no crucible, before the quasi-static cool existed. `machine-discovery.md`'s "breadth" row
   listed random machines without their limit (certified to about 12 states, 0 of 6 at 16, set aside
   since 2026-10-18).

## Gaps: what the aim needs that is missing

7. **No random machine in the panel (4b).** Discovery's targets are unknown and likely unstructured;
   random machines are where extra codes and lost near solutions live (16 states and up), the failure
   kinds option 1 hasn't touched, and every option-1 setting was tuned on S3 and move-to-front. The
   random-machine conjecture (more structure, a quicker cool) has a ready test, the melting curve
   (D21) on a random warm-up, not yet run. random-q16-0 (and a 12-state control that certified
   before) in the panel cost little.

8. **The feasibility register predates option 1.** `information-flow.md` has no rows for the
   crucible or the coupling, and its rows 8 (the commitment) and 13 (stop at certified) predate both.
   The crucible's cap of 8 (D28) was chosen on S3 (6 states, reached within about 3 tokens); on S4,
   S5, move-to-front-5 or division at 169 states its lengths may not reach every state, or tell every
   pair apart, so the seed would be a partial machine. A row is needed (the crucible's lengths reach
   every state and distinguish every pair), checked per backdrop before the panel.

9. **The two strands aren't joined.** P2: an approximation's outputs beside the target's make
   product machines learnable, and one suffices. The state-formation work trains on answers alone and
   then finds its factors are approximations anyway. Missing: S3 with an approximation's output beside
   it (the sign, or an item's place). Does it form without a crucible, and does it steer which factor
   holds what? That joins P2, the discovery premise (known approximations supplied) and the current
   bottleneck.

10. **The harm question has no stated mechanism.** Option 1 runs with the commitment never on;
    before, never-on harmed S4 and division (D11's harm cells), which certified only with it.
    Nothing in the reasoning says why crystallisation would cure that. The certified networks' thin
    margins (near ties) are the plausible link (the commitment widened them). An expectation for the
    harm cells belongs in the panel's registration, not in its reading.

## What holds up

The formation/conversion split; the population view (a per-run rate `p`, the certificate free of
false positives); fixing only failures visible early in a run's own logs; reading the factors as
approximations and covers; the status note's own caveats (8 of 8 measured on the seeds the fixes
came from).

## Recommended before the panel

- Item 1, offline: for every certified run, would an answers-only stop have kept its machine?
- At the panel's registration: the crucible's feasibility row (8), a random-machine cell (7), the
  harm cells' expectation (10), the second arm's reason (5).
- Items 2, 3 and 9 are design questions for after the panel.
