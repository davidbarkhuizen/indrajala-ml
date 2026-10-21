# Note: state formation and crystallisation, where stage A stands (2026-10-20)

**Status: a note, 2026-10-20 (about 18:30), written at the owner's request for a standalone summary of
the day's findings against the project's aim. Nothing here is planned work; the plan is
[state-formation-workplan.md](state-formation-workplan.md), whose findings and decisions (D13-D27)
hold the details and the data's paths. Two registered probes (4a'', 4a''') were still running when
it was written, so no verdict is applied here.**

## The aim, and where stage A stood

The project's aim is machine discovery ([machine-discovery.md](machine-discovery.md)): unknown
finite-state machines found from behaviour and known structure, and proved. Its stage A is one
general architecture and training method that forms a state, keeps it and certifies on every
backdrop ([failure-kinds.md](failure-kinds.md)). The network is the window-recurrent transformer: each
window's tokens and a snapped state (one code a factor) give the next state and the answers, so a
trained network is a finite-state transducer whose certificate (every window from every reachable
state, against the target machine) is a proof.

Before 2026-10-20 training certified some backdrops, but its failures resisted every remedy:
**failure A**, a state that never forms (word-problem-s3 seeds 1 and 4, move-to-front-5 seeds 0 and
1, others), and **harm**, the commitment term (which pulls the cells' outputs to their codes) hurting
formation early while some backdrops needed it late. The literature review
([window-recurrent-literature.md](window-recurrent-literature.md)) proposed four options; option 1,
annealing the snap, was being retried as a crystallisation (heat, a melt, a slow shaped cool).

## The findings

1. **Two problems, not one: formation and conversion.** *Formation*: the continuous network's cell
   outputs come to carry the target's state. *Conversion*: discrete codes take that state over
   without losing it. `state-separation` (nearest-neighbour and centroid accuracy at naming the target's
   state from the cell outputs, the target read only to measure) tells them apart, and earlier remedies
   aimed at one were judged on runs failing at the other.

2. **Conversion: crystallisation with a gradual handover works.** Each factor's next state is
   `(1 - l) h + l q` (the cell's output `h`, its code `q`), `l` rising from 0 to 1 with the falling
   temperature between the factor's measured melting point and its last transition (D25); the codes
   anneal on the continuous state while the forward path keeps its information. Two certificates, each
   of exactly the target's states, with the commitment never on:
   - move-to-front-4 seed 3 (convex cool, step 104): minimised to the target's 24 states;
   - word-problem-s3 seed 0 (in a crucible, below; step 85, 20 minutes): minimised to 6, certified at
     the default snap ratio (1.05, near ties branched) and strict (1.0).
   An abrupt snap had destroyed the same states (S4: joint separation 0.98 to 0.25 at length 16). The
   melt costs nothing lasting: it melts the codes only, while `h` keeps the state (move-to-front-4:
   1.00 at every length through the melt; the cool re-formed and bettered the warm-up's snap).

3. **Formation depends on the machine's kind: resets against groups.**
   - *Aperiodic machines* (move-to-front: every token a reset of part of the order) form their state
     from the whole length distribution at once, no curriculum: separation 1.00 at lengths 4-128 at the
     warm-up's end. A snapping error is forgotten within a few tokens.
   - *Groups* (S3's word problem: every token a permutation) don't: from the whole distribution the
     state never formed (centroid accuracy at chance from length 16). With a curriculum it formed (1.00
     to length 32), but **snapped onto its own k-means codes it fell to chance** (0.41 at 16, 0.20 at 64).
     A permutation machine has no synchronising word, so one wrong snap stays wrong to the word's end.
   - So a group's state must be **seeded**: crystallised, with the handover, on the short-word material
     where it formed, before the long words arrive. The crucible (D26: the curriculum's warm-up, then
     the anneal on its last step's lengths held fixed, then growth on the whole distribution) gave S3's
     certificate. In Krohn-Rhodes terms: the network learns the aperiodic (reset) components readily;
     group components need a seed.

4. **The learned machines carry real structure.** `approximations` (built 2026-10-20) reads what each
   factor's code determines of the target exactly: the finest approximation (a congruence's quotient, a
   homomorphic image in Egri-Nagy and Nehaniv's terms) it fixes, and whether the factors together cover
   the target.
   - move-to-front-4: each factor holds pairwise recency orders exactly, each a flip-flop (the bits of
     an LRU's age matrix; five of six held exactly, the sixth jointly); 13 earlier certified networks
     hold the same kind (relative orders of two or three items), one of them an exact cover (the target a
     subdirect product of its two factors' quotients).
   - word-problem-s3 seed 0: one factor fixes a 3-block approximation, exactly the place of item 2 (the
     group acting on three points; `places {2}`, named since), the rest held jointly.
   This is discovery's decomposition side, read from the network rather than given to it.

5. **A statistical, parallel method suits the problem.** Each run is an independent draw and the
   certificate is a verifier without false positives, so a population of independent runs turns a
   per-run failure (a seed freezing wrong) into a probability. The first population (D27: S3 seeds 0-7,
   independent crucibles) certified within 20 minutes. Population annealing (replicas resampled during
   the cool) is deferred until this baseline is read (the owner).

## What it means for the paper

The publication's frame (2026-10-17: learning finite-state transducers with proof) gains a training
method that converts a learned continuous state into a provably correct machine, trained on inputs
and answers only, with no commitment, no traces and no given decomposition; an account of when the
state forms that turns on the machine's algebra (resets against groups, by synchronising words); and
learned machines whose factors read out as the target's approximations.

## What is not established

- **Evidence:** one certified seed on each of two tasks; the registered decision rules of 4a'' and
  4a''' are applied only when they end.
- **Generality and harm:** the panel (S4, move-to-front-5, division, S5) hasn't run with the crucible
  and the handover; the harm runs certified before only with the commitment on.
- **Failures in the population, and what they mean:** path dependence is expected (the owner: a
  generalising method shouldn't work from every start; random machines will need the phase space sampled
  over many seeds). With a verifier free of false positives the measure is the per-run success rate `p`
  and the cost of a proved machine (about `1 / p` runs), and the question for random machines is how `p`
  scales. Still, a failure that correlates with something observable early is a design flaw lowering `p`,
  to fix; one that correlates with nothing is the phase space, to sample. Tonight's candidates for the
  first kind: the crucibles whose warm-up ended at step 3 (lengths 1-3, stalled at shares 0.34-0.38) and
  S3 seed 2's collapse late in the cool at full coupling (as 4a's late-cool losses); the no-curriculum S3
  runs are expected to fail.
- **Bottleneck 2** (extra codes wrong within a few tokens, the random machines' failure) is untouched.
- **The other options** (noise in the cell, the rotation trick, FSQ factors) are next once option 1 is
  decided.
- **Discovery proper** (stages B-F) still needs a state bound, and the certificate needs the target: here
  it is the final proof, never a guide during training, which keeps the method open to discovery.

## Next

Read 4a''' and 4a'' by their registered rules; if they hold, the panel (4b) with the crucible and
the handover as option 1's candidate, its first questions the harm runs and the crucible's length;
D26 (b), reheating a seed just below its melting point on the whole distribution, is the variant in
reserve.
