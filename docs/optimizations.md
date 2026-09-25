# Optimizations

Performance work on the Rust backend (`rust/`, the `indrajala-math-rust` crate), measured against
numpy. Every optimization is measured before and after and must keep every parity test passing;
most are bit-identical.

**Where it stands:** Rust is 0.11-0.65x numpy's wall-clock time end to end on the conv demo, and
about 0.21x (single-example) and 0.54x (mini-batch 32) on a dense MNIST epoch. The gaps left are
in batch ops.

## Documents

- [Current baseline](optimizations/current-baseline.md): Rust against numpy now, end to end and
  per op, and where a Rust epoch spends its time.
- [Implemented](optimizations/implemented.md): the optimizations in the code and why each works,
  starting with the fixed-summation-order constraint every kernel keeps.
- [Rejected](optimizations/rejected.md): ideas investigated and discarded, with the reason, so
  they aren't proposed again.
- [Candidates](optimizations/candidates.md): future optimizations, ranked by stake, with plans;
  deferred work, unmeasured leads and open questions.
- [Measurement](optimizations/measurement.md): the machine, the benchmark tools, protocols,
  gotchas, and the rules for an optimization PR.

These describe the current state, not its history: a change updates them in place (a number is
replaced, an item moves between Candidates, Implemented and Rejected), and the measurements
behind it live in its PR.
