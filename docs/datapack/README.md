# Verifying paper 1

This is the datapack for *Learning Finite-State Transducers Through Their Approximations, with Proof*
([PAPER LINK]), archived at [DATAPACK DOI]. It holds only what the paper cites. Everything here can
be checked without trusting its authors: the files against their hashes, every number in the paper
against the files, every learned machine against its target, and the training itself against a
rerun.

## What is in the record

| file | what it holds |
| --- | --- |
| `verify.tar.gz` (about 0.4 GB) | what `paper_tables.py check` reads to regenerate every number and table in the paper, plus every certified network (2,762 files) |
| `verify.sha256` | the verify pack's manifest: each file's SHA-256, the same hashes the paper's manifests cite |
| `full-<part>.tar.gz` (about 20 GB in all) | every file the paper's manifests list, one archive per test (`t1`, `t2-t5`, ..., `t12-runs`), plus `approximation-claim` and `addition`; each archive carries its manifests |
| `MANIFEST.sha256` | the SHA-256 of every archive and of every manifest |

The verify pack is enough to check every claim; the full pack is the archive of record (every
training log, every step's network, every reading).

## Setup

The code is [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml) at [RELEASE TAG], with its
native crate [indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust) as a
submodule; the manuscript's source is in
[indrajala-research](https://github.com/davidbarkhuizen/indrajala-research) at [RELEASE TAG].

```bash
git clone --recurse-submodules https://github.com/davidbarkhuizen/indrajala-ml
cd indrajala-ml && git checkout [RELEASE TAG]
./cli setup   # .venv, Python dependencies, a release build of the crate (needs a Rust toolchain)
git clone https://github.com/davidbarkhuizen/indrajala-research ../indrajala-research
```

## 1. The files are what the paper cites

Check the archives you downloaded against `MANIFEST.sha256`, then the verify pack's files against
`verify.sha256`:

```bash
sha256sum -c --ignore-missing MANIFEST.sha256
mkdir verify && tar -xzf verify.tar.gz -C verify
(cd verify && sha256sum -c --quiet ../verify.sha256) && echo "every file verifies"
```

Each hash in `verify.sha256` is one of the paper's manifests' own: the verify pack is a subset of the
full pack, not a new set of results.

## 2. Every number and table in the paper comes from these files

`paper_tables.py check` recomputes every table the manuscript holds, and every measured number in its
prose (each marked in the source with its key), from the result files, and compares them with the
text. It prints a line per table and fails on any difference:

```bash
.venv/bin/python scripts/paper_tables.py --data ../verify \
  check ../indrajala-research/papers/learning-transducers-with-proof.tex
```

The numbers left unmarked in the paper are settings (lengths, bounds, seeds), not measurements.

## 3. Every learned machine is its target

A snapped network is a Mealy machine whose states are its reachable joint codes; the certificate
enumerates that machine from the saved network, minimises it and compares it with the target, every
output and every next state. A pass is a proof at every length, not a sample. For example, T1's
`d5-sum-319` (modular division at p = 5, trained on the sum):

```bash
.venv/bin/python scripts/modular_family_probe.py certify \
  ../verify/paper/t1/runs/sum/seed319/certified-step-9.network.json \
  --task modular-division-5 --arm sum --vocabulary sum suffix token --out cert.json
# certified: 105 codes reached, minimised to 25, 0 wrong outputs, 0 wrong states ...
```

Each test's `certify/` directory holds the certificate the paper counts for every network, in both
verdicts (with the network's own snaps, and robustly: every near tie branched both ways). The
network's SHA-256 is in its long test's record (`*.long-test.json`, key `sha256`).

## 4. Training reproduces bit for bit

Training is deterministic given the seed, the code and the crate. `batch.py command` prints the exact
command that ran an attempt, from its batch's ledger:

```bash
.venv/bin/python scripts/batch.py command ../full/paper/t1 d5-sum-319 --out rerun-d5-sum-319
```

(`../full` is the full pack extracted; step 5.) Run it (T1's `d5-sum-319` takes about 5 minutes on
one core), then compare its log with the
original's, every logged row but the wall-clock seconds, and its certified network's SHA-256 with
the original's:

```bash
.venv/bin/python scripts/modular_family_probe.py same-prefix \
  rerun-d5-sum-319/log.jsonl ../full/paper/t1/attempts/d5-sum-319/log.jsonl
sha256sum rerun-d5-sum-319/certified-step-9.network.json   # 45c72add3cc9d641...
```

The paper's T7 reran the shortest certified run of each of T1, T3, T2 and T5 this way, on aarch64 (a
Raspberry Pi 5) and on x86-64, from the code at release: all four matched bit for bit
(`full-t7-reruns.tar.gz`, `full-t7-reruns-jebel.tar.gz`).

## 5. The full pack

Extract every `full-*.tar.gz` into one directory. Each manifest verifies with `sha256sum -c` from its
own directory or its stem's (`paper/t2-t5.sha256` from `paper/t2-t5/`), with one exception,
`paper/t4-fresh.sha256`, from `paper/t2-t5/approximations/`. `datapack.py check` does all of it,
re-hashing every archive and every file inside against the manifests each archive carries:

```bash
.venv/bin/python scripts/datapack.py check <the directory holding the downloaded files>
```

## What to know

- **Manifests added after the readings.** Four manifests were written on 2026-10-28, after the tests
  they cover were read, because the originals had left those files out: `paper/t11-runs.sha256` and
  `paper/t12-runs.sha256` (T11's and T12's training logs and certified networks),
  `control/models.sha256` and `e3-e4/long-tests.sha256` (two of addition's studies). They attest the
  files as they are now, not at the time the tests ran.
- **Re-read reports.** Where a later re-read (the strict reader, T6) rewrote a test's proof reports,
  the test's manifest names the originals, kept beside them in `proof-loose/`; `t6-proof.sha256`
  covers the rewritten ones.
- **What is not here.** Earlier exploratory studies the paper does not cite (the state-formation
  work among them) are left for the papers that use them.
