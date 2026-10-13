"""
The window-recurrent adder (docs/window-recurrent-adder-workplan.md): whether a network that reads a
window of k columns and hands a snapped state from window to window learns, through a curriculum of
window counts, to add base-3 numbers of any length, under indrajala_ml/studies/window_recurrent_study.py's
harness and indrajala_ml/data/addition_data.py's catalogue.

    python scripts/window_recurrent_study.py run CONFIG.json
    python scripts/window_recurrent_study.py status CONFIG.json
    python scripts/window_recurrent_study.py report CONFIG.json [--md FILE]
    python scripts/window_recurrent_study.py check NETWORK.json --n N [N ...] [--properties ID ...]
        [--sampled S] [--per-stratum P] [--doubling] [--json FILE]
    python scripts/window_recurrent_study.py certify NETWORK.json [...] [--out DIR] [--workers W] [--snap-ratio R]

A config names the study's directory (results/, one JSON file per run, and models/, each run's steps
and their networks in format 2), its workers, its settings and its matrix (load_config's
docstring). `run` resumes from what is on disk, a run from its last saved step, and prints a line
per finished run, then the report; it holds the machine lock, as every study. `check` loads a saved
step's network and runs the catalogue (without B3) at each length: its properties, B1-B3's,
A3's, M1's and M5's samples, M3's and M4's cases per stratum and M3 at 1, 2, 4, ..., n only (the
long tests of docs/window-recurrent-claims-workplan.md, D9) as given. `certify` writes each saved
network's certificate (indrajala_ml/studies/window_recurrent_certificate.py) as
DIR/<run>-<step>.json and prints a line for each, over W processes; --snap-ratio 1 certifies the
network's own snaps only, branching no near tie (strict, D3 as first written).

OPENBLAS_NUM_THREADS=1, as every study.

Findings (stage 9), on jebel (i7-9700K), indrajala-ml 6b94933, crate 8c20e510c9a9,
`scripts/window_recurrent_matrix.json`: the curriculum arm, `k` 1, 2, 4 x `s` 4, 16 x `m` 4, 8,
16, 5 seeds, 3 steps (1 window, then 1-2, then 1-3), at most 200 epochs a step, patience 30, the
catalogue's default sizes, each step's test pass up to its first failing length (W12), 6 workers.
90 runs, none raised. Wall time 2026-10-12 21:43 to 2026-10-13 05:14 (7.5 h; 43.4 h of runs, 24.6
of them training and 18.8 testing); a run took 2-67 minutes (median 27). The results are on jebel
and pyramidon in `data/window-recurrent/matrix/results/` (90 files, listed with their sha256 in
`data/window-recurrent/matrix/results.sha256`, itself sha256 407d6ec3...7941cab9), every step's
network beside them in `models/` (156 MB), and the log in `data/window-recurrent-matrix.log`
(sha256 b47626f0...68a7dd59).

- Half the runs work (W10: a break of 16 columns or more): 45 of 90. 43 of them pass every
  property at every test length up to 64 columns, after training on at most 3 windows: 3, 6 and 12
  columns at `k` 1, 2 and 4, window ratios of 21.3, 10.7 and 5.3. The other two break at 24 and 32
  (`k` 4, one B1 case in 100,000 wrong at the next length). In every working run the carry reaches the
  whole length (`L*` = n at every test length).
- It takes the recurrence: no run works after step 1, which trains one window and so never hands
  a state on; 9 work from step 2 (2 windows; at `k` 1, two columns of training add 64), 36 more
  from step 3, and none that worked at step 2 stopped working at step 3. A working run's step 3
  passed its check in a median of 2 epochs.
- Failure is mostly all or nothing: 43 of the 45 failing runs fail at 8 columns already, most
  across B1, A2, M2-M5 (their step 3 best screens: median 0.924). A few are near misses: two `k` 4
  runs break at 12 (their trained length) with 1-2 cases in 100,000 wrong at 16, and one `k` 2 run
  misses 4 of B1's 100,000 at 8 (and a few of A1's, M3's and M5's).
- Passing the check isn't working at `k` 1: 5 runs passed their step 3 check (every triple at 1-3
  columns) and still got 0.2-8.3% of B1 wrong at 8.
- By axis: `k` 1, 2, 4 work in 16, 11 and 18 of 30 runs; `s` 4 and 16 in 24 and 21 of 45 (the
  state's width barely matters); `m` 4, 8 and 16 in 10, 16 and 19 of 30 (more codes, more working
  runs). No cell works on every seed: the best, 4 of 5, are `k` 1 `s` 16 `m` 8, `k` 4 `s` 4 `m` 16
  and `k` 4 `s` 16 `m` 16; the worst, `k` 2 `s` 16 `m` 4, on none. Within a cell the seed decides,
  so what fails is the optimisation, not the capacity: every cell has a state that can hold a
  carry (3 values), and 6 working runs use exactly 3 codes.
- The codes used by working runs: 3-11 (median 5) over the lengths tested, so a working network
  uses more codes than the carry's 3 values; the probes (stage 10) say what the others hold.
- Cost: a step 3 epoch trains in a median 6.8 s; a working run's step 3 test pass (every length)
  took a median 17.6 minutes, a failing one's (length 8 alone) about a minute.

Per cell, its seeds working and each seed's break (in columns, seeds 0-4) after steps 2 and 3, and
the codes the working seeds used:

| k | s | m | parameters | seeds working | breaks after step 2 | breaks after step 3 | codes used (working) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 7,683 | 2/5 | 0/64/64/0/0 | 0/64/64/0/0 | 4,4 |
| 1 | 4 | 8 | 7,699 | 3/5 | 0/0/0/0/0 | 0/64/64/0/64 | 8,6,7 |
| 1 | 4 | 16 | 7,731 | 2/5 | 0/0/0/0/0 | 0/64/64/0/0 | 11,11 |
| 1 | 16 | 4 | 9,651 | 2/5 | 64/0/0/64/0 | 64/0/0/64/0 | 4,4 |
| 1 | 16 | 8 | 9,715 | 4/5 | 0/0/64/0/0 | 64/64/64/0/64 | 6,4,4,7 |
| 1 | 16 | 16 | 9,843 | 3/5 | 0/0/0/0/0 | 64/64/64/0/0 | 7,7,6 |
| 2 | 4 | 4 | 7,699 | 1/5 | 0/0/0/0/0 | 0/0/64/0/0 | 3 |
| 2 | 4 | 8 | 7,715 | 3/5 | 0/64/0/0/0 | 0/64/64/0/64 | 4,6,4 |
| 2 | 4 | 16 | 7,747 | 3/5 | 0/0/0/0/0 | 0/64/0/64/64 | 8,7,5 |
| 2 | 16 | 4 | 9,667 | 0/5 | 0/0/0/0/0 | 0/0/0/0/0 | - |
| 2 | 16 | 8 | 9,731 | 1/5 | 0/0/0/0/0 | 64/0/0/0/0 | 6 |
| 2 | 16 | 16 | 9,859 | 3/5 | 0/0/0/0/0 | 0/0/64/64/64 | 9,4,8 |
| 4 | 4 | 4 | 7,731 | 3/5 | 8/0/64/0/64 | 64/0/64/0/64 | 4,3,3 |
| 4 | 4 | 8 | 7,747 | 3/5 | 0/0/12/0/0 | 0/64/64/12/64 | 4,5,5 |
| 4 | 4 | 16 | 7,779 | 4/5 | 0/0/0/0/8 | 64/64/12/32/64 | 7,5,7,7 |
| 4 | 16 | 4 | 9,699 | 2/5 | 0/0/0/0/8 | 0/64/0/0/64 | 3,3 |
| 4 | 16 | 8 | 9,763 | 2/5 | 64/0/0/0/0 | 64/0/0/64/0 | 4,3 |
| 4 | 16 | 16 | 9,891 | 4/5 | 8/0/0/0/0 | 64/24/64/64/0 | 5,6,4,6 |

Findings (the control, W16 and W30), on jebel, indrajala-ml 1dd22f9, crate 8c20e510c9a9,
`scripts/window_recurrent_control.json`: the all-at-once arm (one step on window counts 1-3
together, at most 200 epochs) at the matrix's three best cells, seeds 0-4, the matrix's settings
otherwise. 15 runs, none raised. Wall time 2026-10-13 05:28 to about 06:29 (5.3 h of runs; a run
took 1-33 minutes, median 23). The results are on jebel and pyramidon in
`data/window-recurrent/control/results/` (15 files, listed with their sha256 in
`data/window-recurrent/control/results.sha256`, itself sha256 6eed795f...a6facf9b), the networks
beside them in `models/` (9.2 MB), and the log in `data/window-recurrent-control.log` (sha256
913e996f...c52662dd).

- The curriculum isn't needed here: the control works as often, 12 of 15 runs against the
  curriculum's 12 of 15 at the same cells and seeds. At `k` 4 it works on all 10 (curriculum 8 of
  10; Fisher's exact test p 0.47); at `k` 1 `s` 16 `m` 8 on 2 of 5 (curriculum 4 of 5; p 0.52).
  At 5 seeds neither difference is decided; the claims workplan's E4 measures it at 20.
- It needs far fewer examples where it works: at `k` 4 it passes its check within 4-16 epochs,
  0.26-1.05 M examples (median 0.39 M), against the curriculum's 4.8-12.3 M over its three steps
  for the runs that work (median 6.4 M), about 16 times fewer. The curriculum spends 78-99% of
  its examples on its first two steps (one window, then one or two; 61 of the matrix's 90 runs
  stop step 1 on the plateau), and once it trains on 1-3 windows it passes in a few epochs, as the
  control does from the start.
- Which seeds work isn't shared between the arms: at `k` 4 the curriculum's two failing seeds
  (seed 2 of `s` 4, seed 4 of `s` 16) work all at once, and at `k` 1 the arms fail on different
  seeds (all at once: 0, 2, 4; curriculum: 3).
- Exact on short sums still isn't generalising: `k` 1's seed 2 passed its check (every sum of 1-3
  columns) in 3 epochs and failed at 8, as five of the matrix's `k` 1 runs did.

| cell | all at once: working, breaks (seeds 0-4) | epochs | curriculum: working, breaks after step 3 |
|---|---|---|---|
| `k` 1 `s` 16 `m` 8 | 2/5, 0/64/0/64/0 | 42, 1, 3, 3, 46 | 4/5, 64/64/64/0/64 |
| `k` 4 `s` 4 `m` 16 | 5/5, 64/64/64/48/64 | 13, 6, 7, 5, 6 | 4/5, 64/64/12/32/64 |
| `k` 4 `s` 16 `m` 16 | 5/5, 64/64/64/64/64 | 5, 4, 16, 6, 9 | 4/5, 64/24/64/64/0 |

Findings (E1 of docs/window-recurrent-claims-workplan.md: the certificate and the long tests), on
pyramidon: `certify` over every network the matrix and the control tested (the matrix's step 3, the
control's step 1: 105), 4-7 workers. Two verdicts (the claims workplan's D3): **robustly
certified**, every near tie (snap ratio under 1.05) branched both ways, indrajala-ml 094a7b1
(#703), 2026-10-13 about 07:50 to 08:26, in `data/window-recurrent/certificates/` (105 files,
listed with their sha256 in `data/window-recurrent/certificates.sha256`, itself sha256
67a4e6b7...9550f05d), log `data/window-recurrent-certificates.log` (sha256 f8c38080...56ae98a1); and
**certified**, the network's own snaps only (`--snap-ratio 1`), indrajala-ml 35175ea (#704; the
certificate unchanged but for the option), 08:30 to 09:22 in two parts, in
`data/window-recurrent/certificates-strict/` (`certificates-strict.sha256`, sha256
299675b8...a924fc18), log `data/window-recurrent-certificates-strict.all.log` (the two parts
joined; sha256 e1b8e612...3dcce757). 105.9 M window evaluations each way; 3.4 h of worker time
branched, 2.9 h strict; a `k` 4 network 115-689 s branched, 74-759 s strict.

- No failing network is certified, either way: of the 48 networks that break before 16 columns
  (W10's failing), none: here the certificate passes nothing the tests fail.
- Of the 57 working networks (W10), 52 are certified and 48 robustly certified (matrix 42 and 38
  of 45, control 10 and 10 of 12). Every working `k` 1 network is certified both ways; by `k` 1, 2
  and 4, working 18, 11 and 28, certified 18, 11 and 23, robustly 18, 8 and 22.
- Every certified network's machine minimises to 3 states, the carry automaton, while reaching 3-11
  codes (median 5; exactly 3 in 9 certified networks, 8 robustly): the extra codes are equivalent
  copies of a carry value. The smallest digit lead of a certified network
  is 0.153 (a logit difference).
- 16 robust certificates rest on near ties, up to 116,944 a network, at snap ratios down to
  1.0000002: real, not duplicate codes (each codebook's closest pair is 0.02-0.3 apart). The 4
  networks certified but not robustly (matrix `k2-s16-m16-seed3` and `seed4`, `k2-s4-m8-seed2`,
  `k4-s16-m4-seed1`) are correct as they compute and wrong only when a near tie (ratio
  1.006-1.03) snaps the other way.
- **The sampled tests overstate working (C1 refuted for W10's "works", D13).** 5 working networks
  aren't certified: 3 break before 64 columns, as their tests said (matrix `k4-s16-m16-seed1` at
  24, `k4-s4-m16-seed3` at 32, control `k4-s4-m16-seed3` at 48), and **2 passed every test to 64
  columns and are wrong**: control `all-at-once-k4-s16-m16-seed2` (1559 + 6329 + 5231 = 13,119
  at 8 columns; the network says 19,680) and matrix `curriculum-k4-s4-m8-seed2` (514107 + 167355
  + 27153 = 708,615 at 12; it says 531,468), each confirmed by running the network on
  `wr.example(triple, n - k, k, 0)`, `n` the certificate's length (a carry counterexample shows only at alignment 0, which the
  harness's hash picks for few cases).
- The failing networks' counterexamples (strict): 42 a wrong carry, 11 a wrong digit, after 1-4
  windows (22, 18, 6 and 7 networks), so every failure shows within 4 windows. Their machines
  minimise to 2-9 states.

LONG TESTS: to write when jebel's run ends.

Per cell, its networks working (W10), certified and robustly certified, of 5:

| arm | k | s | m | working | certified | robustly certified |
|---|---|---|---|---|---|---|
| curriculum | 1 | 4 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 4 | 8 | 3 | 3 | 3 |
| curriculum | 1 | 4 | 16 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 8 | 4 | 4 | 4 |
| curriculum | 1 | 16 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 4 | 4 | 1 | 1 | 1 |
| curriculum | 2 | 4 | 8 | 3 | 3 | 2 |
| curriculum | 2 | 4 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 16 | 4 | 0 | 0 | 0 |
| curriculum | 2 | 16 | 8 | 1 | 1 | 1 |
| curriculum | 2 | 16 | 16 | 3 | 3 | 1 |
| curriculum | 4 | 4 | 4 | 3 | 3 | 3 |
| curriculum | 4 | 4 | 8 | 3 | 2 | 2 |
| curriculum | 4 | 4 | 16 | 4 | 3 | 3 |
| curriculum | 4 | 16 | 4 | 2 | 2 | 1 |
| curriculum | 4 | 16 | 8 | 2 | 2 | 2 |
| curriculum | 4 | 16 | 16 | 4 | 3 | 3 |
| all at once | 1 | 16 | 8 | 2 | 2 | 2 |
| all at once | 4 | 4 | 16 | 5 | 4 | 4 |
| all at once | 4 | 16 | 16 | 5 | 4 | 4 |
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path

from indrajala_ml.data import addition_data as ad
from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
    check = commands.add_parser("check")
    check.add_argument("network", type=Path)
    check.add_argument("--n", type=int, nargs="+", required=True, help="the lengths to check")
    check.add_argument("--properties", nargs="+", help="these properties only (default: all but B3)")
    check.add_argument("--sampled", type=int, default=ad.SAMPLED, help="B1-B3, A3, M1 and M5's sample")
    check.add_argument("--per-stratum", type=int, default=ad.PER_STRATUM, help="M3's per distance, M4's per carry")
    check.add_argument("--doubling", action="store_true", help="M3 at distances 1, 2, 4, ..., n only")
    check.add_argument("--json", type=Path, help="write each length's results and seconds to this file")
    certify = commands.add_parser("certify")
    certify.add_argument("networks", type=Path, nargs="+")
    certify.add_argument("--out", type=Path, help="write each certificate here, as <run>-<step>.json")
    certify.add_argument("--workers", type=int, default=1)
    certify.add_argument(
        "--snap-ratio", type=float, default=wc.SNAP_RATIO, help="near ties branch under this ratio (1: strict)"
    )
    args = parser.parse_args(argv)

    if args.command == "check":
        catalogue = ad.catalogue(args.sampled, ad.RELATION_SAMPLED, args.per_stratum, doubling=args.doubling)
        properties = [p for p in wr.test_properties(catalogue) if args.properties is None or p.id in args.properties]
        network = load_network(str(args.network))
        model = wr.WindowAdder(network, network.columns)
        results: dict[str, object] = {}
        for n in args.n:
            start = time.perf_counter()
            checked = ad.evaluate(model.adder(n), n, properties, shrink=False)
            for result in checked:
                print(
                    f"n {n}, {result.id}: {result.passed}/{result.cases}"
                    + ("" if result.holds else f", e.g. {result.failures[0]}"),
                    flush=True,
                )
            results[str(n)] = {
                "properties": {r.id: wr.result_json(r) for r in checked},
                "seconds": time.perf_counter() - start,
            }
        if args.json is not None:
            args.json.write_text(json.dumps({"network": str(args.network), "lengths": results}, indent=1))
        return 0
    if args.command == "certify":
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for path, result in zip(
                args.networks,
                pool.map(functools.partial(wc.certify_file, snap_ratio=args.snap_ratio), args.networks),
                strict=True,
            ):
                name = f"{path.parent.name}-{path.name.removesuffix('.network.json')}"
                print(f"{name}: {wc.summary(result)}", flush=True)
                if args.out is not None:
                    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
        return 0
    study = wr.load_config(args.config)
    if args.command == "run":
        with machine_lock.study():
            wr.run(study, log=lambda line: print(line, flush=True))
        print(wr.report(study))
    elif args.command == "status":
        print(wr.status(study))
    else:
        text = wr.report(study)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
