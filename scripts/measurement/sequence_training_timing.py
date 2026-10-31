"""
Seconds per training epoch of the sequence models: the rotary study's 2-layer transformer (rotary
attention, no Position, the default since its D9) on the first windows of Tiny Shakespeare, Adam
at batch 32, on numpy and Rust, each run in its own process and the order rotated each repeat. A
tier 1 benchmark for changes to the training step of attention models (docs/measurement.md, §1),
run through ab.py as a probe:

    python scripts/measurement/ab.py run --bench cmd -- scripts/measurement/sequence_training_timing.py

    python scripts/measurement/sequence_training_timing.py [--windows 1024] [--repeats 3] [--epochs 1]

It prints one JSON row per (backend, metric): the epoch's seconds in learn_batch (the minimum over
the repeats) and, as a check that both sides train alike, the held-out cross-entropy after the
epochs. OPENBLAS_NUM_THREADS=1, as the studies run. Only the package's long-standing API is used
(the study modules' arm_specs, load, train_epoch), so the script times either side of an A/B.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from typing import Any

BACKENDS = ["numpy", "rust"]
CORPUS = "tinyshakespeare"
RATE = 0.001


def worker(backend: str, windows: int, epochs: int) -> dict[str, Any]:
    from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
    from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
    from indrajala_ml.model.specs.update_rules import Adam
    from indrajala_ml.studies import batch_size_scaling as bss
    from indrajala_ml.studies import rotary_study
    from indrajala_ml.studies import sequence_study as ss
    from indrajala_ml.training.sequence_evaluate import sequence_evaluate

    train, held_out, vocabulary = ss.load(CORPUS, windows)
    array_backend = NUMPY if backend == "numpy" else RUST
    network: Any = SequentialArrayNetwork(
        (ss.CONTEXT,), rotary_study.arm_specs("2-layer", vocabulary), Adam(), "sequence", array_backend
    )
    network.rng = array_backend.default_rng(0)
    network.randomize()
    rng = random.Random(0)
    step, seconds = 0, list[float]()
    for _ in range(epochs):
        steps, epoch_seconds = bss.train_epoch(network, train, ss.BATCH_SIZE, RATE, step, rng)
        step += steps
        seconds.append(epoch_seconds)
    return {"seconds": seconds, "held_out": sequence_evaluate(network, held_out[:256]).cross_entropy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--worker", metavar="BACKEND", choices=BACKENDS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(worker(args.worker, args.windows, args.epochs)))
        return

    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    runs: dict[str, list[dict[str, Any]]] = {backend: [] for backend in BACKENDS}
    for repeat in range(args.repeats):
        order = BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]
        for backend in order:
            command = [sys.executable, "-B", __file__, "--worker", backend]
            command += ["--windows", str(args.windows), "--epochs", str(args.epochs)]
            output = subprocess.run(command, env=environment, capture_output=True, text=True, check=True).stdout
            runs[backend].append(json.loads(output.strip().splitlines()[-1]))
    for backend in BACKENDS:
        case = f"2-layer rotary {backend} W={args.windows}"
        seconds = min(min(run["seconds"]) for run in runs[backend])
        print(json.dumps({"case": case, "metric": "epoch", "value": seconds, "unit": "s"}))
        held_out = runs[backend][0]["held_out"]
        print(json.dumps({"case": case, "metric": "held-out nats", "value": held_out, "unit": "nats"}))


if __name__ == "__main__":
    main()
