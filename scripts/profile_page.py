"""
A crystallisation's course as one self-contained page (the state formation workplan's option 1): each
profiled run's epochs from its log.jsonl, read as a time series over the epochs trained, its phases
(warm-up, heat, hold, cool, hard) shaded: each factor's temperature, heat capacity and energy
(`thermals`, D32), its codes' groups (D20) and coupling (D25), and the run's screen with its
diagnoses' state share and certificate. Two views: stacked panels on one epoch axis, and an overlay of
one factor's quantities, each rescaled to its own range; and each factor's phase diagram (its codes
against temperature, from the run's saved step networks, beside the melting curve).

    .venv/bin/python scripts/profile_page.py --out page.html NAME=path/to/log.jsonl ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from indrajala_ml.studies import modular_family as mf


def course(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's epochs as columns (x the epochs trained so far), its diagnoses and its marks."""
    epochs = [r for r in rows if r.get("epoch", 0) > 0 and "screen" in r and "end" not in r]
    factors = max((len(r["anneal"]["temperature"]) for r in epochs if "anneal" in r), default=0)
    columns: dict[str, Any] = {key: [] for key in ("x", "step", "epoch", "phase", "screen", "held")}
    for key in ("T", "C", "E", "G", "L"):
        columns[key] = [[] for _ in range(factors)]
    diagnoses: list[dict[str, Any]] = []
    marks: list[dict[str, Any]] = []
    trained = {id(r) for r in epochs}
    x = 0
    for r in rows:
        if id(r) in trained:
            x += 1
            a = r.get("anneal", {})
            t = a.get("temperature", [None] * factors)
            phase = "warm-up" if r.get("warm") else a.get("phase") or ("hard" if a else "train")
            if phase == "cool" and all(v is None for v in t):
                phase = "hard"
            columns["x"].append(x)
            columns["step"].append(r["step"])
            columns["epoch"].append(r["epoch"])
            columns["phase"].append(phase)
            columns["screen"].append(r["screen"])
            columns["held"].append(a.get("held"))
            soft = "phase" in a
            for key, logged in (("T", "temperature"), ("C", "capacity"), ("E", "energy"), ("G", "groups")):
                values = a.get(logged) if soft else None
                for g in range(factors):
                    columns[key][g].append(None if values is None or t[g] is None else values[g])
            couplings = a.get("coupling")
            for g in range(factors):
                columns["L"][g].append(None if couplings is None else couplings[g])
        elif "certified" in r:
            diagnoses.append(
                {
                    "x": x,
                    "step": r["step"],
                    "share": r.get("state_share"),
                    "certified": r["certified"],
                    "end": r.get("end"),
                }
            )
        elif "warm_up_end" in r:
            marks.append({"x": x, "label": f"warm-up ends (step {r['step']})"})
        elif r.get("crucible") == "grown" and r.get("epoch") == 0 and not any(m["label"] == "growth" for m in marks):
            marks.append({"x": x, "label": "growth"})
    return {"factors": factors, "columns": columns, "diagnoses": diagnoses, "marks": marks}


def diagram(rows: list[dict[str, Any]], directory: Path) -> list[dict[str, Any]]:
    """
    Each factor's phase diagram (Rose's bifurcation diagram): its codes at each profiled step's end
    (the step's saved network), projected on the first principal axis of all of them, against the
    temperature its last epoch trained at; with the melting curve measured at the warm-up's end (D21)
    and training's own groups (D20) against temperature.
    """
    start = next((r for r in rows if "warm_up_end" in r), None)
    if start is None:
        return []
    last: dict[int, dict[str, Any]] = {}  # each step's last profiled epoch's anneal
    for r in rows:
        if r.get("epoch", 0) > 0 and "phase" in r.get("anneal", {}):
            last[r["step"]] = r["anneal"]
    factors = len(start["warm_up_end"])
    codes: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(factors)]
    for step, a in sorted(last.items()):
        path = directory / f"step-{step}.network.json"
        if not path.exists():
            certified = directory / f"certified-step-{step}.network.json"
            if not certified.exists():
                continue
            path = certified
        net = mf.load_network(str(path))
        for g, (factor, t) in enumerate(zip(net.recurrence_layer.factors, a["temperature"], strict=True)):
            if t is not None and factor.codes is not None:
                codes[g].append((float(t), step, np.array(factor.codes.E.tolist(), dtype=float)))
    out: list[dict[str, Any]] = []
    for g, made in enumerate(start["warm_up_end"]):
        if not codes[g]:
            out.append({})
            continue
        pooled = np.concatenate([E for _, _, E in codes[g]])
        centre = pooled.mean(axis=0)
        _, s, vt = np.linalg.svd(pooled - centre, full_matrices=False)
        axis = vt[0]
        points = [[round(t, 6), step, round(float(y), 5)] for t, step, E in codes[g] for y in (E - centre) @ axis]
        training = [[a["temperature"][g], a["groups"][g]] for a in last.values() if a["temperature"][g] is not None]
        out.append(
            {
                "points": points,
                "explained": round(float(s[0] ** 2 / max(float((s**2).sum()), 1e-300)), 3),
                "melting": made.get("melting_point"),
                "critical": made.get("critical_temperature"),
                "curve": made.get("transitions") or [],
                "training": training,
            }
        )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=LOG", help="a name for the run and its log.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Crystallisation Profiles")
    args = parser.parse_args(argv)
    runs: list[dict[str, Any]] = []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        reading = mf.heat_summary(mf.heat_course(rows))
        runs.append({"name": name, "reading": reading, **course(rows), "diagram": diagram(rows, Path(path).parent)})
    template = (Path(__file__).parent / "profile_page.html").read_text()
    page = template.replace("__TITLE__", args.title).replace(
        "__DATA__", json.dumps({"runs": runs}, separators=(",", ":"))
    )
    args.out.write_text(page)
    print(f"{args.out}: {len(runs)} runs, {len(page) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
