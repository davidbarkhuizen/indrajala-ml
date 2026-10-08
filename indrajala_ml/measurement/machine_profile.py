"""
The benchmark machine's profile: a JSON record of the hardware, OS and software stack the documented
timings were measured on (machine_profile_capture.py builds it), a schema for it, and a comparison
of two profiles.

The profile has two parts:
- identity: what a timing depends on and should not change between a documented number and a
  new run (CPU, caches, ISA, cpufreq policy, memory, GPUs, OS, the Python/numpy/BLAS/Rust stack
  and the thread env vars). compare() reports every difference here.
- state: what is expected to change from run to run (clocks now, load, free memory, power,
  commits). It is recorded for context and never compared.

A recorded profile may also carry noise_rules, set by hand from the machine's A/As: the thresholds
scripts/ab.py reports by on that machine. They are never captured; `profile --out` onto an
existing profile keeps them.

    python scripts/machine_profile.py profile [--out FILE]
    python scripts/machine_profile.py compare REFERENCE [CURRENT]
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from indrajala_ml.measurement.machine_profile_capture import JSONObject, capture

SCHEMA_PATH = Path(__file__).with_name("machine_profile.schema.json")


# --- schema and comparison ---


def load_schema() -> JSONObject:
    return json.loads(SCHEMA_PATH.read_text())


def validate(profile: JSONObject) -> None:
    """Raise jsonschema.ValidationError if the profile doesn't match the schema."""
    import jsonschema

    jsonschema.validate(profile, load_schema(), cls=jsonschema.Draft202012Validator)


@dataclass(frozen=True)
class Difference:
    path: str
    reference: object
    current: object

    def __str__(self) -> str:
        return f"{self.path}: {json.dumps(self.reference)} -> {json.dumps(self.current)}"


def _as_json_object(value: object) -> dict[str, object] | None:
    # a parsed JSON value as an object (string keys), or None if it isn't one
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _as_json_list(value: object) -> list[object] | None:
    return cast("list[object]", value) if isinstance(value, list) else None


def _differences(path: str, reference: object, current: object) -> Iterator[Difference]:
    reference_object, current_object = _as_json_object(reference), _as_json_object(current)
    reference_list, current_list = _as_json_list(reference), _as_json_list(current)
    if reference_object is not None and current_object is not None:
        for key in sorted(set(reference_object) | set(current_object)):
            yield from _differences(f"{path}.{key}", reference_object.get(key), current_object.get(key))
    elif reference_list is not None and current_list is not None and len(reference_list) == len(current_list):
        for index, (ref_item, cur_item) in enumerate(zip(reference_list, current_list)):
            yield from _differences(f"{path}[{index}]", ref_item, cur_item)
    elif reference != current:
        yield Difference(path, reference, current)


def compare(reference: JSONObject, current: JSONObject) -> list[Difference]:
    """Every leaf of the identity that differs, as dotted paths. Lists of different lengths
    (an extra GPU or cache) are reported whole. State is never compared."""
    return list(_differences("identity", reference["identity"], current["identity"]))


def _flatten(prefix: str, value: object) -> Iterator[tuple[str, object]]:
    value_object = _as_json_object(value)
    if value_object is not None:
        for key, item in value_object.items():
            yield from _flatten(f"{prefix}.{key}", item)
    else:
        yield prefix, value


# --- CLI ---


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # None under -OO
    commands = parser.add_subparsers(dest="command", required=True)
    profile_parser = commands.add_parser("profile", help="capture this machine's profile")
    profile_parser.add_argument("--out", help="write the JSON here instead of stdout")
    compare_parser = commands.add_parser("compare", help="compare two profiles' identities; exit 1 if they differ")
    compare_parser.add_argument("reference", help="the reference profile (JSON)")
    compare_parser.add_argument("current", nargs="?", help="the profile to check (default: capture one now)")
    args = parser.parse_args(argv)

    if args.command == "profile":
        profile = capture()
        if args.out and Path(args.out).is_file():
            existing = json.loads(Path(args.out).read_text())
            if "noise_rules" in existing:
                profile["noise_rules"] = existing["noise_rules"]
        validate(profile)
        text = json.dumps(profile, indent=2) + "\n"
        if args.out:
            Path(args.out).write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    reference = json.loads(Path(args.reference).read_text())
    validate(reference)
    current = json.loads(Path(args.current).read_text()) if args.current else capture()
    validate(current)

    differences = compare(reference, current)
    current_state = dict(_flatten("state", current["state"]))
    print("state (recorded, not compared): reference | current")
    for path, value in _flatten("state", reference["state"]):
        print(f"  {path}: {json.dumps(value)} | {json.dumps(current_state.get(path))}")
    if differences:
        print(f"identity differs ({len(differences)}):")
        for difference in differences:
            print(f"  {difference}")
        return 1
    print("identity matches")
    return 0
