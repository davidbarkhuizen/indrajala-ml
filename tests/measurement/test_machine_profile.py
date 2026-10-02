import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from indrajala_ml.measurement import machine_profile as mp
from indrajala_ml.measurement import machine_profile_capture as mpc

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "docs/machine_profiles/ryzen7-3700u.json"
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/machine_profile.py"


@pytest.fixture(scope="module")
def reference() -> mpc.JSONObject:
    return json.loads(REFERENCE_PATH.read_text())


# --- capture and schema ---


def test_capture_validates_on_this_machine():

    mp.validate(mpc.capture())


def test_reference_profile_validates(reference: mpc.JSONObject):

    mp.validate(reference)


def test_schema_is_a_valid_draft_2020_12_schema():

    jsonschema.Draft202012Validator.check_schema(mp.load_schema())


def test_schema_rejects_a_missing_required_key(reference: mpc.JSONObject):

    profile = copy.deepcopy(reference)
    del profile["identity"]["cpu"]["isa"]
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


def test_schema_rejects_an_unknown_key(reference: mpc.JSONObject):

    profile = copy.deepcopy(reference)
    profile["identity"]["os"]["colour"] = "blue"
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


def test_schema_rejects_a_wrong_schema_version(reference: mpc.JSONObject):

    profile = copy.deepcopy(reference)
    profile["schema_version"] = 2
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


# --- compare ---


def test_identical_identities_give_no_differences(reference: mpc.JSONObject):

    assert mp.compare(reference, copy.deepcopy(reference)) == []


@pytest.mark.parametrize(
    "path, value",
    [
        (["cpu", "frequency", "governor"], "performance"),
        (["cpu", "isa", "avx512f"], True),
        (["software", "numpy", "blas", "version"], "0.3.30"),
        (["software", "thread_env", "OPENBLAS_NUM_THREADS"], "1"),
    ],
)
def test_a_changed_identity_field_reports_exactly_its_path(reference: mpc.JSONObject, path: list[str], value: object):

    current = copy.deepcopy(reference)
    parent = current["identity"]
    for key in path[:-1]:
        parent = parent[key]
    old = parent[path[-1]]
    parent[path[-1]] = value

    assert mp.compare(reference, current) == [mp.Difference("identity." + ".".join(path), old, value)]


def test_a_changed_cache_inside_the_list_reports_its_index(reference: mpc.JSONObject):

    current = copy.deepcopy(reference)
    current["identity"]["cpu"]["caches"][2]["size_kib"] = 1024

    assert [d.path for d in mp.compare(reference, current)] == ["identity.cpu.caches[2].size_kib"]


def test_an_extra_gpu_is_reported(reference: mpc.JSONObject):

    current = copy.deepcopy(reference)
    current["identity"]["gpus"].append({"class": "3D controller", "vendor": "NVIDIA Corporation", "device": "GA107M"})

    differences = mp.compare(reference, current)

    assert [d.path for d in differences] == ["identity.gpus"]
    assert differences[0].current == current["identity"]["gpus"]


def test_state_changes_are_never_reported(reference: mpc.JSONObject):

    current = copy.deepcopy(reference)
    current["state"] = {
        "captured_at": "2030-01-01T00:00:00Z",
        "hostname": "elsewhere",
        "load_average": [7.0, 7.0, 7.0],
        "memory_available_mib": 1,
        "cpu_mhz_now": None,
        "power": {"on_ac": False, "battery_percent": 3},
        "commits": {"repo": None, "repo_dirty": None, "rust": None},
    }

    assert mp.compare(reference, current) == []


# --- CLI ---


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, check=False)


def test_cli_profile_then_compare_against_itself_and_an_edited_copy(tmp_path: Path):

    out = tmp_path / "profile.json"
    assert run_script("profile", "--out", str(out)).returncode == 0

    same = run_script("compare", str(out), str(out))
    assert same.returncode == 0, same.stdout + same.stderr
    assert "identity matches" in same.stdout

    edited = json.loads(out.read_text())
    edited["identity"]["os"]["kernel_release"] = "0.0.0-edited"
    edited_path = tmp_path / "edited.json"
    edited_path.write_text(json.dumps(edited))

    differs = run_script("compare", str(out), str(edited_path))
    assert differs.returncode == 1
    assert "identity.os.kernel_release: " in differs.stdout
    assert '-> "0.0.0-edited"' in differs.stdout
