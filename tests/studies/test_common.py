import hashlib
import json
import re
from pathlib import Path

from indrajala_ml.studies import sequence_study
from indrajala_ml.studies.common import (
    crate_hash,
    default_out,
    mean_sd,
    provenance,
    read_results,
    table,
    write_results,
)


def test_mean_sd_formats_the_mean_and_sample_sd():
    assert mean_sd([0.9, 0.95, 1.0]) == "95.00% ± 5.00%"
    assert mean_sd([1.0, 2.0, 3.0], ".1f") == "2.0 ± 1.0"


def test_mean_sd_of_one_value_has_no_spread():
    assert mean_sd([0.5]) == "50.00% ± 0.00%"


def test_table_is_markdown_with_a_separator_row():
    assert table(["rate", "accuracy"], [["0.1", "90%"], ["0.2", "91%"]]) == (
        "| rate | accuracy |\n|---|---|\n| 0.1 | 90% |\n| 0.2 | 91% |"
    )


def test_a_default_result_file_is_under_data_studies_by_study_stage_and_time():
    path = default_out("sequence", "sweep")
    assert path.parent == Path("data/studies/sequence")
    assert re.fullmatch(r"sweep-\d{4}-\d{2}-\d{2}-\d{4}\.json", path.name)


def test_provenance_names_the_commit_crate_machine_and_command():
    made_by = provenance(["sweep", "--epochs", "10"])
    assert set(made_by) == {"commit", "dirty", "crate", "machine", "command", "started"}
    assert made_by["command"].endswith("sweep --epochs 10") and made_by["crate"] == crate_hash()


def test_the_crate_hash_is_the_extension_modules_not_the_packages():
    import indrajala_math_rust

    package = Path(indrajala_math_rust.__file__).parent
    (extension,) = package.glob("indrajala_math_rust*.so")
    assert crate_hash() == hashlib.sha256(extension.read_bytes()).hexdigest()[:12]
    assert crate_hash() != hashlib.sha256((package / "__init__.py").read_bytes()).hexdigest()[:12]


def test_a_result_file_holds_its_provenance_and_reads_back(tmp_path: Path):
    results = {("euclid-heath", "2-layer", 0.002): [{"loss": 1.0}, {"loss": 2.0}]}
    path = write_results(tmp_path / "a/b/sweep.json", {"commit": "abc1234"}, results)
    data = json.loads(path.read_text())
    assert data["provenance"] == {"commit": "abc1234"}
    assert read_results(path) == [
        {"config": ["euclid-heath", "2-layer", 0.002], "runs": [{"loss": 1.0}, {"loss": 2.0}]}
    ]


def test_a_file_from_before_provenance_reads_as_it_is(tmp_path: Path):
    entries = [{"config": ["tinyshakespeare", "1-layer", 0.004], "runs": [{"loss": 3.0}]}]
    (tmp_path / "old.json").write_text(json.dumps(entries))
    assert read_results(tmp_path / "old.json") == entries


def test_the_sequence_studys_tuned_rates_read_either_form(tmp_path: Path):
    results = {("tinyshakespeare", "1-layer", 0.004): [{"loss": 3.0}]}
    new = write_results(tmp_path / "new.json", {}, results)
    (tmp_path / "old.json").write_text(
        json.dumps([{"config": ["tinyshakespeare", "1-layer", 0.004], "runs": [{"loss": 3.0}]}])
    )
    assert sequence_study.read_results(str(new)) == sequence_study.read_results(str(tmp_path / "old.json")) == results
