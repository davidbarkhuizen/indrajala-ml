"""
The Hub functions (model/hub.py, the export-and-hub workplan, stage 2), offline: huggingface_hub's
repository creation, upload and download replaced by a repository in a temporary directory, one
folder a commit. A pushed network comes back by revision with its bits, a resumable one trains on
by bits, an inference download leaves the training state behind, and the model card carries
library_name. The live round trip runs only with HF_TOKEN and INDRAJALA_HUB_TEST_REPO set (O1).
"""

import fnmatch
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import pytest

from indrajala_ml.generation.tokenizer import CharacterTokenizer
from indrajala_ml.model.hub import (
    CARD_FILE,
    LIBRARY_NAME,
    download_pretrained,
    from_pretrained,
    model_card,
    provenance,
    push_to_hub,
    save_pretrained,
)
from indrajala_ml.model.persistence.format3 import (
    CONFIG_FILE,
    OPTIMIZER_FILE,
    TENSORS_FILE,
    TOKENIZER_FILE,
    TRAINING_STATE_FILE,
)
from tests.helpers import bits
from tests.model.persistence.test_checkpoint import _state_bits
from tests.model.persistence.test_format2 import _examples, _trained
from tests.saved_model_fixtures import FIXTURES

NAME = "WindowRecurrentRustArrayNetwork"


@dataclass
class _Commit:
    oid: str


class FakeHub:
    """One repository's commits, each a folder under root; the last is the branch's head."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commits: list[str] = []
        self.created: list[tuple[str, bool | None]] = []

    def create_repo(self, repo_id: str, *, private: bool | None = None, exist_ok: bool = False, **_: Any) -> str:
        self.created.append((repo_id, private))
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(
        self, *, repo_id: str, folder_path: str, commit_message: str, delete_patterns: Any = None, **_: Any
    ) -> _Commit:
        del repo_id, commit_message
        oid = f"{len(self.commits):040x}"
        commit = self.root / oid
        if self.commits:
            shutil.copytree(self.root / self.commits[-1], commit)
            for path in list(commit.iterdir()):
                if any(fnmatch.fnmatch(path.name, pattern) for pattern in delete_patterns or ()):
                    path.unlink()
        else:
            commit.mkdir(parents=True)
        for path in Path(folder_path).iterdir():
            shutil.copy(path, commit / path.name)
        self.commits.append(oid)
        return _Commit(oid)

    def snapshot_download(
        self, repo_id: str, *, revision: str | None = None, allow_patterns: Any = None, **_: Any
    ) -> str:
        del repo_id
        source = self.root / (revision or self.commits[-1])
        target = self.root / "downloads" / f"{revision}-{len(list((self.root).glob('downloads/*')))}"
        target.mkdir(parents=True)
        for path in source.iterdir():
            if allow_patterns is None or any(fnmatch.fnmatch(path.name, pattern) for pattern in allow_patterns):
                shutil.copy(path, target / path.name)
        return str(target)


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    hub = FakeHub(tmp_path / "hub")

    class Api:
        def __init__(self, token: str | None = None) -> None:
            del token

        create_repo = staticmethod(hub.create_repo)
        upload_folder = staticmethod(hub.upload_folder)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub.snapshot_download)
    return hub


def _train_on(network: Any, seed: int) -> None:
    examples = _examples(network, FIXTURES[NAME].predict, 6, seed=seed)
    network.learn_batch(0.1, examples[:3])
    network.learn_batch(0.1, examples[3:])


def test_save_pretrained_writes_the_directory_and_its_card(tmp_path: Path):
    network = _trained(NAME)
    save_pretrained(network, tmp_path / "model")

    assert sorted(path.name for path in (tmp_path / "model").iterdir()) == [CARD_FILE, CONFIG_FILE, TENSORS_FILE]
    data = huggingface_hub.ModelCard.load(tmp_path / "model" / CARD_FILE).data.to_dict()
    assert data["library_name"] == LIBRARY_NAME
    assert data["license"] == "mit"
    assert "window-recurrent" in data["tags"]
    assert bits(from_pretrained(tmp_path / "model").snapshot()) == bits(network.snapshot())


def test_the_card_names_the_commits_and_the_size():
    network = _trained(NAME)
    text = str(model_card(network, made_by={"ml": "abc123", "crate": "def456"}, description="A note."))

    assert "`abc123`" in text
    assert "`def456`" in text
    assert f"| parameters | {sum(a.size for entry in network.snapshot() for a in _numpy(entry)):,} |" in text
    assert text.rstrip().endswith("A note.")


def _numpy(entry: Any) -> list[Any]:
    return [np.array(array.tolist()) for array in entry]


def test_provenance_names_this_checkout():
    made_by = provenance()
    assert made_by["ml"] != "unknown"
    assert len(made_by["ml"].removesuffix("-dirty")) == 40


def test_a_push_comes_back_by_revision(fake_hub: FakeHub):
    network = _trained(NAME)
    first = push_to_hub(network, "someone/model")
    first_bits = bits(network.snapshot())
    _train_on(network, 3)
    second = push_to_hub(network, "someone/model")

    assert fake_hub.created == [("someone/model", True), ("someone/model", True)]
    assert first != second
    assert bits(from_pretrained("someone/model", revision=first).snapshot()) == first_bits
    assert bits(from_pretrained("someone/model", revision=second).snapshot()) == bits(network.snapshot())
    assert bits(from_pretrained("someone/model").snapshot()) == bits(network.snapshot())


@pytest.mark.usefixtures("fake_hub")
def test_a_resumable_push_trains_on_by_bits():
    network = _trained(NAME)
    revision = push_to_hub(network, "someone/model", training_state=True)

    loaded = from_pretrained("someone/model", revision=revision, training_state=True)
    assert _state_bits(loaded) == _state_bits(network)
    _train_on(network, 3)
    _train_on(loaded, 3)

    assert _state_bits(loaded) == _state_bits(network)


@pytest.mark.usefixtures("fake_hub")
def test_an_inference_download_leaves_the_training_state():
    revision = push_to_hub(_trained(NAME), "someone/model", training_state=True)

    directory = download_pretrained("someone/model", revision=revision)

    assert sorted(path.name for path in directory.iterdir()) == [CARD_FILE, CONFIG_FILE, TENSORS_FILE]
    assert from_pretrained("someone/model", revision=revision).optimizer.state().t == 0


def test_a_push_deletes_our_files_it_no_longer_holds(fake_hub: FakeHub):
    network = _trained("SequentialSequenceArrayNetwork")
    push_to_hub(network, "someone/model", training_state=True, tokenizer=CharacterTokenizer.of("t", "abcdefg"))
    push_to_hub(network, "someone/model")

    head = fake_hub.root / fake_hub.commits[-1]
    assert sorted(path.name for path in head.iterdir()) == [CARD_FILE, CONFIG_FILE, TENSORS_FILE]
    assert not {OPTIMIZER_FILE, TRAINING_STATE_FILE, TOKENIZER_FILE} & {path.name for path in head.iterdir()}


@pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") and os.environ.get("INDRAJALA_HUB_TEST_REPO")),
    reason="the live Hub: HF_TOKEN and INDRAJALA_HUB_TEST_REPO (a private repository's id, O1)",
)
def test_the_live_hub_round_trip():
    network = _trained(NAME)
    revision = push_to_hub(network, os.environ["INDRAJALA_HUB_TEST_REPO"], training_state=True)

    loaded = from_pretrained(os.environ["INDRAJALA_HUB_TEST_REPO"], revision=revision, training_state=True)

    assert _state_bits(loaded) == _state_bits(network)


def test_a_window_networks_card_warns_against_float32():
    text = str(model_card(_trained(NAME), made_by={"ml": "a", "crate": "b"}))
    assert "**Float64 required: a conversion to float32" in text
    assert "may change the machine.**" in text
    assert "float64 required" not in str(model_card(_trained("SequentialSequenceArrayNetwork"))).lower()
