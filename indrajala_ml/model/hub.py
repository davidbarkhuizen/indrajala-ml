"""
The Hugging Face Hub (the export-and-hub workplan, D5): free functions over huggingface_hub's plain
API, writing and reading format-3 directories (persistence/format3.py).

- save_pretrained(network, directory): the directory, and its model card, README.md.
- push_to_hub(network, repo_id): the same directory, uploaded as one commit to a repository
  created private (O1: every push is outward-facing); returns the commit's id.
- from_pretrained(repo_id_or_directory, revision=...): the network a local directory or a Hub
  repository at revision describes, built from its specs as load_network builds (never from a
  class named in a downloaded file). revision pins the exact weights, as a DOI does.
- download_pretrained: the local directory from_pretrained reads, for its tokenizer
  (generation/tokenizer.tokenizer_path) or for export.
- onnx=True (save_pretrained, push_to_hub) adds the network's checked ONNX export, model.onnx,
  and tags the card onnx (stage 8): the repository is usable without this library. A download
  for our library leaves it behind; it is the file for every other runtime.

No ModelHubMixin: it overrides __new__ and writes config.json from constructor arguments, neither
of which fits our networks (D5). The model card carries library_name: indrajala-ml, the network's
shape, implementation and parameter count, and the ml and crate commits it was saved at.

huggingface_hub is the [hub] extra's (D7), imported only when used.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from indrajala_ml.model.export.onnx.exporter import FLOAT64_WARNING, export_onnx
from indrajala_ml.model.export.onnx.graph import Dtype
from indrajala_ml.model.persistence.format2 import WINDOW_RECURRENT
from indrajala_ml.model.persistence.format3 import (
    CONFIG_FILE,
    OPTIMIZER_FILE,
    TENSORS_FILE,
    TOKENIZER_FILE,
    TRAINING_STATE_FILE,
    Tokenizer,
    is_directory,
    save_directory,
)
from indrajala_ml.model.persistence.load_network import load_network

LIBRARY_NAME = "indrajala-ml"
CARD_FILE = "README.md"
_HUB_EXTRA = "pip install 'indrajala-ml[hub]'"
# what an inference download takes; a resumable one adds the training state (D4)
# a repository's ONNX export (stage 8): the model, and an encoder-decoder's other two beside it
ONNX_FILE = "model.onnx"
ONNX_FILES = (ONNX_FILE, "model_encoder.onnx", "model_decoder_step.onnx")
INFERENCE_FILES = (CONFIG_FILE, TENSORS_FILE, TOKENIZER_FILE, CARD_FILE)
TRAINING_FILES = (OPTIMIZER_FILE, TRAINING_STATE_FILE)


def _hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as error:
        raise ImportError(f"the Hub functions need huggingface_hub, the [hub] extra's: {_HUB_EXTRA}") from error
    return huggingface_hub


# the model card


def _git(directory: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def provenance() -> dict[str, str]:
    """
    The ml commit (with "-dirty" if its tracked files differ from it) and the crate's commit, from
    the checkout the package runs from; "unknown" for an installed package outside a checkout.
    """
    package = Path(__file__).resolve().parents[1]
    commit = _git(package, "rev-parse", "HEAD")
    if commit and _git(package, "status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    crate = _git(package.parent / "rust", "rev-parse", "HEAD") if (package.parent / "rust").is_dir() else ""
    return {"ml": commit or "unknown", "crate": crate or "unknown"}


def _parameter_count(network: Any) -> int:
    networks = getattr(network, "classifiers", None) or [network]
    count = 0
    for each in networks:
        for entry in each.snapshot():
            for array in entry:
                shape = getattr(array, "shape", ())
                size = 1
                for extent in shape:
                    size *= extent
                count += size
    return count


def model_card(
    network: Any,
    *,
    license: str = "mit",
    description: str = "",
    tags: tuple[str, ...] = (),
    made_by: Mapping[str, str] | None = None,
    onnx: bool = False,
) -> Any:
    """
    network's model card (huggingface_hub.ModelCard): its metadata (library_name, license, tags)
    and a body naming its shape, implementation, size and commits, description after them.
    made_by is the commits it was trained at, provenance()'s when not given.
    """
    hub = _hub()
    classifiers = getattr(network, "classifiers", None)
    first = classifiers[0] if classifiers else network
    shape = "ensemble" if classifiers else first.format2_shape
    made_by = dict(made_by or provenance())
    data = hub.ModelCardData(
        library_name=LIBRARY_NAME,
        license=license,
        tags=[LIBRARY_NAME, shape.replace("_", "-"), *(["onnx"] if onnx else []), *tags],
    )
    lines = [
        "---",
        data.to_yaml(),
        "---",
        "",
        f"# {shape.replace('_', ' ')} network",
        "",
        (
            f"A {shape.replace('_', '-')} network saved by [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml)"
            " in its format 3: `config.json` describes it, `model.safetensors` holds its float64 weights."
        ),
        "",
        "| | |",
        "|---|---|",
        f"| implementation | {first.implementation} |",
        f"| parameters | {_parameter_count(network):,} |",
        f"| indrajala-ml commit | `{made_by.get('ml', 'unknown')}` |",
        f"| crate commit | `{made_by.get('crate', 'unknown')}` |",
        "",
        "```python",
        "from indrajala_ml.model.hub import from_pretrained",
        "",
        'network = from_pretrained("<repo id>", revision="<commit>")',
        "```",
    ]
    if shape == WINDOW_RECURRENT:
        # its ONNX step is float64 only (the export-and-hub workplan, D10)
        lines += ["", f"**{FLOAT64_WARNING[0].upper()}{FLOAT64_WARNING[1:]}.**"]
    if description:
        lines += ["", description.strip()]
    return hub.ModelCard("\n".join(lines) + "\n")


# saving and loading


def save_pretrained(
    network: Any,
    directory: str | Path,
    *,
    training_state: bool = False,
    tokenizer: Tokenizer | None = None,
    card: Any = None,
    onnx: bool = False,
    onnx_dtype: Dtype | None = None,
) -> Path:
    """
    network as a format-3 directory (format3.save_directory) with its model card, README.md: card,
    or model_card(network)'s. With onnx, also its checked ONNX export (export_onnx, onnx_dtype's
    precision) as model.onnx, an encoder-decoder's encoder and decode step beside it, and the card
    tagged onnx (the export-and-hub workplan, stage 8): the repository is then usable without
    this library. Without it, a model.onnx already there is removed.
    """
    directory = save_directory(network, directory, training_state=training_state, tokenizer=tokenizer)
    for name in ONNX_FILES:
        (directory / name).unlink(missing_ok=True)
    if onnx:
        export_onnx(network, directory / ONNX_FILE, dtype=onnx_dtype)
    (card if card is not None else model_card(network, onnx=onnx)).save(str(directory / CARD_FILE))
    return directory


def push_to_hub(
    network: Any,
    repo_id: str,
    *,
    training_state: bool = False,
    tokenizer: Tokenizer | None = None,
    card: Any = None,
    commit_message: str = "Upload an indrajala-ml network",
    token: str | None = None,
    onnx: bool = False,
    onnx_dtype: Dtype | None = None,
) -> str:
    """
    network's directory (save_pretrained's, with its ONNX export when onnx) uploaded to repo_id as
    one commit, the repository created private if it doesn't exist; returns the commit's id,
    from_pretrained's revision. Files of ours the directory no longer holds (a training state, a
    tokenizer, an export) are deleted.
    """
    hub = _hub()
    api = hub.HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        save_pretrained(
            network,
            folder,
            training_state=training_state,
            tokenizer=tokenizer,
            card=card,
            onnx=onnx,
            onnx_dtype=onnx_dtype,
        )
        held = {path.name for path in Path(folder).iterdir()}
        stale = [name for name in (*TRAINING_FILES, TOKENIZER_FILE, *ONNX_FILES) if name not in held]
        commit = api.upload_folder(
            repo_id=repo_id, folder_path=folder, commit_message=commit_message, delete_patterns=stale or None
        )
    return str(commit.oid)


def download_pretrained(
    repo_id_or_directory: str | Path,
    *,
    revision: str | None = None,
    training_state: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """
    A local format-3 directory: repo_id_or_directory itself if it is one, else the Hub repository's
    snapshot at revision, its training state downloaded only with training_state (D4).
    """
    if is_directory(repo_id_or_directory):
        return Path(repo_id_or_directory)
    hub = _hub()
    patterns = [*INFERENCE_FILES, *(TRAINING_FILES if training_state else ())]
    path = hub.snapshot_download(
        str(repo_id_or_directory),
        revision=revision,
        allow_patterns=patterns,
        token=token,
        cache_dir=None if cache_dir is None else str(cache_dir),
        library_name=LIBRARY_NAME,
    )
    return Path(path)


def from_pretrained(
    repo_id_or_directory: str | Path,
    *,
    revision: str | None = None,
    training_state: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """
    The network or ensemble a local format-3 directory, or a Hub repository at revision,
    describes (load_network's); with training_state, ready to train on by bits.
    """
    return load_network(
        download_pretrained(
            repo_id_or_directory, revision=revision, training_state=training_state, token=token, cache_dir=cache_dir
        )
    )
