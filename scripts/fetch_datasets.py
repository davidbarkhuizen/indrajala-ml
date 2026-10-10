"""
Ensures this checkout's dataset source files are present and checksum-verified, fetching from
their pinned `indrajala-datasets-*` repo only when a file is missing or doesn't match its
expected SHA-256.

Presence+checksum is checked first, network only as a last resort: a repeated `./cli setup`/
`./cli fetch-data` against an unchanged local checkout performs zero network calls after the
first successful fetch. MNIST and the sequence task's corpora are fetched here - UCI digits' digits.csv stays committed
directly in `indrajala-ml`, with its own `indrajala-datasets-uci-digits` packaging existing for
metadata consistency, not because indrajala-ml needs to fetch it.

Also regenerates each MNIST file's derived `.bin` (via `mnist_data.convert_parquet_to_binary`) if it's
missing: `tests/data/test_mnist_data.py` reads the `.bin` files directly, so they need to exist
before the test suite runs. `.bin` files are gitignored, regenerable artifacts (see
`.gitignore`'s own comment on `data/mnist/*.bin`), so this only ever runs the conversion once
per fresh checkout, exactly like the fetch above.
"""

import hashlib
import os
import sys
import urllib.request

# so `indrajala_ml.data.mnist_data` imports regardless of cwd - this script is invoked as
# `python scripts/fetch_datasets.py` from the repo root, which puts scripts/ (not the repo root)
# on sys.path by default.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PINNED_REF = "v2026-09-16"
_RAW_BASE = f"https://raw.githubusercontent.com/davidbarkhuizen/indrajala-datasets-mnist/{PINNED_REF}/data"
TEXT_PINNED_REF = "v2026-10-09"


def _text_url(repository: str, filename: str) -> str:
    # one of the sequence task's corpora, from its indrajala-datasets-* repository at the pinned tag
    return f"https://raw.githubusercontent.com/davidbarkhuizen/{repository}/{TEXT_PINNED_REF}/data/{filename}"


DATASETS = [
    {
        "local_path": "data/mnist/mnist-train.parquet",
        "binary_path": "data/mnist/mnist-train.bin",
        "sha256": "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
        "url": f"{_RAW_BASE}/mnist-train.parquet",
    },
    {
        "local_path": "data/mnist/mnist-test.parquet",
        "binary_path": "data/mnist/mnist-test.bin",
        "sha256": "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
        "url": f"{_RAW_BASE}/mnist-test.parquet",
    },
    {
        # the sequence task's corpus (the sequence task workplan, D2, D3): read as text, no conversion
        "local_path": "data/tinyshakespeare/tinyshakespeare.txt",
        "sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        "url": _text_url("indrajala-datasets-tinyshakespeare", "tinyshakespeare.txt"),
    },
    {
        "local_path": "data/herodotus-rawlinson/herodotus-rawlinson.txt",
        "sha256": "ef4c270ac327f310e667cabf736ab75c479aee346163239b8b21759479ad6052",
        "url": _text_url("indrajala-datasets-herodotus-rawlinson", "herodotus-rawlinson.txt"),
    },
    {
        # CC BY-NC-SA 4.0 (the repository's LICENSE-TEXT.md): fetched for study, never redistributed here
        "local_path": "data/muqaddimah/muqaddimah.txt",
        "sha256": "883cd692c518cb84901bdcd3d46552279065be865af3a6f15400a075a2249715",
        "url": _text_url("indrajala-datasets-muqaddimah", "muqaddimah.txt"),
    },
    {
        # CC BY-SA 4.0 (the repository's LICENSE-TEXT.md), adapted from the Perseus Digital Library
        "local_path": "data/euclid-heath/euclid-heath.txt",
        "sha256": "5b2a8d39410a422b2fca03f682030dc978f585a03ffa6e83ba6c84290ef2b2d6",
        "url": _text_url("indrajala-datasets-euclid-heath", "euclid-heath.txt"),
    },
]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_file(local_path: str, expected_sha256: str, fetch_url: str) -> None:
    if os.path.exists(local_path) and sha256_of(local_path) == expected_sha256:
        print(f"{local_path}: present and verified, skipping fetch")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"{local_path}: fetching from {fetch_url} ...")
    urllib.request.urlretrieve(fetch_url, local_path)

    actual_sha256 = sha256_of(local_path)
    assert actual_sha256 == expected_sha256, (
        f"{local_path}: downloaded but checksum didn't match (expected {expected_sha256}, got {actual_sha256})"
    )
    print(f"{local_path}: fetched and verified")


def ensure_binary_conversion(parquet_path: str, binary_path: str) -> None:
    if os.path.exists(binary_path):
        print(f"{binary_path}: already present, skipping conversion")
        return

    from indrajala_ml.data.mnist_data import convert_parquet_to_binary

    print(f"{binary_path}: converting from {parquet_path} ...")
    convert_parquet_to_binary(parquet_path, binary_path)
    print(f"{binary_path}: converted")


def main() -> None:
    for dataset in DATASETS:
        ensure_dataset_file(dataset["local_path"], dataset["sha256"], dataset["url"])
        if "binary_path" in dataset:
            ensure_binary_conversion(dataset["local_path"], dataset["binary_path"])


if __name__ == "__main__":
    main()
