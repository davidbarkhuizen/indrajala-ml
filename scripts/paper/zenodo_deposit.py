#!/usr/bin/env python3
"""
Paper 1's datapack to Zenodo (the paper workplan's D38), through Zenodo's deposit API; never
publishes. The owner reviews the draft in the browser and presses Publish, the irreversible step.

The token is a Zenodo personal access token with the scopes deposit:write and deposit:actions, read
from `~/.zenodo-token` (or `--token-file`); it is never printed or logged. The draft's id, DOI and
file bucket are kept in DIR's `zenodo.json`, so each command resumes the same draft.

`zenodo_deposit.py reserve DIR` creates a draft with a pre-reserved DOI and prints the DOI, to be
filled into the README, the paper and the letters before the upload. `zenodo_deposit.py metadata DIR`
sets the draft's metadata from docs/datapack/zenodo-metadata.json. `zenodo_deposit.py upload DIR`
uploads every file at DIR's top level (but `zenodo.json`; `datapack.py split` leaves the large
archives in parts there, the wholes in `whole/`), `--connections` at once, the largest first,
skipping those already there with the same MD5, and checks each against the checksum Zenodo reports. `zenodo_deposit.py status DIR` compares the
draft's files with DIR's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

API = "https://zenodo.org/api"
METADATA = Path(__file__).resolve().parents[2] / "docs" / "datapack" / "zenodo-metadata.json"


def request(method: str, url: str, token: str, body: Any = None, data: Any = None, size: int | None = None) -> Any:
    """One API call; a JSON body, or a file's bytes streamed (`data`, `size`)."""
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif data is not None:
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(size)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=3600) as response:
        text = response.read().decode()
    return json.loads(text) if text else None


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state(out: Path) -> dict[str, Any]:
    path = out / "zenodo.json"
    if not path.exists():
        raise SystemExit(f"{path}: no draft yet; run reserve first")
    return json.loads(path.read_text())


def local_files(out: Path) -> list[Path]:
    return sorted(p for p in out.iterdir() if p.is_file() and p.name != "zenodo.json")


def remote_files(draft: dict[str, Any], token: str) -> dict[str, str]:
    """The draft's files: name -> MD5."""
    files = request("GET", f"{API}/deposit/depositions/{draft['id']}/files", token)
    return {f["filename"]: f["checksum"].removeprefix("md5:") for f in files}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".zenodo-token")
    parser.add_argument("--connections", type=int, default=12, help="files uploaded at once")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reserve", "metadata", "upload", "status"):
        commands.add_parser(name).add_argument("dir", type=Path)
    args = parser.parse_args(argv)
    token = args.token_file.read_text().strip()
    out: Path = args.dir

    if args.command == "reserve":
        if (out / "zenodo.json").exists():
            raise SystemExit(f"{out / 'zenodo.json'}: a draft exists already ({state(out)['doi']})")
        made = request("POST", f"{API}/deposit/depositions", token, body={"metadata": {"prereserve_doi": True}})
        draft = {
            "id": made["id"],
            "doi": made["metadata"]["prereserve_doi"]["doi"],
            "bucket": made["links"]["bucket"],
            "html": made["links"]["html"],
        }
        (out / "zenodo.json").write_text(json.dumps(draft, indent=1) + "\n")
        print(f"draft {draft['id']}: DOI {draft['doi']} (reserved, not yet registered); {draft['html']}")
        return 0
    draft = state(out)
    if args.command == "metadata":
        metadata = json.loads(METADATA.read_text()) | {"prereserve_doi": True}
        if "[RELEASE TAG]" in json.dumps(metadata):
            raise SystemExit(f"{METADATA}: fill [RELEASE TAG] first")
        request("PUT", f"{API}/deposit/depositions/{draft['id']}", token, body={"metadata": metadata})
        print(f"draft {draft['id']}: metadata set; review it at {draft['html']}")
        return 0
    there = remote_files(draft, token)
    if args.command == "status":
        for path in local_files(out):
            mark = "uploaded" if there.get(path.name) == md5(path) else "missing or differs"
            print(f"{path.name}: {mark}")
        return 0

    def send(path: Path) -> str:
        local = md5(path)
        if there.get(path.name) == local:
            return f"{path.name}: there already"
        with path.open("rb") as f:
            made = request("PUT", f"{draft['bucket']}/{path.name}", token, data=f, size=path.stat().st_size)
        reported = made["checksum"].removeprefix("md5:")
        if reported != local:
            raise SystemExit(f"{path.name}: Zenodo reports MD5 {reported}, the file's is {local}")
        return f"{path.name}: uploaded, {path.stat().st_size / 1e9:.2f} GB, MD5 checked"

    with ThreadPoolExecutor(args.connections) as pool:  # Zenodo's connections are slow one by one
        for line in pool.map(send, sorted(local_files(out), key=lambda p: -p.stat().st_size)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
