"""Fetch the model bundle at startup for hosts with no mounted artifacts, such as Render."""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def ensure_artifacts(directory: Path, url: str | None, sha256: str | None) -> None:
    """Download and unpack a .tar.gz bundle when the directory is empty. The checksum is mandatory
    because this archive supplies the models the service will load and execute."""
    if not url or (directory.is_dir() and any(directory.iterdir())):
        return
    if not sha256:
        raise RuntimeError("ARTIFACTS_SHA256 is required when ARTIFACTS_URL is set")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        bundle = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, bundle)
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if digest != sha256.lower():
            raise RuntimeError(f"Artifact checksum mismatch: got {digest}")
        with tarfile.open(bundle) as archive:
            archive.extractall(directory, filter="data")
    finally:
        bundle.unlink(missing_ok=True)
