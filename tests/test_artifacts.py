import hashlib
import io
import tarfile

import pytest

from triage.artifacts import ensure_artifacts


def bundle(tmp_path, name="baseline.joblib"):
    path = tmp_path / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = 2
        archive.addfile(info, io.BytesIO(b"ok"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_verified_bundle(tmp_path):
    path, digest = bundle(tmp_path)
    target = tmp_path / "artifacts"
    ensure_artifacts(target, path.as_uri(), digest)
    assert (target / "baseline.joblib").read_bytes() == b"ok"


def test_rejects_wrong_checksum(tmp_path):
    path, _ = bundle(tmp_path)
    target = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="checksum"):
        ensure_artifacts(target, path.as_uri(), "0" * 64)
    assert not (target / "baseline.joblib").exists()


def test_requires_checksum_when_url_set(tmp_path):
    with pytest.raises(RuntimeError, match="ARTIFACTS_SHA256"):
        ensure_artifacts(tmp_path / "a", "file:///x", None)


def test_blocks_path_traversal(tmp_path):
    path, digest = bundle(tmp_path, "../evil.txt")
    target = tmp_path / "artifacts"
    with pytest.raises(tarfile.TarError):
        ensure_artifacts(target, path.as_uri(), digest)
    assert not (tmp_path / "evil.txt").exists()


def test_existing_artifacts_skip_download(tmp_path):
    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "x").write_text("y")
    ensure_artifacts(target, "file:///does-not-exist", None)
