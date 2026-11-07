import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PROBE = "import os, triage.settings; print(os.environ.get('GEMINI_API_KEY'))"


def run_probe(cwd: Path, **env: str) -> str:
    """A fresh process is needed because .env loads once, when the module is imported."""
    clean = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=cwd,
        env={**clean, "PYTHONPATH": str(SRC), **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_dotenv_reaches_os_environ_for_provider_sdks(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    assert run_probe(tmp_path) == "from-dotenv"


def test_real_environment_beats_dotenv(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    assert run_probe(tmp_path, GEMINI_API_KEY="from-shell") == "from-shell"
