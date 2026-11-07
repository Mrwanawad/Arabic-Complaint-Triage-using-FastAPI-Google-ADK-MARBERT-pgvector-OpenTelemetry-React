"""Runtime configuration, loaded from environment variables or a local .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider SDKs and OpenTelemetry read os.environ directly, so .env must reach it as well as
# Settings. Variables already set in the real environment take priority.
load_dotenv(".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None
    database_replica_url: str | None = None
    db_pool_size: int = 10

    artifacts_dir: Path = Path("artifacts")
    artifacts_url: str | None = None
    artifacts_sha256: str | None = None
    default_model: str = "encoder"

    encoder_threshold: float = 0.80
    llm_threshold: float = 0.70
    few_shot_examples: int = 5

    cache_threshold: float = 0.97
    cache_ttl_days: int = 7

    llm_enabled: bool = True
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "anthropic/claude-haiku-5-5"
    deepseek_model: str = "deepseek/deepseek-chat"
    gemini_weight: float = 0.5
    anthropic_weight: float = 0.5
    provider_rpm: int = 60
    llm_timeout_seconds: float = 20.0
    breaker_failures: int = 3
    breaker_cooldown_seconds: float = 60.0

    transcriber_enabled: bool = True
    whisper_model: str = "base"
    max_audio_bytes: int = 20_000_000

    # Review and dashboard endpoints refuse every request while this is unset.
    admin_api_key: str | None = None

    otel_enabled: bool = True

    @property
    def encoder_dir(self) -> Path:
        return self.artifacts_dir / "encoder"

    @property
    def baseline_path(self) -> Path:
        return self.artifacts_dir / "baseline.joblib"

    @property
    def urgency_path(self) -> Path:
        return self.artifacts_dir / "urgency_lgbm.txt"

    @property
    def embedder_dir(self) -> Path:
        return self.artifacts_dir / "embedder"
