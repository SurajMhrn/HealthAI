"""Project settings, loaded from config.yaml and .env.

Values come from three places, highest priority first:

1. environment variables (nested keys use a double underscore, e.g. ``CHUNKING__SIZE=800``)
2. the ``.env`` file in the project root
3. ``config.yaml`` in the project root

Secrets such as ``PATIENT_ID_SALT`` only ever come from the environment or ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathSettings(BaseModel):
    """Input and output locations. Relative paths are resolved against the project root."""

    raw_synthea: Path
    raw_synthea_sample: Path
    raw_guidelines: Path
    processed: Path
    chunks_file: Path
    redaction_report: Path
    vendor: Path

    @field_validator("*")
    @classmethod
    def _make_absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value


class ChunkingSettings(BaseModel):
    """Chunk size and overlap, both in characters."""

    size: int
    overlap: int

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingSettings:
        if self.size <= 0 or not 0 <= self.overlap < self.size:
            raise ValueError("chunking needs size > 0 and 0 <= overlap < size")
        return self


class PatientIdSettings(BaseModel):
    """How pseudonymous patient IDs are derived."""

    length: int


class SyntheaSettings(BaseModel):
    """Pinned Synthea release and the generation parameters."""

    version: str
    jar_url: str
    jar_sha256: str
    min_java_version: int
    seed: int
    clinician_seed: int
    reference_date: str
    population: int
    state: str


class SyntheaSampleSettings(BaseModel):
    """Synthea's pre-generated sample CSV archive."""

    url: str
    sha256: str
    description: str


class GuidelineSource(BaseModel):
    """One public guideline PDF to download."""

    name: str
    title: str
    short_title: str
    publisher: str
    license: str
    published: str
    url: str
    sha256: str | None = None


class DownloadSettings(BaseModel):
    """HTTP options for the data scripts."""

    user_agent: str
    timeout_seconds: int
    attempts: int


class GuidelineSettings(BaseModel):
    """The reviewed list of guideline PDFs."""

    sources: list[GuidelineSource]


class PdfSettings(BaseModel):
    """Heuristics used when turning PDF pages into sections."""

    heading_max_chars: int
    boilerplate_min_share: float
    boilerplate_min_pages: int


class RedactionSettings(BaseModel):
    """Presidio and spaCy options."""

    spacy_model: str
    language: str
    score_threshold: float
    batch_size: int
    report_samples: int


class Settings(BaseSettings):
    """All project settings. Use :func:`get_settings` to read them."""

    model_config = SettingsConfigDict(
        yaml_file=PROJECT_ROOT / "config.yaml",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: PathSettings
    chunking: ChunkingSettings
    patient_id: PatientIdSettings
    downloads: DownloadSettings
    synthea: SyntheaSettings
    synthea_sample: SyntheaSampleSettings
    patient_datasets: list[Literal["generated", "sample"]]
    guidelines: GuidelineSettings
    pdf: PdfSettings
    redaction: RedactionSettings

    patient_id_salt: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env vars first, then .env, then config.yaml."""
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))

    def patient_csv_dirs(self) -> dict[str, Path]:
        """CSV folder for each enabled patient dataset, keyed by dataset name."""
        folders = {"generated": self.paths.raw_synthea / "csv", "sample": self.paths.raw_synthea_sample / "csv"}
        return {name: folders[name] for name in self.patient_datasets}

    def require_salt(self) -> str:
        """Return the patient ID salt, or fail with a clear message if it is not set."""
        if self.patient_id_salt is None or not self.patient_id_salt.get_secret_value().strip():
            raise RuntimeError(
                "PATIENT_ID_SALT is not set. Copy .env.example to .env and set a long random value, "
                'e.g. python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return self.patient_id_salt.get_secret_value()


def load_settings() -> Settings:
    """Read settings fresh from config.yaml, .env and the environment."""
    return Settings()  # type: ignore[call-arg]  # fields are filled by the settings sources


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the settings, read once per process."""
    return load_settings()
