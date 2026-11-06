import pytest

from healthrag.config import PROJECT_ROOT, Settings, load_settings


def test_yaml_values_load() -> None:
    settings = load_settings()
    assert settings.chunking.size > settings.chunking.overlap >= 0
    assert settings.synthea.population == 100
    assert len(settings.guidelines.sources) >= 3
    assert settings.paths.chunks_file.is_relative_to(PROJECT_ROOT)


def test_env_vars_override_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNKING__SIZE", "640")
    monkeypatch.setenv("REDACTION__SCORE_THRESHOLD", "0.55")
    settings = load_settings()
    assert settings.chunking.size == 640
    assert settings.redaction.score_threshold == 0.55


def test_salt_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATIENT_ID_SALT", "from-env")
    assert load_settings().require_salt() == "from-env"


def test_missing_salt_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATIENT_ID_SALT", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="PATIENT_ID_SALT"):
        settings.require_salt()


def test_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNKING__OVERLAP", "5000")
    with pytest.raises(ValueError):
        load_settings()
