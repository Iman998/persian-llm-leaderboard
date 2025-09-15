from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from leaderboard_lib.utils import _load_module, load_api_key


def test_load_module_success():
    path = Path(__file__).resolve().parent.parent / "leaderboard_lib" / "utils.py"
    mod = _load_module(str(path))
    assert mod.__name__ == "leaderboard_lib.utils"


def test_load_module_failure():
    with pytest.raises(ImportError):
        _load_module("nonexistent_module.py")


def test_load_api_key_precedence(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    secrets_file = repo_root / "secrets.toml"
    secrets_file.write_text(
        "[openai]\napi_key = 'openai_section'\n"
        "[model_keys]\nmodelA = 'file_model_key'\n"
    )
    try:
        monkeypatch.setenv("OPENAI_API_KEY_MODELA", "env_model_key")
        monkeypatch.setenv("OPENAI_API_KEY", "env_global_key")
        assert load_api_key("modelA") == "env_model_key"

        monkeypatch.delenv("OPENAI_API_KEY_MODELA", raising=False)
        assert load_api_key("modelA") == "env_global_key"

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert load_api_key("modelA") == "file_model_key"

        assert load_api_key("unknown") == "openai_section"

        secrets_file.write_text(
            "api_key = 'top_level'\n"
            "[model_keys]\nmodelA = 'file_model_key'\n"
        )
        assert load_api_key("unknown") == "top_level"
    finally:
        if secrets_file.exists():
            secrets_file.unlink()
