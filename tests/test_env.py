"""Tests for the minimal .env loader."""

import os

from env import load_dotenv


def _write_dotenv(tmp_path, content: str):
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_file_is_silent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() is False


def test_basic_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_dotenv(tmp_path, "OPENAI_API_KEY=sk-test\nOPENAI_MODEL=deepseek-chat\n")
    assert load_dotenv() is True
    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert os.environ["OPENAI_MODEL"] == "deepseek-chat"


def test_comments_blank_and_quotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_dotenv(
        tmp_path,
        "# comment line\n\nKEY_A=plain\nKEY_B=\"double quoted\"\nKEY_C='single quoted'\n",
    )
    load_dotenv()
    assert os.environ["KEY_A"] == "plain"
    assert os.environ["KEY_B"] == "double quoted"
    assert os.environ["KEY_C"] == "single quoted"


def test_existing_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    _write_dotenv(tmp_path, "OPENAI_API_KEY=from-dotenv\n")
    load_dotenv()
    assert os.environ["OPENAI_API_KEY"] == "from-env"


def test_malformed_lines_tolerated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_dotenv(tmp_path, "not a valid line\nOK=1\n")
    assert load_dotenv() is True
    assert os.environ["OK"] == "1"
