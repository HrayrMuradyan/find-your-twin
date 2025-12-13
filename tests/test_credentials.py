import logging
from unittest.mock import patch
from find_your_twin.credentials import setup_google_credentials 

def test_setup_credentials_success(tmp_path, monkeypatch, caplog):
    """Verifies that environment variables are correctly read and written to the credentials files."""
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", '{"secret": "123"}')
    monkeypatch.setenv("GOOGLE_TOKEN_JSON", '{"token": "abc"}')

    with caplog.at_level(logging.INFO):
        setup_google_credentials(tmp_path)

    assert (tmp_path / "credentials" / "client_secret.json").read_text(encoding="utf-8") == '{"secret": "123"}'
    assert (tmp_path / "credentials" / "token.json").read_text(encoding="utf-8") == '{"token": "abc"}'
    assert "Detected GOOGLE_CLIENT_SECRET" in caplog.text
    assert "Detected GOOGLE_TOKEN_JSON" in caplog.text

def test_setup_credentials_no_env_warning(tmp_path, monkeypatch, caplog):
    """Ensures a warning is logged when neither environment variables nor credential files exist."""
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_TOKEN_JSON", raising=False)

    setup_google_credentials(tmp_path)

    assert "No Google Credentials found" in caplog.text
    assert not (tmp_path / "credentials" / "client_secret.json").exists()

def test_setup_credentials_existing_files_no_warning(tmp_path, monkeypatch, caplog):
    """Ensures no warning is logged if environment variables are missing but credential files already exist."""
    cred_dir = tmp_path / "credentials"
    cred_dir.mkdir()
    (cred_dir / "client_secret.json").touch()
    (cred_dir / "token.json").touch()
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_TOKEN_JSON", raising=False)
    caplog.clear()

    setup_google_credentials(tmp_path)

    assert "No Google Credentials found" not in caplog.text

def test_setup_credentials_io_error(tmp_path, monkeypatch, caplog):
    """Verifies that IOErrors during file writing are caught and logged as errors."""
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "data")
    monkeypatch.setenv("GOOGLE_TOKEN_JSON", "data")

    with patch("builtins.open", side_effect=IOError("Disk full")):
        setup_google_credentials(tmp_path)

    assert "Failed to write client_secret.json" in caplog.text
    assert "Failed to write token.json" in caplog.text