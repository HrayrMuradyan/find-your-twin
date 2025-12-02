import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def setup_google_credentials(project_root: Path) -> None:
    """
    Reads credentials from Environment Variables (injected by Docker/HF)
    and writes them to the expected JSON files in the 'credentials' directory.
    
    This replaces the legacy shell script logic for better cross-platform compatibility
    and error handling.
    """
    credentials_dir = project_root / "credentials"
    credentials_dir.mkdir(parents=True, exist_ok=True)
    
    client_secret_path = credentials_dir / "client_secret.json"
    token_path = credentials_dir / "token.json"

    # 1. Handle Client Secret
    secret_env = os.getenv("GOOGLE_CLIENT_SECRET")
    if secret_env:
        logger.info("Detected GOOGLE_CLIENT_SECRET env var. Writing to %s", client_secret_path)
        try:
            with open(client_secret_path, "w", encoding="utf-8") as f:
                f.write(secret_env)
        except IOError as e:
            logger.error("Failed to write client_secret.json: %s", e)
    
    # 2. Handle Token
    token_env = os.getenv("GOOGLE_TOKEN_JSON")
    if token_env:
        logger.info("Detected GOOGLE_TOKEN_JSON env var. Writing to %s", token_path)
        try:
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(token_env)
        except IOError as e:
             logger.error("Failed to write token.json: %s", e)
            
    if not client_secret_path.exists() and not token_path.exists():
        logger.warning(
            "No Google Credentials found in Environment Variables or local files. "
            "Google Drive integration may fail."
        )