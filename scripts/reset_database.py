import sys
import os
import shutil
from pathlib import Path
from googleapiclient.errors import HttpError

# Add project root to sys.path
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.google_drive import get_drive_service, get_or_create_app_folder

# Setup Logging
import logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def delete_remote_folder(service, folder_id):
    """
    Deletes the entire App Folder on Google Drive.
    This is faster than deleting files one by one.
    """
    try:
        logger.info("Deleting Google Drive folder (ID: %s)...", folder_id)
        service.files().delete(fileId=folder_id).execute()
        logger.info("Successfully deleted remote folder.")
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning("Folder already deleted or not found.")
        else:
            logger.error("Failed to delete folder: %s", e)

def delete_local_index():
    """Deletes the local FAISS index directory and temp files."""
    config = load_config()
    
    # 1. Delete Index Folder
    index_path = PROJECT_ROOT / config['faiss']['paths']['index_path']
    if index_path.exists():
        try:
            shutil.rmtree(index_path)
            logger.info("Deleted local index folder: %s", index_path)
        except Exception as e:
            logger.error("Failed to delete %s: %s", index_path, e)
    else:
        logger.info("No local index folder found.")

    # 2. Delete Temp Parquet File
    temp_file = PROJECT_ROOT / "batch_processing_temp.parquet"
    if temp_file.exists():
        try:
            os.remove(temp_file)
            logger.info("Deleted temp file: %s", temp_file)
        except Exception as e:
            logger.error("Failed to delete %s: %s", temp_file, e)

def run_reset():
    logger.warning("⚠️ DANGER ZONE ⚠️")
    logger.warning("This will permanently delete:")
    logger.warning("1. The ENTIRE 'face-similarity' folder on Google Drive.")
    logger.warning("2. Your local FAISS index and metadata.")
    logger.warning("3. Any temporary batch processing files.")
    
    confirmation = input("\nType 'DELETE' to confirm: ")
    
    if confirmation.strip() != "DELETE":
        logger.info("Aborted")
        return

    logger.info("Starting system reset")
    
    # Remove google drive folder
    service = get_drive_service()
    if service:
        folder_id = get_or_create_app_folder(service)
        if folder_id:
            delete_remote_folder(service, folder_id)
    else:
        logger.error("Could not connect to Drive. Skipping remote deletion.")

    # Delete the local index (FAISS)
    delete_local_index()
    
    logger.info("System reset complete")
    logger.info("Next time you run the app, an empty Google Drive folder will be created automatically.")

if __name__ == "__main__":
    run_reset()