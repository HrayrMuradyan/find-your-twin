import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

# Add the root to the path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
import logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.logging_config import setup_logging
from src.google_drive import get_drive_service, get_or_create_app_folder


# Load Environment Variables
load_dotenv()

DB_DSN = os.getenv("DB_CONNECTION_STRING")

if not DB_DSN:
    raise ValueError("Error: DB_CONNECTION_STRING not found in .env file")

def clear_database():
    logging.info("⚠️  DANGER ZONE ⚠️")
    logging.info("This will permanently DELETE:")
    logging.info("  1. The following tables from your database:")
    logging.info("     - face_data (All vectors and metadata)")
    logging.info("     - index_metadata (FAISS settings)")
    logging.info("  2. The entire Google Drive Folder (All images)")
    
    # Verification Step
    verification = input("\nType 'DELETE' to confirm: ")
    
    if verification != "DELETE":
        logging.error("Verification failed. Operation aborted")
        return

    # ---------------------------------------------------------
    # Clear Metadata, FAISS Database
    # ---------------------------------------------------------
    conn = None
    cursor = None
    try:
        logging.info("Connecting to Supabase...")
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop the tables
        tables_to_drop = ["face_data", "index_config"]
        
        for table in tables_to_drop:
            logging.info(f"Dropping table '{table}'...")
            cursor.execute(f"DROP TABLE IF EXISTS {table};")

        logging.info("Database tables dropped successfully")

    except Exception as e:
        logging.exception(f"Database Error: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    # ---------------------------------------------------------
    # Clear Google Drive
    # ---------------------------------------------------------
    try:
        logging.info("Connecting to Google Drive...")
        service = get_drive_service()
        
        if service:
            # Get the folder ID. If it doesn't exist, it creates an empty one (this ensures clean state)
            # Removes the folder ID
            folder_id = get_or_create_app_folder(service)
            
            if folder_id:
                logging.info("Deleting Drive Folder (ID: %s)...", folder_id)
                # The delete method works for files and folders alike
                service.files().delete(fileId=folder_id).execute()
                logging.info("Google Drive folder deleted successfully")
            else:
                logging.warning("Could not resolve App Folder ID.")
        else:
            logging.error("Failed to initialize Drive service. Skipping Drive deletion.")

    except Exception as e:
        logging.exception("Google Drive Error: %s", e)

if __name__ == "__main__":
    clear_database()