import os
import psycopg2
from dotenv import load_dotenv
import logging

# Setup Logging
from find_your_twin.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from find_your_twin.google_drive import get_drive_service, get_or_create_app_folder

# Load Environment Variables
load_dotenv()

DB_DSN = os.getenv("DB_CONNECTION_STRING")

if not DB_DSN:
    raise ValueError("Error: DB_CONNECTION_STRING not found in .env file")

def get_db_file_ids():
    """Fetches all drive_file_ids stored in Postgres."""
    conn = None
    try:
        conn = psycopg2.connect(DB_DSN)
        cursor = conn.cursor()
        cursor.execute("SELECT drive_file_id FROM face_data")
        rows = cursor.fetchall()
        # Return as a set for O(1) lookups
        return {row[0] for row in rows}
    except Exception as e:
        logger.exception(f"Database Error: {e}")
        return set()
    finally:
        if conn:
            conn.close()

def get_drive_files(service, folder_id):
    """
    Fetches file IDs from Drive using larger pages and minimal fields.
    """
    files_set = set()
    page_token = None
    
    page_size = 1000 
    
    logger.info("Scanning Google Drive folder (this may take a moment for 200k files)...")
    
    while True:
        try:

            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces='drive',
                pageSize=page_size,
                fields="nextPageToken, files(id)",
                pageToken=page_token
            ).execute()

            files = response.get('files', [])
            
            # Batch update the set for slight speedup
            files_set.update(f['id'] for f in files)
            
            page_token = response.get('nextPageToken')
            
            # Log progress every ~10k files
            if len(files_set) % 10000 == 0:
                logger.info(f"   ...scanned {len(files_set)} files so far")
                
            if not page_token:
                break
                
        except Exception as e:
            logger.exception(f"Drive Scanning Error: {e}")
            break
            
    return files_set

def resync_database():
    logger.info("--- STARTING RESYNC ---")
    
    # Connect to Services
    logger.info("Connecting to Google Drive...")
    drive_service = get_drive_service()
    if not drive_service:
        logger.error("Failed to connect to Drive.")
        return

    folder_id = get_or_create_app_folder(drive_service)
    if not folder_id:
        logger.error("Failed to find Google Drive Folder.")
        return

    # Fetch Data
    logger.info("Fetching Database Records...")
    db_ids = get_db_file_ids()
    logger.info(f"   Found {len(db_ids)} records in Postgres.")

    logger.info("Fetching Drive Files...")
    drive_ids = get_drive_files(drive_service, folder_id)
    logger.info(f"   Found {len(drive_ids)} files in Drive.")

    # Calculate Differences
    # In Drive, but not in DB
    orphans = drive_ids - db_ids
    
    # In DB, but not in Drive
    broken_links = db_ids - drive_ids

    if not orphans and not broken_links:
        logger.info("Systems are in sync. No actions needed.")
        return

    # Report and Confirm
    logger.info("-" * 40)
    logger.info("SYNC REPORT:")
    if orphans:
        logger.info(f"ORPHANS (In Drive, No DB Record): {len(orphans)}")
        logger.info("   -> Action: Will be DELETED from Google Drive.")
    
    if broken_links:
        logger.info(f"BROKEN LINKS (In DB, No Drive File): {len(broken_links)}")
        logger.info("   -> Action: Will be DELETED from PostgreSQL.")
    logger.info("-" * 40)

    verification = input("Type 'SYNC' to execute these changes: ")
    if verification != "SYNC":
        logger.info("Operation aborted.")
        return

    # Execute Cleanup
    
    # Clean Drive (Orphans)
    if orphans:
        logger.info("Cleaning up Drive orphans...")
        count = 0
        for file_id in orphans:
            try:
                drive_service.files().delete(fileId=file_id).execute()
                count += 1
                if count % 10 == 0: logger.info(f"Deleted {count}/{len(orphans)}...")
            except Exception as e:
                logger.error(f"Failed to delete Drive file {file_id}: {e}")
        logger.info(f"Drive cleanup complete. Removed {count} files.")

    # Clean DB (Broken Links)
    if broken_links:
        logger.info("Cleaning up Database broken links...")
        conn = None
        try:
            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Convert set to tuple for SQL IN clause
            broken_tuple = tuple(broken_links)
            
            # Execute Delete
            cursor.execute("DELETE FROM face_data WHERE drive_file_id IN %s", (broken_tuple,))
            deleted_count = cursor.rowcount
            
            # Also clean up FAISS index metadata if necessary
            
            logger.info(f"Database cleanup complete. Removed {deleted_count} rows.")
            
        except Exception as e:
            logger.exception(f"Database Deletion Error: {e}")
        finally:
            if conn: conn.close()

    logger.info("Resync Complete.")

if __name__ == "__main__":
    resync_database()