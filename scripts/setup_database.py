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

from src.config import load_config
from src.model import read_model_config

# Load Environment Variables
load_dotenv()

# Get the connection string from .env
DB_DSN = os.getenv("DB_CONNECTION_STRING")

# Load the config
config = load_config()
face_detection_model = config['models']['paths']['face_detect_model']
face_detection_config = read_model_config(PROJECT_ROOT / face_detection_model)

embeddings_model = config['models']['paths']['embeddings_model']
embeddings_config = read_model_config(PROJECT_ROOT / embeddings_model)


# Define the metadata
# This will be saved in the index config table
db_metadata = {
    "index_type": config['faiss']['index_type'],
    "dim": embeddings_config['parameters']['dim'],
    "face_detect_model": face_detection_model,
    "embeddings_model": embeddings_model,
    "vector_count": 0
}

if not DB_DSN:
    raise ValueError("Error: DB_CONNECTION_STRING not found in .env file.")

# Define SQL Commands
CREATE_TABLES = [
    # Enable the Vector Extension
    "CREATE EXTENSION IF NOT EXISTS vector;",

    # Create the Index Metadata Table
    """
    CREATE TABLE IF NOT EXISTS index_metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """,

    # Create the image metadata table
    # This stores the raw vector and metadata.
    """
    CREATE TABLE IF NOT EXISTS face_data (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        drive_file_id TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        embedding vector(512)
    );
    """
]

POPULATE_METADATA = """
    INSERT INTO index_metadata (key, value)
    VALUES {}
    ON CONFLICT (key) DO NOTHING;
""".format(
    ",".join(["(%s, %s)"] * len(db_metadata))
)

def setup_database():
    conn = None
    cursor = None
    try:
        logging.info("Connecting to Supabase...")
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True 
        cursor = conn.cursor()
        
        logging.info("Creating the tables...")
        for command in CREATE_TABLES:
            try:
                # Print a snippet of the command being run
                cmd_preview = command.strip().split('\n')[0]
                logging.info("-> Executing: %s...", cmd_preview)
                
                cursor.execute(command)
            except Exception as e:
                logging.exception("Couldn't run the command. Error: %s", e)

        # Populating the faiss metadata table
        params = [item for kv in db_metadata.items() for item in kv]
        cursor.execute(POPULATE_METADATA, params)

        logging.info("Database setup complete!")
        logging.info("- Extension 'vector' enabled.")
        logging.info("- Table 'index_metadata' created/verified.")
        logging.info("- Table 'face_data' created/verified.")

    except Exception as e:
        logging.exception("An error occurred. Check if your Connection String is correct (port 5432 or 6543).")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_database()