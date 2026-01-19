import asyncio
import logging
import threading
import io
import time
import os
import psycopg2 
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent

# Imports from your src modules
from find_your_twin.logging_config import setup_logging
from find_your_twin.config import load_config
from find_your_twin.google_drive import get_drive_service, get_or_create_app_folder
from find_your_twin.image import read_image
from find_your_twin.file import read_json
from find_your_twin.validation import validate_model
from find_your_twin.model import read_model_config, load_model

# Setup logging, get the env variables
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

# Limit Concurrency to prevent crashes and Drive API Rate Limits
MAX_CONCURRENT_TASKS = 7
N_ATTEMPTS_TO_UPLOAD = 3

# Configuration Loading
CONFIG = load_config()
DATASET_ROOT = PROJECT_ROOT / CONFIG["data"]["path"]

class DirectUploader:
    def __init__(self):
        # Get the postgreSQL connection string
        self.dsn = os.getenv("DB_CONNECTION_STRING")
        if not self.dsn:
            raise ValueError("DB_CONNECTION_STRING environment variable is missing.")
            
        # Google Drive setup
        # Creates a temporary service here just to get the Folder ID once.
        # This ensures we fail fast if Auth is broken, before loading models.
        logger.info("Authenticating with Google Drive...")
        tmp_service = get_drive_service()
        if not tmp_service:
            raise ValueError("Could not authenticate with Google Drive.")
        self.drive_folder_id = get_or_create_app_folder(tmp_service)
        logger.info("Target Drive Folder ID: %s", self.drive_folder_id)

        # Each thread will lazily create its own service instance.
        self._thread_local = threading.local()
        
        logger.info("Loading ML Models locally...")
        
        # Face Detector
        detect_path = Path(PROJECT_ROOT) / CONFIG['models']['paths']['face_detect_model']
        validate_model(detect_path)
        face_detect_model_config = read_model_config(detect_path)
        self.face_detector = load_model(face_detect_model_config)

        # Embeddings
        embed_path = Path(PROJECT_ROOT) / CONFIG['models']['paths']['embeddings_model']
        validate_model(embed_path)
        embeddings_model_config = read_model_config(embed_path)
        self.embedder = load_model(embeddings_model_config)
        
        # Thread Locks
        self.ml_lock = threading.Lock()

    def _get_thread_safe_service(self):
        """
        Returns a unique Google Drive Service instance for the current thread.
        """
        if not hasattr(self._thread_local, 'service'):
            self._thread_local.service = get_drive_service()
        
        return self._thread_local.service

    def process_image_ml(self, img_path):
        """
        Reads image, detects face, computes embedding.
        """
        image_arr = read_image(img_path)
        
        with self.ml_lock:
            # Detect
            face = self.face_detector.detect(image_arr)
            if face is None:
                raise ValueError("No face detected")
            
            # Embed
            vector = self.embedder.compute_embeddings(face)
            
            # Prepare bytes for upload
            pil_image = Image.fromarray(image_arr)
            with io.BytesIO() as output:
                pil_image.save(output, format="JPEG")
                img_bytes = output.getvalue()
                
        return vector, img_bytes

    def upload_to_drive_with_retry(self, filename, file_bytes, uuid_str):
        from googleapiclient.http import MediaIoBaseUpload
        from googleapiclient.errors import HttpError
        import ssl

        # Get the unique service for THIS thread
        service = self._get_thread_safe_service()

        metadata = {
            'name': filename,
            'parents': [self.drive_folder_id],
            'appProperties': {'uuid': uuid_str}
        }

        # Retry logic for network flakiness
        for attempt in range(N_ATTEMPTS_TO_UPLOAD):
            try:
                # Create a fresh stream and media object for every attempt
                stream = io.BytesIO(file_bytes)
                media = MediaIoBaseUpload(stream, mimetype='image/jpeg', resumable=False)

                file = service.files().create(
                    body=metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                return file.get('id')

            except (ssl.SSLError, HttpError, BrokenPipeError) as e:
                logger.warning(f"Upload failed for {filename} (Attempt {attempt+1}): {e}")
                time.sleep(2) 
            except Exception as e:
                logger.exception(f"Fatal drive error: {e}")
                raise e
        
        raise ConnectionError("Failed to upload to Drive after 3 attempts")

    def insert_to_postgres(self, vector, drive_id, source, filename):
        """
        Connects to DB and executes insert.
        Opens/Closes connection per request to ensure thread safety 
        """
        conn = None
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor()
            
            # Serialize vector
            vector_json = str(vector.tolist())

            cursor.execute("""
                INSERT INTO face_data (source, original_filename, drive_file_id, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (source, filename, drive_id, vector_json))
            
            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id
            
        except Exception as e:
            if conn: conn.rollback()
            raise e
        finally:
            if conn: conn.close()

    def get_processed_files(self):
        """
        Returns a set of unique keys (source + filename) that are already in the DB.
        """
        conn = None
        processed = set()
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor()
            
            # Select only what we need to identify unique files
            cursor.execute("SELECT source, original_filename FROM face_data")
            rows = cursor.fetchall()
            
            for source, fname in rows:
                # Create a unique key string to check against later
                processed.add(f"{source}::{fname}")
                
            return processed
        finally:
            if conn:
                conn.close()

async def worker(sem, uploader, img_path, meta, index, total):
    async with sem:
        filename = Path(img_path).name
        try:
            # Run in thread executor to allow event loop to breathe
            vector, img_bytes = await asyncio.to_thread(uploader.process_image_ml, img_path)
            
            # Run in thread executor because Google Client is blocking
            import uuid
            new_uuid = str(uuid.uuid4())
            drive_file_name = f"{new_uuid}.jpg"
            
            drive_id = await asyncio.to_thread(
                uploader.upload_to_drive_with_retry, 
                drive_file_name, img_bytes, new_uuid
            )

            # Run in thread executor
            await asyncio.to_thread(
                uploader.insert_to_postgres,
                vector, drive_id, meta.get('name', 'Unknown'), meta.get('filename', 'Unknown')
            )
            
            logger.info(f"[{index}/{total}] Success: {filename}")
            return True

        except ValueError as ve:
            logger.warning(f"[{index}/{total}] Skipped {filename}: {ve}")
            return False
        except Exception as e:
            logger.exception(f"[{index}/{total}] Failed {filename}: {e}")
            return False

def scan_data_sources(data_root: Path):
    """Scans over data folders and finds all images to upload"""
    if not data_root.exists(): return [], []
    all_paths = []
    all_metas = []
    sub_sources = [d for d in data_root.iterdir() if d.is_dir()]
    
    for source_dir in sub_sources:
        images_dir = source_dir / "images"
        if not images_dir.exists(): continue
        
        try:
            # Try to read source_info.json if it exists
            info_path = source_dir / "source_info.json"
            if info_path.exists():
                source_info = read_json(info_path)
                # Filter for relevant keys
                base_meta = {k: v for k, v in source_info.items() if k in ["name"]}
            else:
                base_meta = {}
        except Exception:
            base_meta = {}
            
        if "name" not in base_meta:
            base_meta["name"] = source_dir.name

        valid_ext = {'.jpg', '.jpeg', '.png'}
        
        for p in images_dir.glob("*.*"):
            if p.suffix.lower() in valid_ext:
                all_paths.append(str(p))
                all_metas.append(base_meta.copy())
                
    return all_paths, all_metas

async def main():
    logger.info("Starting Direct DB Injection Pipeline")

    try:
        uploader = DirectUploader()
    except Exception as e:
        logger.critical(f"Failed to initialize uploader: {e}")
        return
    
    logger.info("Fetching existing records to avoid duplicates...")
    existing_files = uploader.get_processed_files()
    logger.info(f"Found {len(existing_files)} files already in database.")
    
    # Scan
    paths, metas = scan_data_sources(DATASET_ROOT)

    paths, metas = paths[:1000], metas[:1000]
    
    if not paths:
        logger.error("No images found.")
        return
    
    # Filter tasks
    new_tasks = []

    for path, meta in zip(paths, metas):
        filename = Path(path).name
        source_name = meta.get('name', 'Unknown')
        
        # Construct the unique key
        unique_key = f"{source_name}::{filename}"
        
        # Only add if NOT in existing_files
        if unique_key not in existing_files:
            meta['filename'] = str(filename)
            new_tasks.append((path, meta))
            
    logger.info(f"Found {len(paths)} total images. {len(existing_files)} skipped. {len(new_tasks)} to upload.")

    if not new_tasks:
        logger.info("All files already up to date.")
        return
    

    # Run Batch
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []
    total = len(new_tasks)
    
    logger.info(f"Processing {total} images with concurrency {MAX_CONCURRENT_TASKS}...")

    for i, (p, m) in enumerate(new_tasks, 1):
        tasks.append(worker(sem, uploader, p, m, i, total))
        
    await asyncio.gather(*tasks)
    logger.info("Upload Finished")

if __name__ == "__main__":
    asyncio.run(main())