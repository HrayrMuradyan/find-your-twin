import sys
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from googleapiclient.http import MediaIoBaseUpload

# Add project root to the system path (for accessing modules in src folder)
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.google_drive import get_drive_service, get_or_create_app_folder
from src.config import load_config
from src.file import read_json

# Setup Logging
import logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Load the config
CONFIG = load_config()

# Define global variables
DATASET_ROOT = PROJECT_ROOT / CONFIG["data"]["path"]
TEMP_DATA_FILE = Path(CONFIG["data"]["temp_file"])

# Multiprocessing global variables
MAX_WORKERS = int(CONFIG["data"]["upload_max_workers"])
worker_service = None
worker_folder_id = None

def worker_init(folder_id):
    """
    Initializes the Google Drive service ONCE per worker process.
    """
    global worker_service, worker_folder_id
    # Suppress noisy logs in workers
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

    worker_service = get_drive_service()
    worker_folder_id = folder_id

def process_upload_task(row):
    """
    Uploads a single file using the process-local service.
    Each argument (row) is a metadata row of the temporary data file
    """
    global worker_service, worker_folder_id
    
    try:
        local_path = row['local_path']
        file_uuid = row['uuid']
        file_name = f"{file_uuid}.jpg"
        
        file_metadata = {
            'name': file_name,
            'parents': [worker_folder_id],
            'appProperties': {'uuid': file_uuid}
        }
        
        # Open file in binary mode
        with open(local_path, 'rb') as f:
            media = MediaIoBaseUpload(f, mimetype='image/jpeg', resumable=False)
            
            file = worker_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
        return (row.name, file.get('id'))
    except Exception as e:
        return (row.name, None)

def scan_data_sources(data_root: Path,
                      metadata_to_include: list = ["name", "keywords"]):
    """
    Iterates through all sub-folders in the data_root.
    If a sub-folder contains 'images/' and 'source_info.json', it is processed.
    """
    if not data_root.exists():
        logger.error("Data root not found: %s", data_root)
        return [], []

    all_paths = []
    all_metas = []
    
    # Get all subdirectories in the data folder
    sub_sources = [d for d in data_root.iterdir() if d.is_dir()]
    
    logger.info(
        "Scanning %s potential sources in %s...",
        len(sub_sources),
        data_root.relative_to(PROJECT_ROOT)
    )

    for source_dir in sub_sources:
        images_dir = source_dir / "images"
        metadata_file = source_dir / "source_info.json"

        # Check if this folder is a valid data source
        # To be valid it should contain 'images' and 'source_info.json' folder
        if not (images_dir.exists() and metadata_file.exists()):
            logger.warning(
                "Skipping '%s': Missing 'images' folder or 'source_info.json'",
                source_dir.name
            )
            continue

        logger.info("Processing source: %s", source_dir.name)

        try:
            raw_data = read_json(metadata_file)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON in %s. Skipping.", source_dir.name)
            continue

        # Extract metadata that is included in the metadata_to_include list
        base_metadata = {
            k: str(v) for k, v in raw_data.items() if k in metadata_to_include
        }

        # Gather ONLY Images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        source_images = [
            p for p in images_dir.glob("*.*") 
            if p.suffix.lower() in valid_extensions
        ]

        for img_path in source_images:
            # Create a specific metadata dict for this image
            # Copy base metadata and add specific filename
            base_metadata["original_filename"] = img_path.name
            
            all_paths.append(str(img_path))
            all_metas.append(base_metadata)
            
        logger.info("> Found %s images in %s", len(source_images), source_dir.name)

    logger.info("Total Scan Complete: %s images ready for processing.", len(all_paths))
    return all_paths, all_metas

def run_pipeline():
    """
    Main function for running the full data population pipeline
    """
    config = load_config()
    
    # ---------------------------------------------------------
    # UPLOAD PHASE 1: PROCESS LOCAL IMAGES
    # ---------------------------------------------------------
    phase_1_needed = True
    if TEMP_DATA_FILE.exists():
        try:
            df = pd.read_parquet(TEMP_DATA_FILE)
            if not df.empty:
                logger.info("Found existing temp data. Skipping PHASE 1.")
                phase_1_needed = False
        except:
            pass 

    if phase_1_needed:
        from src.embeddings_database import AutoFaissIndex
        
        logger.info("Starting PHASE 1: Computation")
        
        # Get all image_paths and their metadata from all sources from the dataset root
        image_paths, metadata_list = scan_data_sources(DATASET_ROOT)
        
        image_paths, metadata_list = image_paths[:50], metadata_list[:50]
        
        if not image_paths:
            logger.error("No images found in any valid source folders!")
            return

        faiss_index = AutoFaissIndex(
            index_path=PROJECT_ROOT / config['faiss']['paths']['index_path'],
            face_detect_model=config['models']['paths']['face_detect_model'],
            embeddings_model=config['models']['paths']['embeddings_model'],
        )
        
        # Save temporary file which includes both the metadata and embeddings
        # Temporary file is a Pandas Dataframe saved as parquet
        faiss_index.process_local_to_parquet(image_paths, metadata_list, TEMP_DATA_FILE)
        
        del faiss_index
        import gc
        gc.collect()

    # ---------------------------------------------------------
    # PHASE 2: MULTIPROCESS UPLOAD 
    # ---------------------------------------------------------
    logger.info("Starting PHASE 2: Uploads")
    if not TEMP_DATA_FILE.exists():
        logging.error("The temporary data file '%s' doesn't exist. Can't upload...", TEMP_DATA_FILE)
        return

    df = pd.read_parquet(TEMP_DATA_FILE)

    # Initially, gdrive_id shouldn't be present
    # This means that none of the files are uploaded
    if 'gdrive_id' not in df.columns:
        df['gdrive_id'] = None

    # Get only those rows that are not uploaded (don't have gdrive id assigned)
    pending_mask = df['gdrive_id'].isnull()
    pending_rows = df[pending_mask]
    
    if pending_rows.empty:
        logger.info("No pending uploads found. Proceeding to Indexing.")
    else:
        logger.info(
            "Uploading %s images with %s processes...",
            len(pending_rows),
            MAX_WORKERS
        )
        
        # Initialize drive service and get or create a drive folder
        drive_service = get_drive_service()
        if not drive_service:
            return
        
        folder_id = get_or_create_app_folder(drive_service)
        
        # Initialize MAX_WORKERS different workers
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS, 
            initializer=worker_init, 
            initargs=(folder_id,)
        ) as executor:
            
            future_map = {
                executor.submit(process_upload_task, row): index 
                for index, row in pending_rows.iterrows()
            }
            
            counter = 0
            
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(future_map), total=len(pending_rows)
                ):
                    idx, gdrive_id = future.result()
                    
                    if gdrive_id:
                        df.at[idx, 'gdrive_id'] = gdrive_id
                    
                    counter += 1
                    
                    # Saving checkpoint
                    if counter % 50 == 0: 
                        df.to_parquet(TEMP_DATA_FILE, engine='fastparquet', compression='snappy')

            finally:
                logger.info("Interruption detected! Saving progress before exiting...")
                df.to_parquet(TEMP_DATA_FILE, engine='fastparquet', compression='snappy')
                logger.info("Progress saved")

        # Final Save
        # May be redundant but safe
        df.to_parquet(TEMP_DATA_FILE, engine='fastparquet', compression='snappy')
        logger.info("Phase 2 Complete.")

    # ---------------------------------------------------------
    # PHASE 3: INDEXING
    # ---------------------------------------------------------
    logger.info("Starting PHASE 3: Saving FAISS Index")
    
    from src.embeddings_database import AutoFaissIndex
    
    faiss_index = AutoFaissIndex(
        index_path=PROJECT_ROOT / config['faiss']['paths']['index_path'],
        face_detect_model=config['models']['paths']['face_detect_model'],
        embeddings_model=config['models']['paths']['embeddings_model'],
    )
    
    # This would create the index from the temp parquet file
    faiss_index.build_index_from_parquet(TEMP_DATA_FILE)
    
    # Remove the temp data file if it exists
    if TEMP_DATA_FILE.exists():
        TEMP_DATA_FILE.unlink()
        logger.info("Temporary file %s removed", TEMP_DATA_FILE)
        
    logger.info("Done. Index built")

if __name__ == "__main__":
    run_pipeline()