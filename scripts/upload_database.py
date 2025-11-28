import sys
import json
from pathlib import Path
import logging

# Add project root to sys.path
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.google_drive import get_drive_service, get_or_create_app_folder
from src.embeddings_database import AutoFaissIndex
from src.config import load_config
from src.file import read_json

# Load Config
CONFIG = load_config()
DATASET_ROOT = PROJECT_ROOT / CONFIG["data"]["path"]

def scan_data_sources(data_root: Path, metadata_to_include: list = ["name", "keywords"]):
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
            # We ensure 'name' exists as it's required by the DB insert logic
            meta = base_metadata.copy()
            
            meta["original_filename"] = img_path.name
            
            all_paths.append(str(img_path))
            all_metas.append(meta)
            
        logger.info("> Found %s images in %s", len(source_images), source_dir.name)

    logger.info("Total Scan Complete: %s images ready for processing.", len(all_paths))
    return all_paths, all_metas

def main():
    logger.info("--- Starting Database Population Pipeline ---")
    
    # Scan all subfolders for images
    image_paths, metadata_list = scan_data_sources(DATASET_ROOT)
    
    if not image_paths:
        logger.error("No images found. Exiting.")
        return

    image_paths = image_paths[:50]
    metadata_list = metadata_list[:50]

    # Initialize services
    try:
        logger.info("Initializing Google Drive...")
        drive_service = get_drive_service()
        if not drive_service:
            logger.error("Failed to init Drive service.")
            return
        
        folder_id = get_or_create_app_folder(drive_service)
        
        logger.info("Initializing Database Connection & Models...")
        
        # Initialize the FAISS index
        db = AutoFaissIndex(
            face_detect_model=CONFIG['models']['paths']['face_detect_model'],
            embeddings_model=CONFIG['models']['paths']['embeddings_model'],
            drive_service=drive_service,
            drive_folder_id=folder_id
        )
        
        # Run the uploading
        # This method handles: Read -> Detect -> Embed -> Upload Drive -> Insert DB -> Update RAM
        logger.info("Starting batch upload and indexing...")

        db.process_and_upload_batch(image_paths, metadata_list)
        
        logger.info("--- Pipeline Complete ---")
        logger.info("Total vectors in index: %s", db.index.ntotal)

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)

if __name__ == "__main__":
    main()