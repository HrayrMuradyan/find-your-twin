import numpy as np
import logging
from pathlib import Path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.google_drive import get_file_by_uuid, delete_file_by_id
from src.utils import blur_str

# Setup logger
logger = logging.getLogger(__name__)

def remove_by_uuid(faiss_index, drive_service, uuid):
    """
    Remove an image and its corresponding embedding from the system using a UUID.

    This function performs a coordinated deletion across:
    - Google Drive storage (using the Drive API)
    - The FAISS index (vector embeddings)
    - The embeddings metadata DataFrame

    It ensures consistency through rollback steps if any part of the deletion fails.

    Args:
        faiss_index: An object that holds:
            - `index`: the FAISS index instance
            - `emb_metadata`: a pandas DataFrame containing embedding metadata
            - `_save()`: a method to persist index/metadata changes
        drive_service: Authenticated Google Drive API service instance.
        uuid (str): UUID of the image to delete (used to look up file on Drive).

    Returns:
        bool: True if deletion succeeds, False if the UUID is not found in Drive.
              If deletion fails in one of the later steps, rollbacks are attempted
              and the function still returns True after finishing rollback.
    """
    logging.info("Trying to remove the file with UUID %s from drive", blur_str(uuid))
    file = get_file_by_uuid(drive_service, uuid)
    if not file:
        logging.info("Couldn't find the file with UUID %s in Drive", blur_str(uuid))
        return False
    
    upload_id = file['id']

    # Given the upload id of the image, fetch the embedding metadata from the dataframe
    emb_metadata_row = faiss_index.emb_metadata[faiss_index.emb_metadata.gdrive_id == upload_id]
    if emb_metadata_row.empty:
        logging.error(
            "Couldn't find Upload ID=%s in the embedding metadata dataframe. Something went wrong...",
            upload_id
        )

    image_id = emb_metadata_row.squeeze()['id']
    image_index = emb_metadata_row.index[0]

    # Remove from the embedding metadata
    try:
        faiss_index.emb_metadata.drop(image_index, inplace=True)
        logger.info("Metadata deleted from the Embeddings Metadata DataFrame")
    except Exception as e:
        logger.exception("Couldn't delete the metadata from the FAISS index metadata file. Reason: %s", e)

    # Remove from faiss
    # Fetch the embedding vector first to rollback if there is an error
    emb_vector = faiss_index.index.index.reconstruct(int(image_id)).reshape(1, -1)
    try:
        faiss_index.index.remove_ids(np.array([image_id], dtype='int64'))
        logger.info("Embeddings removed from the FAISS Index")
    except Exception as e:
        logger.exception("Can't remove the image embeddings from the FAISS index. Reason: %s", e)
        logger.exception("Rolling back the metadata removal...")
        faiss_index.emb_metadata.loc[image_index] = emb_metadata_row.iloc[0]

    # Remove from the drive
    # Rollback if there is an error
    try:
        _ = delete_file_by_id(drive_service, upload_id)
        logger.info("Successfully removed the image from the drive")
    except Exception as e:
        logger.exception("Can't delete the image from the drive. Reason: %s", e)
        logger.exception("Rolling back the metadata and embedding removal...")
        faiss_index.emb_metadata.loc[image_index] = emb_metadata_row.iloc[0]
        faiss_index.index.add_with_ids(emb_vector, np.array([image_id], dtype=np.int64))

    # Save the changes
    faiss_index._save() 
    logger.info("\033[1mDeleting Done.\033[0m")
    return True