# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

import uvicorn
import numpy as np
from PIL import Image, ImageOps
import io
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
from pathlib import Path


# Add project root to sys.path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules 
from src.embeddings_database import AutoFaissIndex
from src.google_drive import (
    get_drive_service,
    get_or_create_app_folder,
    upload_bytes_to_folder,
    get_image_bytes_by_id
)
from src.config import load_config
from src.delete import remove_by_uuid
from src.utils import blur_str

# Setup logger
import logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# DECLARING IMPORTANT VARIABLES
# ---------------------------------------------------

# Read the config
CONFIG = load_config()

# App Setup 
app = FastAPI(title=CONFIG['app']['title'])

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Global Variables
faiss_index: AutoFaissIndex = None
drive_service = None
drive_folder_id = None

# Define the Base URL
BASE_URL = CONFIG['app']['base_url']

# Other important global variables
FACE_DETECT_MODEL = PROJECT_ROOT / CONFIG['models']['paths']['face_detect_model']
EMBEDDINGS_MODEL = PROJECT_ROOT / CONFIG['models']['paths']['embeddings_model']
FAISS_INDEX_PATH = PROJECT_ROOT / CONFIG['faiss']['paths']['index_path']
IMAGE_MAX_SIZE = int(CONFIG['image']['max_size'])

# Search variables
RETRIEVAL_SIMILARITY_THRESHOLD = int(CONFIG['search']['retrieval_similarity_threshold'])
MIN_OTHER_IMAGES = int(CONFIG['search']['min_other_images'])
K_TO_SEARCH = int(CONFIG['search']['k_to_search']) 
SAVE_SIMILARITY_THRESHOLD = int(CONFIG['search']['save_similarity_threshold'])

# On Startup, Load the models
@app.on_event("startup")
def startup_event():
    global faiss_index, drive_service, drive_folder_id
    
    logging.info("--- Server starting up... ---")
    try:
        logging.info("Initializing Google Drive service...")
        drive_service = get_drive_service()
        if drive_service:
            logging.info("Google Drive service initialized successfully.")
            drive_folder_id = get_or_create_app_folder(drive_service)

            if drive_folder_id:
                logging.info("Google Drive folder ID set: %s", drive_folder_id)
            else:
                logging.error("Could not get or create Google Drive folder.")
        else:
            logging.error("Failed to initialize Google Drive service. Check ENV vars.")

        logging.info(
            "Loading FAISS index from %s...",
            FAISS_INDEX_PATH.relative_to(PROJECT_ROOT)
        )

        faiss_index = AutoFaissIndex(
            index_path=FAISS_INDEX_PATH,
            face_detect_model=FACE_DETECT_MODEL,
            embeddings_model=EMBEDDINGS_MODEL,
            drive_service=drive_service,
            drive_folder_id=drive_folder_id
        )

        logging.info("FAISS Index loaded successfully.")

    except Exception as e:
        logging.error("Something went wrong during startup: %s", e)
    
    logging.info("--- Server startup complete. ---")


# Image search endpoint
@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    consent: bool = Form(False) 
):
    logging.info("Starting an image search...")
    global faiss_index, drive_service, drive_folder_id

    if not faiss_index:
        raise HTTPException(
            status_code=503,
            detail="Server is still initializing models. Please try again in a moment."
        )

    # Get the image from the upload
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # This physically rotates the image if the metadata says so
    image = ImageOps.exif_transpose(image)
    
    # Resize the image to meet IMAGE_MAX_SIZE limit
    if image.height > IMAGE_MAX_SIZE or image.width > IMAGE_MAX_SIZE:
        if image.height > image.width:
            new_height = IMAGE_MAX_SIZE
            new_width = int(image.width * (IMAGE_MAX_SIZE / image.height))
        else:
            new_width = IMAGE_MAX_SIZE
            new_height = int(image.height * (IMAGE_MAX_SIZE / image.width))

        logging.info("Resizing image from %sx%s to %sx%s", image.width, image.height, new_width, new_height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logging.info("Resizing successful")
    
    img_array = np.array(image)
    logging.info("Image shape for processing: %s", img_array.shape)
    
    # Search an image using the FAISS Index
    try:
        scores, ids, metadata, face_img = faiss_index.search_image(
            img_array, k=K_TO_SEARCH, return_metadata=True, return_face=True
        )
    except ValueError as e:
        logging.warning(f"Face detection failed: {e}")
        raise HTTPException(
            status_code=400, 
            detail="No face detected. Please upload a photo with a clear, visible face."
        )
    except Exception as e:
        logging.error(f"Search processing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")
    
    if scores is None or len(scores[0]) == 0:
        return {"results": []} 
    
    scores, ids, metadata_df = scores[0], ids[0], metadata[0]
    
    all_results = []
    highest_score = 0

    # For all similar IDS
    for i in range(len(ids)):
        gdrive_file_id = metadata_df.loc[ids[i], 'gdrive_id']

        score_i = int(scores[i] * 100) 

        if i == 0:
            highest_score = score_i 

        # Construct the URL pointing to the proxy endpoint
        image_url = f"/gdrive-image/{gdrive_file_id}"
        
        all_results.append({"url": image_url, "similarity": score_i})

    # If no results found, return an empty list
    if not all_results:
        return {"results": []} 
    
    # Get the top result
    top_result = all_results[0]
    other_results_all = all_results[1:]
    results_above_threshold = [
        r for r in other_results_all if r['similarity'] >= RETRIEVAL_SIMILARITY_THRESHOLD
    ]

    min_results = other_results_all[:MIN_OTHER_IMAGES]
    
    final_other_results = results_above_threshold if (len(results_above_threshold) >= MIN_OTHER_IMAGES) else min_results

    response_data = {"results": [top_result] + final_other_results}

    # The image would be uploaded only if consent is given and
    # The highest score is less than the save similarity threshold
    if consent and (highest_score <= SAVE_SIMILARITY_THRESHOLD):
        if not drive_service or not drive_folder_id:
            logging.warning("Consent given, but Google Drive service is not available. Skipping upload.")
        else:
            try:
                new_uuid = str(uuid.uuid4())
                file_name = f"{new_uuid}.jpg"
                logging.info("Consent given. Uploading %s...", blur_str(file_name))

                with io.BytesIO() as output:
                    image.save(output, format="JPEG")
                    resized_contents = output.getvalue()

                image_embeddings = faiss_index.embedder.compute_embeddings(img=face_img)
                
                uploaded_file_id = upload_bytes_to_folder(
                    service=drive_service,
                    folder_id=drive_folder_id,
                    file_name=file_name,
                    file_bytes=resized_contents, 
                    mime_type="image/jpeg", 
                    uuid_str=new_uuid 
                )

                if uploaded_file_id:
                    image_metadata = [{
                        'gdrive_id': uploaded_file_id,
                        'name': "user_data",
                        'keywords': "user"
                    }]

                    faiss_index.add(
                        embeddings = image_embeddings,
                        metadata = image_metadata
                    )

                    logging.info("Successfully added image to FAISS and Google Drive.")
                    response_data["uuid"] = new_uuid

                else:
                    logging.error("Failed to upload consented image to Google Drive.")
            except Exception as e:
                logging.exception("Error saving consented image: %s", e)
    elif consent:
        logging.warning(
            "Consent given, but image is a likely duplicate (score: %s). Skipping save.", highest_score
        )

    return response_data

# Endpoint for deleting data
@app.delete("/delete/{uuid}")
async def delete_image_by_uuid(uuid: str):
    """
    Deletes a user-uploaded image from Google Drive and FAISS by its UUID.
    """
    global drive_service, faiss_index
    if not drive_service or not faiss_index:
        raise HTTPException(status_code=503, detail="Service is not available.")
    
    try:
        logging.info("Attempting to delete file with UUID: %s", blur_str(uuid))
        image_exists = remove_by_uuid(faiss_index, drive_service, uuid)
        
        if image_exists:
            logging.info(f"Successfully deleted the file with UUID: %s", blur_str(uuid))
            return {
                "success": True,
                "message": f"Successfully deleted image associated with UUID: {blur_str(uuid)}"
            }
        else:
            return {
                "success": False,
                "message": f"There is no image in the database with the provided UUID."
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("An error occurred while deleting the image: %s", e)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.get("/gdrive-image/{file_id}")
async def get_gdrive_image(file_id: str):
    global drive_service
    if not drive_service:
        raise HTTPException(
            status_code=503,
            detail="Google Drive service is not available."
        )
    try:
        image_bytes = get_image_bytes_by_id(drive_service, file_id)
        if image_bytes:
            return Response(content=image_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(
                status_code=404,
                detail="Image data not found in Google Drive."
            )
    except Exception as e:
        logging.exception("An error occurred retrieving GDrive file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# Root Endpoint (GET /)
@app.get("/")
async def read_root():
    return {
        "status": "online",
        "message": "Visual Twin Search API is running. Please visit the frontend website to use the app.",
        "docs": f"{BASE_URL}/docs"
    }

# Run the App
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)