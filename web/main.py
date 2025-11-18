# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

import uvicorn
import numpy as np
from PIL import Image
import io
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import sys, os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
from pathlib import Path
import logging

# Add project root to sys.path
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

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
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# DECLARING IMPORTANT VARIABLES
# ---------------------------------------------------

# Read the config
config = load_config()

# App Setup 
app = FastAPI(title=config['app']['title'])

# CORS Middlewar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Path Setup ---
static_dir = root_dir / config['app']['paths']['static']
data_dir = root_dir / config['app']['paths']['data']
index_file = root_dir / config['app']['paths']['index_html']

# --- Mount Static Directories ---
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static"
)
app.mount("/data",
          StaticFiles(directory=data_dir),
          name="data"
)

# Global Variables
faiss_index: AutoFaissIndex = None
drive_service = None
drive_folder_id = None

# Define the Base URL
BASE_URL = config['app']['base_url']

# Other important global variables
FACE_DETECT_MODEL = root_dir / config['models']['paths']['face_detect_model']

EMBEDDINGS_MODEL = root_dir / config['models']['paths']['embeddings_model']

FAISS_INDEX_PATH = root_dir / config['faiss']['paths']['index_path']

IMAGE_MAX_SIZE = int(config['image']['max_size'])

# Search variables
RETRIEVAL_SIMILARITY_THRESHOLD = int(config['search']['retrieval_similarity_threshold'])
MIN_OTHER_IMAGES = int(config['search']['min_other_images'])
K_TO_SEARCH = int(config['search']['k_to_search']) 
SAVE_SIMILARITY_THRESHOLD = int(config['search']['save_similarity_threshold'])

# On Startup, Load Models ---
@app.on_event("startup")
def startup_event():
    global faiss_index, drive_service, drive_folder_id
    
    logging.info("--- Server starting up... ---")
    try:
        logging.info(
            "Loading FAISS index from %s...",
            FAISS_INDEX_PATH.relative_to(root_dir)
        )

        faiss_index = AutoFaissIndex(
            index_path=FAISS_INDEX_PATH,
            face_detect_model=FACE_DETECT_MODEL,
            embeddings_model=EMBEDDINGS_MODEL,
        )

        logging.info("FAISS Index loaded successfully.")
        
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
            logging.error(
                "Failed to initialize Google Drive service. Check ENV vars."
            )
    except Exception as e:
        logging.error("Something went wrong during startup: %s", e)
    
    logging.info("--- Server startup complete. ---")


# --- API Endpoint (POST /search/) ---
@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    consent: bool = Form(False) 
):
    logging.info("Starting an image search...")
    global faiss_index, drive_service, drive_folder_id

    if not faiss_index:
        raise HTTPException(status_code=503, detail="Server is still initializing models. Please try again in a moment.")

    # Get the image from the upload
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Resize the image
    if image.height > IMAGE_MAX_SIZE or image.width > IMAGE_MAX_SIZE:
        if image.height > image.width:
            new_height = IMAGE_MAX_SIZE
            new_width = int(image.width * (IMAGE_MAX_SIZE / image.height))
        else:
            new_width = IMAGE_MAX_SIZE
            new_height = int(image.height * (IMAGE_MAX_SIZE / image.width))

        logging.info(
            "Resizing image from %sx%s to %sx%s",
            image.width,
            image.height,
            new_width,
            new_height
        )

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logging.info("Resizing successful")
    
    img_array = np.array(image)
    logging.info(
        "Image shape for processing: %s", img_array.shape
    )
    
    # Search Logic
    scores, ids, metadata, face_img = faiss_index.search_image(
        img_array, k=K_TO_SEARCH, return_metadata=True, return_face=True
    )
    
    if scores is None or len(scores[0]) == 0:
        return {"results": []} 
    scores, ids, metadata_df = scores[0], ids[0], metadata[0]
    
    all_results = []
    highest_score = 0
    data_dir_path = Path(data_dir).resolve()

    for i in range(len(ids)):
        # Image path here can be either upload ID or Local Path
        image_path_str = metadata_df.loc[ids[i], 'img_path']
        score_i = int(scores[i] * 100) 

        if i == 0:
            highest_score = score_i 

        image_url = None
        try:
            local_path = Path(image_path_str)
            if not local_path.is_absolute():
                local_path = root_dir / local_path
            
            # If the path is a file and is relative to the data directory
            if local_path.is_file() and local_path.resolve().is_relative_to(data_dir_path):
                url_path_fragment = local_path.relative_to(data_dir_path).as_posix()
                image_url = f"/data/{url_path_fragment}"
            else:
                image_url = f"/gdrive-image/{image_path_str}"
        except Exception as e:
            logging.exception(
                "Error processing path '%s': %s. Skipping.",
                image_path_str,
                e
            )
            continue
        all_results.append({"url": image_url, "similarity": score_i})

    # Filter results
    if not all_results:
        return {"results": []} 
    
    # Get the top result
    # Get at least MIN_OTHER_IMAGES results
    # The high bound is decided using RETRIEVAL_SIMILARITY_THRESHOLD
    top_result = all_results[0]
    other_results_all = all_results[1:]
    results_above_threshold = [r for r in other_results_all if r['similarity'] >= RETRIEVAL_SIMILARITY_THRESHOLD]
    min_results = other_results_all[:MIN_OTHER_IMAGES]
    
    final_other_results = []
    final_other_results = results_above_threshold if (len(results_above_threshold) >= MIN_OTHER_IMAGES) else min_results

    response_data = {"results": [top_result] + final_other_results}

    # Handle Consent
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
                        'img_path': uploaded_file_id,
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
        logging.warning("Consent given, but image is a likely duplicate (score: %s}). Skipping save.", highest_score)

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
            logging.info(f"Successfuly deleted the file with UUID: %s", blur_str(uuid))
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
        raise HTTPException(status_code=503, detail="Google Drive service is not available.")
    try:
        image_bytes = get_image_bytes_by_id(drive_service, file_id)
        if image_bytes:
            return Response(content=image_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image data not found in Google Drive.")
    except Exception as e:
        logging.exception("An error occurred retrieving GDrive file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# Root Endpoint (GET /)
@app.get("/")
async def read_root():
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail=f"index.html not found at {index_file}")
    return FileResponse(index_file)

# Run the App
if __name__ == "__main__":
    uvicorn.run("web.main:app", host="127.0.0.1", port=8000, reload=True)