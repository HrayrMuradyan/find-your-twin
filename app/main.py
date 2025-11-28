# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

import uvicorn
import numpy as np
from PIL import Image, ImageOps
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from pathlib import Path
import logging

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
    get_image_bytes_by_id
)
from src.config import load_config
from src.utils import blur_str
from src.image import resize_image
from src.logging_config import setup_logging

# Setup logger
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

CONFIG = load_config()
app = FastAPI(title=CONFIG['app']['title'])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Global State
faiss_index: AutoFaissIndex = None
drive_service = None
drive_folder_id = None

BASE_URL = CONFIG['app']['base_url']
FACE_DETECT_MODEL = PROJECT_ROOT / CONFIG['models']['paths']['face_detect_model']
EMBEDDINGS_MODEL = PROJECT_ROOT / CONFIG['models']['paths']['embeddings_model']
IMAGE_MAX_SIZE = int(CONFIG['image']['max_size'])

# Search thresholds
RETRIEVAL_SIMILARITY_THRESHOLD = int(CONFIG['search']['retrieval_similarity_threshold'])
MIN_OTHER_IMAGES = int(CONFIG['search']['min_other_images'])
K_TO_SEARCH = int(CONFIG['search']['k_to_search']) 
SAVE_SIMILARITY_THRESHOLD = int(CONFIG['search']['save_similarity_threshold'])

# ---------------------------------------------------
# STARTUP
# ---------------------------------------------------
@app.on_event("startup")
def startup_event():
    global faiss_index, drive_service, drive_folder_id
    
    logger.info("--- Server starting up... ---")
    try:
        # 1. Init Google Drive
        logger.info("Initializing Google Drive service...")
        drive_service = get_drive_service()
        if drive_service:
            drive_folder_id = get_or_create_app_folder(drive_service)
            logger.info("Google Drive initialized. Folder ID: %s", drive_folder_id)
        else:
            logger.error("Failed to initialize Google Drive. Check ENV vars.")

        # Init FAISS and Database
        logger.info("Initializing Database & FAISS...")
        faiss_index = AutoFaissIndex(
            face_detect_model=FACE_DETECT_MODEL,
            embeddings_model=EMBEDDINGS_MODEL,
            drive_service=drive_service,
            drive_folder_id=drive_folder_id
        )
        logger.info("System ready")

    except Exception as e:
        logger.exception("Startup failed: %s", e)

# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------

@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    consent: bool = Form(False) 
):
    global faiss_index
    if not faiss_index:
        raise HTTPException(status_code=503, detail="Server initializing.")

    # Read & Preprocess Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = resize_image(image, IMAGE_MAX_SIZE)
    img_array = np.array(image)
    
    # Search
    try:
        # We request 3 values: distances, indices, and the list of result dicts
        _, _, similar_images = faiss_index.search_image(img_array, k=K_TO_SEARCH)
    except ValueError:
        raise HTTPException(status_code=400, detail="No face detected in upload.")
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Processing error.")
    
    # Format results for frontend
    if not similar_images:
        # Even if no results, we might still want to save if consent is given
        formatted_results = []
        highest_similarity = 0
    else:
        formatted_results = []
        highest_similarity = 0
        
        for similar_i in similar_images:
            raw_score = similar_i['score']
            similarity_score = int(max(0, raw_score) * 100)
            
            formatted_results.append({
                "url": f"/gdrive-image/{similar_i['drive_id']}",
                "similarity": similarity_score
            })
            
            if similarity_score > highest_similarity:
                highest_similarity = similarity_score

    # Prepare response data
    if formatted_results:
        top_result = formatted_results[0]
        other_results = formatted_results[1:]
        
        valid_others = [res for res in other_results if res['similarity'] >= RETRIEVAL_SIMILARITY_THRESHOLD]
        final_others = valid_others if len(valid_others) >= MIN_OTHER_IMAGES else other_results[:MIN_OTHER_IMAGES]
        
        response_data = {"results": [top_result] + final_others}
    else:
        response_data = {"results": []}

    # Handle Consent (Save Image)
    if consent:
        # Check if it's a duplicate (using the configured threshold)
        if highest_similarity > SAVE_SIMILARITY_THRESHOLD:
            logger.info("Consent given, but image is likely duplicate (Score: %s). Skipping save.", highest_similarity)
        else:
            try:
                logger.info("Consent given. Saving new face...")
                
                # Save the image
                new_uuid = faiss_index.add_image(
                    image=img_array, 
                    metadata={'name': 'user_data'}
                )
                
                # Return the UUID to the frontend
                response_data["uuid"] = new_uuid
                logger.info("Saved successfully.")
            except Exception as e:
                logger.exception("Failed to save consented image: %s", e)

    return response_data


@app.delete("/delete/{uuid}")
async def delete_image_by_uuid(uuid: str):
    """
    Deletes an image by UUID from Drive, DB, and RAM.
    """
    global faiss_index
    if not faiss_index:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    
    try:
        success = faiss_index.delete_image_by_uuid(uuid)
        if success:
            return {"success": True, "message": f"Deleted {uuid}"}
        else:
            return {"success": False, "message": "UUID not found or delete failed."}
    except Exception as e:
        logger.exception("Delete error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error.")


@app.get("/gdrive-image/{file_id}")
async def get_gdrive_image(file_id: str):
    global drive_service
    if not drive_service:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    try:
        image_bytes = get_image_bytes_by_id(drive_service, file_id)
        if image_bytes:
            return Response(content=image_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        logger.exception("Proxy error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error.")

@app.get("/")
async def read_root():
    return {
        "status": "online",
        "message": "Visual Twin Search API is running.",
        "docs": f"{BASE_URL}/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)