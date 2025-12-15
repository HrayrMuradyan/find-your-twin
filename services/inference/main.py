import uvicorn
import numpy as np
from PIL import Image, ImageOps
import io
import threading  # <--- ADDED
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import logging
import os

from dotenv import load_dotenv 

# Load the variables from .env 
load_dotenv()

script_dir = Path(__file__).parent

# Import custom modules 
from find_your_twin.embeddings_database import DatabaseServiceClient
from find_your_twin.google_drive import (
    get_drive_service,
    get_or_create_app_folder,
    get_image_bytes_by_id
)
from find_your_twin.config import load_config
from find_your_twin.image import resize_image
from find_your_twin.logging_config import setup_logging
from find_your_twin.credentials import setup_google_credentials

# Setup logger and config
setup_logging()
logger = logging.getLogger(__name__)

CONFIG = load_config()

# App initialization and CORS
app = FastAPI(title="Inference Service")

INFERENCE_PORT = os.getenv("INFERENCE_PORT", "7860")
FRONTEND_PORT = os.getenv("FRONTEND_PORT", "8000")

allow_origins = [
    f"http://localhost:{FRONTEND_PORT}",
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://localhost:{INFERENCE_PORT}",
    f"http://127.0.0.1:{INFERENCE_PORT}",

    "https://hrayrmuradyan.com/",
    "https://hrayrmuradyan-find-your-twin-inference.hf.space"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Global State
db_client: DatabaseServiceClient = None
drive_folder_id = None

# Thread Local Storage
# Used for multithreading to improve google drive downloading
_thread_local = threading.local()

def get_thread_safe_service():
    """
    Returns a Google Drive service instance unique to the current thread.
    """
    if not hasattr(_thread_local, "service"):
        # If this thread doesn't have a service yet, create one.
        _thread_local.service = get_drive_service()
    return _thread_local.service

# Wrapper for the threaded execution
def _download_worker(file_id: str):
    service = get_thread_safe_service()
    return get_image_bytes_by_id(service, file_id)


# Configs
FACE_DETECT_MODEL = Path(CONFIG['models']['paths']['face_detect_model'])
EMBEDDINGS_MODEL = Path(CONFIG['models']['paths']['embeddings_model'])
IMAGE_MAX_SIZE = int(CONFIG['image']['max_size'])

# Search configuration
RETRIEVAL_SIMILARITY_THRESHOLD = int(CONFIG['search']['retrieval_similarity_threshold'])
MIN_OTHER_IMAGES = int(CONFIG['search']['min_other_images'])
K_TO_SEARCH = int(CONFIG['search']['k_to_search']) 
SAVE_SIMILARITY_THRESHOLD = int(CONFIG['search']['save_similarity_threshold'])


@app.on_event("startup")
def startup_event():
    global db_client, drive_folder_id
    
    logger.info("--- Inference Service Starting ---")
    try:
        logger.info("Initializing the Credentials...")
        setup_google_credentials(script_dir)

        # Initialize one service just to check connection & get folder ID
        logger.info("Initializing Google Drive service (Main Thread)...")
        main_service = get_drive_service()
        
        if main_service:
            drive_folder_id = get_or_create_app_folder(main_service)
            logger.info("Google Drive initialized. Folder ID: %s", drive_folder_id)
        else:
            logger.error("Failed to initialize Google Drive. Check ENV vars.")

        logger.info("Initializing ML Models & Database Connection...")

        db_client = DatabaseServiceClient(
            face_detect_model=FACE_DETECT_MODEL,
            embeddings_model=EMBEDDINGS_MODEL,
            drive_service=main_service, 
            drive_folder_id=drive_folder_id
        )
        logger.info("Inference Service Ready")

    except Exception as e:
        logger.exception("Startup failed: %s", e)


@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    consent: bool = Form(False) 
):
    global db_client
    if not db_client:
        raise HTTPException(status_code=503, detail="Server initializing.")

    # Read & Preprocess Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = resize_image(image, IMAGE_MAX_SIZE)
    img_array = np.array(image)
    
    try:
        # Detect -> Embed -> POST to Database Service
        similar_images = await db_client.search_image(img_array, k=K_TO_SEARCH)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Processing error.")
    
    # Response formatting
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

    if formatted_results:
        top_result = formatted_results[0]
        other_results = formatted_results[1:]
        
        MAX_OTHER_IMAGES = 10 
        
        valid_others = [
            res for res in other_results if res['similarity'] >= RETRIEVAL_SIMILARITY_THRESHOLD
        ]
        final_others = valid_others if len(valid_others) >= MIN_OTHER_IMAGES else other_results[:MIN_OTHER_IMAGES]
        
        response_data = {"results": [top_result] + final_others[:MAX_OTHER_IMAGES]}
    else:
        response_data = {"results": []}

    # If consent is given, save the image
    if consent:
        if highest_similarity > SAVE_SIMILARITY_THRESHOLD:
            logger.info("Consent given, but duplicate (Score: %s). Skipping.", highest_similarity)
        else:
            try:
                logger.info("Consent given. Saving new face...")
                new_uuid = await db_client.add_image(
                    image=img_array, 
                    metadata={'name': 'user_data', 'filename': file.filename}
                )
                response_data["uuid"] = new_uuid
            except Exception as e:
                logger.exception("Failed to save consented image: %s", e)

    return response_data


@app.delete("/delete/{uuid}")
async def delete_image_by_uuid(uuid: str):
    global db_client
    if not db_client:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    
    try:
        success = await db_client.delete_image_by_uuid(uuid)
        if success:
            return {"success": True, "message": f"Deleted {uuid}"}
        else:
            return {"success": False, "message": "UUID not found or delete failed."}
    except Exception as e:
        logger.exception("Delete error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error.")


@app.get("/gdrive-image/{file_id}")
async def get_gdrive_image(file_id: str):
    # Check if folder ID is loaded (implies system is ready)
    if not drive_folder_id:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    
    try:
        # Run download in a thread, using a THREAD-LOCAL service instance
        image_bytes = await run_in_threadpool(_download_worker, file_id)
        
        if image_bytes:
            # Cache for 1 year
            return Response(
                content=image_bytes, 
                media_type="image/jpeg",
                headers={}
            )
        else:
            raise HTTPException(status_code=404, detail="Image not found.")
            
    except Exception as e:
        logger.exception("Proxy error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error.")

@app.get("/stats")
async def get_stats():
    global db_client
    if not db_client:
        return {"count": 0}
    return await db_client.get_index_count()

@app.get("/")
def read_root():
    return {"status": "Inference Service Online"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("INFERENCE_PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)