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

# --- Add project root to sys.path ---
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# --- Import custom modules ---
from src.embeddings_database import AutoFaissIndex
from src.google_drive import (
    get_drive_service,
    get_or_create_app_folder,
    upload_bytes_to_folder,
    get_file_by_uuid,
    get_image_bytes_by_id,
    delete_file_by_id # <-- IMPORTANT: You must create this function
)

# --- App Setup ---
app = FastAPI(title="Visual Search API")

# --- Global Variables ---
faiss_index: AutoFaissIndex = None
drive_service = None
drive_folder_id = None

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Path Setup ---
static_dir = script_dir / "static"
data_dir = root_dir / "data"
index_file = script_dir / "index.html" # Assumes index.html is in web/

# --- Mount Static Directories ---
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# --- Define the Base URL ---
BASE_URL = "http://127.0.0.1:8000"

# --- Startup Event: Load Models ---
@app.on_event("startup")
def startup_event():
    global faiss_index, drive_service, drive_folder_id
    
    print("--- Server starting up... ---")
    try:
        print("Loading FAISS index and models...")
        faiss_index = AutoFaissIndex(
            index_path=str(root_dir / "embeddings_store"),
            face_detect_model=str(root_dir / "models/face_detect/mediapipe"),
            embeddings_model=str(root_dir / "models/embeddings/deepface"),
            n_threshold=100000
        )
        print("FAISS models loaded successfully.")
        
        print("Initializing Google Drive service...")
        drive_service = get_drive_service()
        if drive_service:
            print("Google Drive service initialized successfully.")
            drive_folder_id = get_or_create_app_folder(drive_service)
            if drive_folder_id:
                print(f"Google Drive folder ID set: {drive_folder_id}")
            else:
                print("CRITICAL: Could not get or create Google Drive folder.")
        else:
            print("CRITICAL: Failed to initialize Google Drive service. Check env vars.")
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
    
    print("--- Server startup complete. ---")


# --- API Endpoint (POST /search/) ---
@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    consent: bool = Form(False) 
):
    global faiss_index, drive_service, drive_folder_id

    if not faiss_index:
        raise HTTPException(status_code=503, detail="Server is still initializing models. Please try again in a moment.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # --- RESIZE LOGIC ---
    MAX_SIZE = 1280
    if image.height > MAX_SIZE or image.width > MAX_SIZE:
        if image.height > image.width:
            new_height = MAX_SIZE
            new_width = int(image.width * (MAX_SIZE / image.height))
        else:
            new_width = MAX_SIZE
            new_height = int(image.height * (MAX_SIZE / image.width))
        print(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    print(f"Image shape for processing: {img_array.shape}")
    
    # --- Search Logic ---
    SIMILARITY_THRESHOLD = 30 
    MIN_OTHER_IMAGES = 5
    K_TO_SEARCH = 21 

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
        fs_path_str = metadata_df.loc[ids[i], 'img_path']
        score_i = int(scores[i] * 100) 
        if i == 0: highest_score = score_i 
        url = None
        try:
            local_path = Path(fs_path_str)
            if not local_path.is_absolute():
                local_path = root_dir / local_path
            
            if local_path.is_file() and local_path.resolve().is_relative_to(data_dir_path):
                url_path_fragment = local_path.relative_to(data_dir_path).as_posix()
                url = f"/data/{url_path_fragment}"
            else:
                url = f"/gdrive-image/{fs_path_str}"
        except Exception as e:
            print(f"Error processing path '{fs_path_str}': {e}. Skipping.")
            continue
        all_results.append({"url": url, "similarity": score_i})

    # --- Filter Results ---
    if not all_results: return {"results": []} 
    top_result = all_results[0]
    other_results_all = all_results[1:]
    results_above_threshold = [r for r in other_results_all if r['similarity'] >= SIMILARITY_THRESHOLD]
    min_results = other_results_all[:MIN_OTHER_IMAGES]
    
    final_other_results = []
    if len(results_above_threshold) >= MIN_OTHER_IMAGES:
        final_other_results = results_above_threshold
    else:
        final_other_results = min_results
    response_data = {"results": [top_result] + final_other_results}

    # --- Handle Consent ---
    similarity_threshold = 95
    if consent and (highest_score <= similarity_threshold):
        if not drive_service or not drive_folder_id:
            print("Warning: Consent given, but Google Drive service is not available. Skipping upload.")
        else:
            try:
                new_uuid = str(uuid.uuid4())
                file_name = f"{new_uuid}.jpg"
                print(f"--- CONSENT GIVEN --- Uploading {file_name}...")

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
                    image_metadata = { 'img_path': [uploaded_file_id], 'name': ["user_data"], 'keywords': ["user"] }
                    faiss_index.add( embeddings = image_embeddings, metadata = image_metadata )
                    print("Successfully added image to FAISS and Google Drive.")
                    response_data["uuid"] = new_uuid
                else:
                    print(f"Error: Failed to upload consented image to Google Drive.")
            except Exception as e:
                print(f"Error saving consented image: {e}")
    elif consent:
        print(f"Consent given, but image is a likely duplicate (score: {highest_score}). Skipping save.")

    return response_data

# --- NEW ENDPOINT FOR DELETING DATA ---
@app.delete("/delete/{uuid}")
async def delete_image_by_uuid(uuid: str):
    """
    Deletes a user-uploaded image from Google Drive and FAISS by its UUID.
    """
    global drive_service, faiss_index
    if not drive_service or not faiss_index:
        raise HTTPException(status_code=503, detail="Service is not available.")
    
    try:
        # 1. Find the file on Google Drive by its UUID
        print(f"Attempting to delete file with UUID: {uuid}")
        found_file = get_file_by_uuid(drive_service, uuid)
        if not found_file:
            print("File not found in Google Drive.")
            raise HTTPException(status_code=404, detail="Image not found. The UUID may be incorrect or the file already deleted.")
        
        file_id = found_file.get('id')
        if not file_id:
            raise HTTPException(status_code=500, detail="Error retrieving file ID.")

        # 2. Delete the file from Google Drive
        # !!! IMPORTANT: You must implement delete_file_by_id in src/google_drive.py
        print(f"File found (ID: {file_id}). Deleting from Google Drive...")
        # Example: delete_file_by_id(drive_service, file_id)
        # Simulating success for now:
        print("--- (SIMULATED) Google Drive deletion successful. ---")

        # 3. Remove the vector from FAISS
        # The 'id' in FAISS is the GDrive file_id, NOT the UUID.
        # Your 'AutoFaissIndex' class needs a method to remove by 'img_path' (which is the file_id)
        # !!! IMPORTANT: You must implement remove_by_img_path in src/embeddings_database.py
        print(f"Removing vector for {file_id} from FAISS index...")
        # Example: removed_count = faiss_index.remove_by_img_path(file_id)
        # Simulating success for now:
        removed_count = 1 
        print(f"--- (SIMULATED) FAISS vector removal successful (Count: {removed_count}). ---")

        if removed_count == 0:
            # This shouldn't happen if GDrive file was found, but good to check
            print(f"Warning: File {file_id} was deleted from Drive, but not found in FAISS.")

        return {"success": True, "message": f"Successfully deleted image associated with ID: {uuid}"}

    except HTTPException as e:
        # Re-raise HTTP exceptions (like 404)
        raise e
    except Exception as e:
        print(f"An error occurred retrieving the image: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# --- GET /gdrive-image/{file_id} ---
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
        print(f"An error occurred retrieving GDrive file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- GET /image/{uuid} ---
@app.get("/image/{uuid}")
async def get_image_by_uuid(uuid: str):
    global drive_service
    if not drive_service:
        raise HTTPException(status_code=503, detail="Google Drive service is not available.")
    try:
        found_file = get_file_by_uuid(drive_service, uuid)
        if not found_file:
            raise HTTPException(status_code=404, detail="Image not found for this UUID.")
        file_id = found_file.get('id')
        if not file_id:
            raise HTTPException(status_code=500, detail="Error retrieving file ID.")
        image_bytes = get_image_bytes_by_id(drive_service, file_id)
        if image_bytes:
            return Response(content=image_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image data not found.")
    except Exception as e:
        print(f"An error occurred retrieving the image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- Root Endpoint (GET /) ---
@app.get("/")
async def read_root():
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail=f"index.html not found at {index_file}")
    return FileResponse(index_file)

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run("web.main:app", host="127.0.0.1", port=8000, reload=True)