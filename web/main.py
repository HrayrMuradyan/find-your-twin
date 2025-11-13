import uvicorn
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings_database import AutoFaissIndex
from urllib.parse import quote
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# --- App Setup ---
app = FastAPI(title="Visual Search API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for this demo
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Mount the Static Files Directory ---
script_dir = Path(__file__).parent 
root_dir = script_dir.parent 
data_dir = root_dir / "data"
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# ADD THIS ⬇️
app.mount("/static", StaticFiles(directory=script_dir / "static"), name="static")


# --- NEW: Define the Base URL for your API server ---
# This tells the frontend where to find the static files
BASE_URL = "http://127.0.0.1:8000"


# --- API Endpoint ---
@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs a FAISS search, 
    and returns top-6 results with full URLs.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    
    print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
    
    faiss_index = AutoFaissIndex(
                 index_path="./embeddings_store/",
                 face_detect_model="./models/face_detect/mediapipe/",
                 embeddings_model="./models/embeddings/deepface/",
                 n_threshold=100000
    ) 

    scores, ids, metadata = faiss_index.search_image(img_array, k=6)
    scores, ids, metadata = scores[0], ids[0], metadata[0]
    
    img_paths = list(metadata.loc[ids, 'img_path'])
    
    # --- THIS IS THE FIX ---
    # Convert the raw file paths into full, absolute URLs
    results = []
    for i in range(6):
        # Get the raw file system path (e.g., "data\kaggle_ashwingupta3012\...")
        fs_path_str = img_paths[i]
        
        # Convert it to a POSIX-style path (e.g., "data/kaggle_ashwingupta3012/...")
        url_path_fragment = quote(Path(fs_path_str).as_posix())
        
        # Create the full URL the browser can use
        full_url = f"{BASE_URL}/{url_path_fragment}"
        
        results.append({
            "url": full_url, 
            "similarity": int(scores[i])
        })
    # --- END OF FIX ---

    print(results) # This will now print full URLs

    # Return the results in the required format
    return {"results": results}

from fastapi.responses import HTMLResponse

# --- Serve index.html at root ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = script_dir / "index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Run the App ---
if __name__ == "__main__":
    # This assumes you run from the root directory (e.g., `python web/main.py`)
    # If you `cd web` and run `python main.py`, change this to "main:app"
    uvicorn.run("web.main:app", host="127.0.0.1", port=8000, reload=True)