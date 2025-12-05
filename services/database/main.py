from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import sys
from pathlib import Path
import uvicorn

# Add project root to path to allow importing from 'src'
# Docker will copy 'src' to /app/src, so we need /app in sys.path
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.search_engine import VectorDB
from src.logging_config import setup_logging
from src.config import load_config

# Setup
setup_logging()
logger = logging.getLogger("DatabaseService")
CONFIG = load_config()

app = FastAPI(title="Database & Search Service")

# Global Vector DB Instance
# This holds the FAISS index in RAM
vectordb: VectorDB = None

@app.on_event("startup")
def startup_event():
    global vectordb
    try:
        logger.info("Initializing VectorDB (FAISS + Postgres)...")
        # This triggers the load from Postgres -> FAISS
        vectordb = VectorDB()
        logger.info("VectorDB Ready. Indexed %s vectors.", vectordb.index.ntotal if vectordb.index else 0)
    except Exception as e:
        logger.critical("Failed to initialize VectorDB: %s", e)
        sys.exit(1)

# --- Pydantic Models ---

class SearchPayload(BaseModel):
    vector: List[float]
    k: int = 5

class AddPayload(BaseModel):
    id: int # The Postgres ID
    vector: List[float]
    metadata: Dict[str, Any]

class DeletePayload(BaseModel):
    id: int

# --- Endpoints ---

@app.post("/search")
def search(payload: SearchPayload):
    """
    Receives a vector, searches FAISS, returns nearest neighbors.
    """
    if not vectordb:
        raise HTTPException(503, "Database not initialized")
    
    try:
        # returns list of dicts: [{'id': 1, 'score': 0.8, ...}]
        results = vectordb.search(payload.vector, payload.k)
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(500, str(e))

@app.post("/add")
def add_item(payload: AddPayload):
    """
    Adds a new vector to the running FAISS index.
    """
    if not vectordb:
        raise HTTPException(503, "Database not initialized")
    
    try:
        vectordb.add_item(payload.id, payload.vector, payload.metadata)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Add error")
        raise HTTPException(500, str(e))

@app.post("/delete")
def delete_item(payload: DeletePayload):
    """
    Removes a vector from the running FAISS index.
    """
    if not vectordb:
        raise HTTPException(503, "Database not initialized")
    
    try:
        vectordb.remove_item(payload.id)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Delete error")
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    count = vectordb.index.ntotal if vectordb and vectordb.index else 0
    return {"status": "ok", "count": count}

if __name__ == "__main__":
    # Hugging Face Spaces expects port 7860
    uvicorn.run("main:app", host="0.0.0.0", port=7860)