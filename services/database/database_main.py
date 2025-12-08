from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import sys
from pathlib import Path
import uvicorn
import psycopg2 
import os

# Add project root to path to allow importing from 'src'
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.search_engine import VectorDB
from src.logging_config import setup_logging
from src.config import load_config

# Setup logging
setup_logging()
logger = logging.getLogger("DatabaseService")

# Load config
CONFIG = load_config()

app = FastAPI(title="Database & Search Service")

# Global Vector DB Instance
vectordb: VectorDB = None

@app.on_event("startup")
def startup_event():
    global vectordb
    try:
        logger.info("Initializing VectorDB (FAISS + Postgres)...")
        vectordb = VectorDB()
        logger.info("VectorDB Ready. Indexed %s vectors.", vectordb.index.ntotal if vectordb.index else 0)
    except Exception as e:
        logger.critical("Failed to initialize VectorDB: %s", e)
        sys.exit(1)

# Payload classes
class SearchPayload(BaseModel):
    vector: List[float]
    k: int = 5

class AddPayload(BaseModel):
    vector: List[float]
    drive_file_id: str
    source: str
    metadata: Dict[str, Any]

class DeletePayload(BaseModel):
    drive_file_id: str

# Helper for DB Connection
def get_db_connection():
    # Use config or env directly
    dsn = os.getenv("DB_CONNECTION_STRING")
    if not dsn:
        raise ValueError("DB_CONNECTION_STRING not found")
    return psycopg2.connect(dsn)

@app.post("/search")
def search(payload: SearchPayload):
    if not vectordb:
        raise HTTPException(503, "Database not initialized")
    try:
        results = vectordb.search(payload.vector, payload.k)
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(500, str(e))

@app.post("/add")
def add_item(payload: AddPayload):
    """
    1. Inserts metadata into PostgreSQL
    2. Adds vector to in-memory FAISS index
    """
    if not vectordb:
        raise HTTPException(503, "Database not initialized")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL Insert
        cursor.execute("""
            INSERT INTO face_data (source, drive_file_id, embedding)
            VALUES (%s, %s, %s)
            RETURNING id;
        """, (payload.source, payload.drive_file_id, str(payload.vector))) 
        
        new_id = cursor.fetchone()[0]
        conn.commit()
        
        # FAISS Insert
        vectordb.add_item(new_id, payload.vector, payload.metadata)
        
        return {"status": "ok", "id": new_id}
        
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        logger.exception("Add error")
        raise HTTPException(500, "Failed to persist data")
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'conn' in locals() and conn: conn.close()

@app.post("/delete")
def delete_item(payload: DeletePayload):
    if not vectordb: raise HTTPException(503, "Database not initialized")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get ID from Postgres using drive_file_id
        cursor.execute("SELECT id FROM face_data WHERE drive_file_id = %s", (payload.drive_file_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Item not found")
        
        db_id = row[0]
        
        # SQL Delete
        cursor.execute("DELETE FROM face_data WHERE id = %s", (db_id,))
        conn.commit()
        
        # FAISS Delete
        vectordb.remove_item(db_id)
        
        return {"status": "ok"}
        
    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        logger.exception("Delete error")
        raise HTTPException(500, str(e))
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'conn' in locals() and conn: conn.close()

@app.get("/health")
def health():
    count = vectordb.index.ntotal if vectordb and vectordb.index else 0
    return {"status": "ok", "count": count}

if __name__ == "__main__":
    # Local dev port from env
    port = int(os.getenv("DATABASE_PORT", 8800))
    uvicorn.run("main:app", host="0.0.0.0", port=port)