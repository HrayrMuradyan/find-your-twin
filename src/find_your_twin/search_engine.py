import faiss
import numpy as np
import psycopg2
import os
import json
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)
load_dotenv()

class VectorDB:
    def __init__(self):
        # Load PostgreSQL DB connection string from environment
        self.db_dsn = os.getenv("DB_CONNECTION_STRING")
        if not self.db_dsn:
            logging.error("DB_CONNECTION_STRING environment variable is missing")
            raise ValueError("Missing DB Config")
        
        # Build FAISS index from existing vectors in Postgres

        self.index = None          
        self.metadata_map = {}    

        self._build_faiss_from_postgres()

    def _build_faiss_from_postgres(self):
        logger.info("Building FAISS Index from Postgres...")
        try:
            # Connect to postgres
            conn = psycopg2.connect(self.db_dsn)
            cursor = conn.cursor()

            # Fetch stored embedding dimension from metadata table
            # Used to create FAISS
            cursor.execute("SELECT value FROM index_metadata WHERE key = 'dim'")
            row = cursor.fetchone()
            if not row:
                logging.error("Dimension not found in DB Metadata.")
                raise ValueError("Dimension metadata not found in DB")
            dim = int(row[0])

            # Initialize FAISS index
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

            # Fetch all stored embeddings and metadata
            cursor.execute("SELECT id, embedding, source, drive_file_id FROM face_data")
            rows = cursor.fetchall()

            ids_list = []
            vectors_list = []

            for row in rows:
                db_id, emb_raw, source, drive_id = row

                # Convert embedding from JSON text to Python list if needed
                vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                
                ids_list.append(db_id)
                vectors_list.append(vec)

                # Store metadata for fast lookup during search
                self.metadata_map[db_id] = {
                    "name": source,
                    "drive_id": drive_id,
                    "drive_link": f"https://drive.google.com/uc?id={drive_id}"
                }
            
            # Add embeddings to FAISS index, if any exist
            if ids_list:
                ids_np = np.array(ids_list, dtype=np.int64)
                vecs_np = np.array(vectors_list, dtype=np.float32)
                
                # Normalize vectors
                faiss.normalize_L2(vecs_np)
                
                self.index.add_with_ids(vecs_np, ids_np)

            logger.info("Index built. Total vectors: %s", self.index.ntotal)
            cursor.close()
            conn.close()

        except Exception as e:
            logger.exception("Failed to build index: %s", e)
            raise e
    
    def search(self, vector, k=5):
        # Return empty list if index is not ready
        if not self.index or self.index.ntotal == 0:
            return []
        
        # Prepare query vector for FAISS
        query_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Perform similarity search
        distances, indices = self.index.search(query_np, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            # Skip invalid IDs
            if idx != -1 and idx in self.metadata_map:
                item = self.metadata_map[idx]
                results.append({
                    "id": int(idx),
                    "score": float(score),
                    "name": item['name'],
                    "drive_id": item['drive_id'],
                    "drive_link": item['drive_link']
                })
        return results
    
    def add_item(self, db_id, vector, metadata):
        # Add new vector to FAISS
        vec_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec_np)
        id_np = np.array([db_id], dtype=np.int64)

        self.index.add_with_ids(vec_np, id_np)

        # Store metadata for this vector
        self.metadata_map[db_id] = metadata
        logger.info("Added ID %s to FAISS index", db_id)

    def remove_item(self, db_id):
        # Remove the vector from FAISS
        id_np = np.array([db_id], dtype=np.int64)
        self.index.remove_ids(id_np)

        # Remove metadata
        if db_id in self.metadata_map:
            del self.metadata_map[db_id]

        logger.info("Removed ID %s from FAISS index", db_id)