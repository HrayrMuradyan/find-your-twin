import faiss
import numpy as np
import psycopg2
import os
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class VectorDB:
    def __init__(self):
        self.db_dsn = os.getenv("DB_CONNECTION_STRING")
        if not self.db_dsn:
            logging.error("DB_CONNECTION_STRING environment variable is missing")
            raise ValueError("Missing DB Config")
        
        self.index = None
        self.metadata_map = {}
        self._build_faiss_from_postgres()

    def _build_faiss_from_postgres(self):
        logger.info("Building FAISS Index from Postgres...")
        try:
            conn = psycopg2.connect(self.db_dsn)
            cursor = conn.cursor()

            # 1. Get Dimension
            cursor.execute("SELECT value FROM index_metadata WHERE key = 'dim'")
            row = cursor.fetchone()
            if not row:
                logging.error("Dimension not found in DB Metadata.")
                raise ValueError("Dimension metadata not found in DB")
            dim = int(row[0])

            # 2. Init FAISS (IndexFlatIP = Inner Product = Cosine Sim if normalized)
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

            # 3. Fetch Data
            cursor.execute("SELECT id, embedding, source, drive_file_id FROM face_data")
            rows = cursor.fetchall()

            ids_list = []
            vectors_list = []

            for row in rows:
                db_id, emb_raw, source, drive_id = row
                
                # Parse JSON vector if stored as string
                vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                
                ids_list.append(db_id)
                vectors_list.append(vec)

                # Keep metadata in memory for quick lookup
                self.metadata_map[db_id] = {
                    "name": source,
                    "drive_id": drive_id,
                    "drive_link": f"[https://drive.google.com/uc?id=](https://drive.google.com/uc?id=){drive_id}"
                }
            
            # 4. Add to Index
            if ids_list:
                ids_np = np.array(ids_list, dtype=np.int64)
                vecs_np = np.array(vectors_list, dtype=np.float32)
                
                # Normalize L2 is crucial for Cosine Similarity using Inner Product
                faiss.normalize_L2(vecs_np) 
                
                self.index.add_with_ids(vecs_np, ids_np)

            logger.info("Index built. Total vectors: %s", self.index.ntotal)
            cursor.close()
            conn.close()

        except Exception as e:
            logger.exception("Failed to build index: %s", e)
            raise e
    
    def search(self, vector, k=5):
        if not self.index or self.index.ntotal == 0:
            return []
        
        query_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        distances, indices = self.index.search(query_np, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
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
        vec_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec_np)
        id_np = np.array([db_id], dtype=np.int64)

        self.index.add_with_ids(vec_np, id_np)
        self.metadata_map[db_id] = metadata
        logger.info("Added ID %s to FAISS index", db_id)

    def remove_item(self, db_id):
        id_np = np.array([db_id], dtype=np.int64)
        self.index.remove_ids(id_np)
        if db_id in self.metadata_map:
            del self.metadata_map[db_id]
        logger.info("Removed ID %s from FAISS index", db_id)