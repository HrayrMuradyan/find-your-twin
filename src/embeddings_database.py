from pathlib import Path
import faiss
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import io
import os
import psycopg2
from dotenv import load_dotenv
from PIL import Image
import concurrent.futures
import json 
from tqdm import tqdm

# Add the project root to the path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Import the helper functions
from src.image import read_image
from src.validation import validate_model
from src.model import load_model, read_model_config
from src.google_drive import upload_bytes_to_folder, get_file_by_uuid, get_drive_service

# Setup logger
logger = logging.getLogger(__name__)
load_dotenv()

# Define common types
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, np.ndarray]

# Global variables for workers
worker_drive_service = None
worker_folder_id = None
worker_face_detector = None
worker_embedder = None

def worker_init(folder_id, face_model_path, embed_model_path):
    """
    Initialize a new worker
    """
    global worker_drive_service, worker_folder_id, worker_face_detector, worker_embedder
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    worker_drive_service = get_drive_service()
    worker_folder_id = folder_id
    
    from src.validation import validate_model
    from src.model import load_model, read_model_config
    
    if face_model_path and embed_model_path:
        validate_model(face_model_path)
        face_config = read_model_config(face_model_path)
        worker_face_detector = load_model(face_config)
        
        validate_model(embed_model_path)
        embed_config = read_model_config(embed_model_path)
        worker_embedder = load_model(embed_config)

def process_single_image_task(args):
    """
    Given one image, meta pair -> extract face, compute embeddings -> upload to drive -> return upload metadata
    """
    img_path, meta = args
    global worker_drive_service, worker_folder_id, worker_face_detector, worker_embedder
    
    try:
        img = read_image(img_path, return_numpy=False)
        face = worker_face_detector.detect(img)
        if face is None:
            return False, f"No face detected in {img_path}"
            
        # Compute embedding (Already normalized by DeepFaceEmbedder)
        embedding = worker_embedder.compute_embeddings(face)
        
        new_uuid = str(uuid.uuid4())
        with io.BytesIO() as output:
            img.save(output, format="JPEG")
            image_bytes = output.getvalue()

        drive_file_id = upload_bytes_to_folder(
            service=worker_drive_service,
            folder_id=worker_folder_id,
            file_name=f"{new_uuid}.jpg",
            file_bytes=image_bytes,
            mime_type="image/jpeg",
            uuid_str=new_uuid
        )

        if not drive_file_id:
            return False, f"Drive upload failed for {img_path}"

        return True, {
            "source": meta.get('name', 'Unknown'),
            "drive_file_id": drive_file_id,
            "embedding": embedding.tolist()
        }

    except Exception as e:
        return False, f"Error processing {img_path}: {str(e)}"


class AutoFaissIndex:
    """
    Manages a FAISS index backed by PostgreSQL (Supabase).
    Uses IndexFlatIP (Inner Product) for Cosine Similarity.
    """

    def __init__(self,
                 face_detect_model: Optional[str] = None,
                 embeddings_model: Optional[str] = None,
                 drive_service: Optional[Any] = None,
                 drive_folder_id: Optional[str] = None):
        
        self.db_dsn = os.getenv("DB_CONNECTION_STRING")
        if not self.db_dsn:
            raise ValueError("DB_CONNECTION_STRING not found in environment variables.")
        
        self.drive_service = drive_service
        self.drive_folder_id = drive_folder_id

        self.face_detect_model_path = Path(PROJECT_ROOT) / face_detect_model if face_detect_model else None
        self.embeddings_model_path = Path(PROJECT_ROOT) / embeddings_model if embeddings_model else None
        
        self.face_detector = None
        self.embedder = None
        self.face_detect_model_config = None
        self.embeddings_model_config = None

        self._init_db_connection()
        self._load_or_create_index()

    def _init_db_connection(self):
        """
        Connect to the supabase database
        """
        try:
            self.conn = psycopg2.connect(self.db_dsn)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL.")
        except Exception as e:
            logger.exception("Failed to connect to database: %s", e)
            raise

    def _load_or_create_index(self):
        """
        Load existing tables or create new
        """
        cursor = self.conn.cursor()
        
        # Get the row that has the vector dimension record
        cursor.execute("SELECT value FROM index_metadata WHERE key='dim'")
        row = cursor.fetchone()
        
        if row:
            self.dim = int(row[0])
            logger.info(f"Configuration loaded: Dimension={self.dim}")
        else:
            logger.error("Vector dimension under key 'dim' is not found. Can't initialize FAISS without it")
            raise KeyError

        # Create a FAISS Index
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

        logger.info("Downloading vectors from Postgres...")
        cursor.execute("SELECT id, embedding FROM face_data")
        rows = cursor.fetchall()

        # 0 -> id, 1 -> embedding
        if rows:
            # Convert ids to a numpy array for populating the FAISS
            ids = np.array([row[0] for row in rows], dtype=np.int64)
            
            embeddings_list = []
            for i, row in rows:
                emb_raw = row[1]
                if isinstance(emb_raw, str):
                    emb_list = json.loads(emb_raw)
                elif isinstance(emb_raw, list):
                    emb_list = emb_raw
                
                # If the embedding is not a string or list, something is wrong. Skip that row
                else:
                    logger.warning(
                        "Unknown embedding format: %s. Index = %s",
                        type(emb_raw),
                        i
                    )
                    continue

                embeddings_list.append(emb_list)

            embeddings = np.array(embeddings_list, dtype=np.float32)
            
            # For safety, normalize the embeddings
            # Although the embeddings class has to do that
            faiss.normalize_L2(embeddings)
            
            if len(ids) > 0:
                self.index.add_with_ids(embeddings, ids)
                logger.info("Rebuilt FAISS index with %s vectors.", self.index.ntotal)
            else:
                logger.info("No valid vectors found to add.")
        else:
            logger.info("Database is empty. Initialized empty index.")

        cursor.close()

    def process_and_upload_batch(self,
                                 image_paths: List[str],
                                 metadata_list: List[Dict],
                                 max_workers: int = 4):
        
        if not self.drive_folder_id:
            raise ValueError("Drive folder ID not configured.")

        logger.info("Starting Multiprocess Batch Upload (%s workers)...", max_workers)
        
        tasks = list(zip(image_paths, metadata_list))
        cursor = self.conn.cursor()
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(self.drive_folder_id, self.face_detect_model_path, self.embeddings_model_path)
        ) as executor:
            
            # Define the tasks and the function that handles those tasks
            future_to_file = {
                executor.submit(process_single_image_task, task): task[0] 
                for task in tasks
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(tasks), desc="Processing"):
                img_path = future_to_file[future]
                try:
                    success, result = future.result()
                    if success:
                        cursor.execute("""
                            INSERT INTO face_data (source, drive_file_id, embedding)
                            VALUES (%s, %s, %s)
                            RETURNING id;
                        """, (result['source'], result['drive_file_id'], result['embedding']))
                        new_db_id = cursor.fetchone()[0]
                        
                        # Update RAM
                        vec_np = np.array([result['embedding']], dtype=np.float32)

                        id_np = np.array([new_db_id], dtype=np.int64)
                        self.index.add_with_ids(vec_np, id_np)
                    else:
                        logger.warning("Worker failed for %s: %s", img_path, result)
                except Exception as e:
                    logger.exception("Exception collecting result for %s: %s", img_path, e)

        cursor.close()
        logger.info("Batch processing complete.")

    def search_image(self, image: ImageInput, k: int = 5):
        """
        Search an image in the index
        """
        self._ensure_models_loaded()
        
        if isinstance(image, (str, Path)):
            image_arr = read_image(image)
        else:
            image_arr = image
        
        face = self.face_detector.detect(image_arr)
        if face is None:
            logger.warning("No face detected in query image.")
            return [], [], []
        
        query_vector = self.embedder.compute_embeddings(face)
        
        # FAISS Search
        query_np = np.array([query_vector], dtype=np.float32)
        
        distances, indices = self.index.search(query_np, k)
        
        found_ids = [idx for idx in indices[0] if idx != -1]
        results = []
        
        if found_ids:
            py_ids = tuple(int(x) for x in found_ids)
            cursor = self.conn.cursor()

            # Get the metadata from the postgresql matching the similiar IDs fetched from FAISS
            query = "SELECT id, source, drive_file_id FROM face_data WHERE id IN %s"
            cursor.execute(query, (py_ids,))
            rows = cursor.fetchall()
            cursor.close()
            
            row_map = {row[0]: {'name': row[1], 'drive_id': row[2]} for row in rows}
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx in row_map:
                    data = row_map[idx]
                    results.append({
                        'id': idx,
                        'score': float(dist),
                        'name': data['name'],
                        'drive_link': f"https://drive.google.com/uc?id={data['drive_id']}",
                        'drive_id': data['drive_id']
                    })

        return distances, indices, results

    def add_image(self, image: ImageInput, metadata: Dict[str, Any]):
        """
        Upload an image to Drive and PostgreSQL
        """
        self._ensure_models_loaded()
        if not (self.drive_service and self.drive_folder_id):
             raise ValueError("Google Drive service not configured")

        if isinstance(image, (str, Path)):
            image_arr = read_image(image)
        else:
            image_arr = image

        face = self.face_detector.detect(image_arr)
        if face is None:
            raise ValueError("No face detected.")
        
        embedding = self.embedder.compute_embeddings(face)

        # Upload Drive
        new_uuid = str(uuid.uuid4())
        pil_image = Image.fromarray(image_arr)
        with io.BytesIO() as output:
            pil_image.save(output, format="JPEG")
            image_bytes = output.getvalue()
        
        drive_id = upload_bytes_to_folder(
            service=self.drive_service,
            folder_id=self.drive_folder_id,
            file_name=f"{new_uuid}.jpg",
            file_bytes=image_bytes,
            mime_type="image/jpeg",
            uuid_str=new_uuid
        )

        if not drive_id:
            raise ValueError("Drive upload failed")

        # Insert DB
        source_name = metadata.get('name', 'Unknown')
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO face_data (source, drive_file_id, embedding)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (source_name, drive_id, embedding.tolist()))
            
            new_id = cursor.fetchone()[0]
            
            # Update RAM
            vec_np = np.array([embedding], dtype=np.float32)
            id_np = np.array([new_id], dtype=np.int64)
            self.index.add_with_ids(vec_np, id_np)
            
            logger.info("Added face: %s (ID: %s)", source_name, new_id)
            return new_uuid
        except Exception as e:
            logger.error("DB Insert failed: %s", e)
            raise e
        finally:
            cursor.close()

    def delete_image_by_uuid(self, uuid_str: str) -> bool:
        """
        Delete an image by uuid
        """
        if not self.drive_service:
            return False

        file_info = get_file_by_uuid(self.drive_service, uuid_str)
        if not file_info:
            logger.warning("UUID %s not found in Drive", uuid_str)
            return False
        
        drive_file_id = file_info.get('id')

        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT id FROM face_data WHERE drive_file_id = %s", (drive_file_id,))
            row = cursor.fetchone()
            
            if not row:
                logger.warning("Drive ID %s not found in Database", drive_file_id)
                return False
            
            db_id = row[0]

            self.index.remove_ids(np.array([db_id], dtype=np.int64))
            cursor.execute("DELETE FROM face_data WHERE id = %s", (db_id,))
            self.drive_service.files().delete(fileId=drive_file_id).execute()
            
            logger.info("Deleted face UUID: %s (DB ID: %s)", uuid_str, db_id)
            return True

        except Exception as e:
            logger.exception("Error deleting %s: %s", uuid_str, e)
            return False
        finally:
            cursor.close()

    def _ensure_models_loaded(self):
        """
        Ensure the models are loaded
        """
        if self.face_detector and self.embedder:
            return
        
        logger.info("Loading models...")
        if not self.face_detect_model_path or not self.embeddings_model_path:
             raise ValueError("Model paths not provided in init.")

        if self.face_detect_model_config is None:
            validate_model(self.face_detect_model_path)
            self.face_detect_model_config = read_model_config(self.face_detect_model_path)
            
        if self.embeddings_model_config is None:
            validate_model(self.embeddings_model_path)
            self.embeddings_model_config = read_model_config(self.embeddings_model_path)

        self.face_detector = load_model(self.face_detect_model_config)
        self.embedder = load_model(self.embeddings_model_config)
        logger.info("Models loaded.")