from pathlib import Path
import numpy as np
import logging
import zmq
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

    If success:
        Returns True, {"source": ..., "drive_file_id": ..., "embedding": ...}
    
    If fail:
        Return False, Message
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


class InferenceClient:
    """
    Inference Service
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
        
        # Setup ZeroMQ
        logger.info("Connecting to the FAISS Search Service")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5555")

    def get_index_count(self) -> int:
        """
        Ask the Sidecar for the total number of vectors.
        """
        try:
            # We use a short timeout so the homepage doesn't hang if ZMQ is busy
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.socket.setsockopt(zmq.SNDTIMEO, 2000)
            
            self.socket.send_pyobj({"command": "health"})
            response = self.socket.recv_pyobj()
            
            # Reset timeouts to default (safe practice)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000) 
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)

            if response.get("status") == "ok":
                return response.get("count", 0)
            return 0
        except Exception as e:
            logger.warning(f"Could not get stats from Sidecar: {e}")
            return 0

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
                    # result is a dictionary with keys:
                    # "source", "drive_file_id", "embedding"
                    success, result = future.result()
                    if success:
                        cursor.execute("""
                            INSERT INTO face_data (source, drive_file_id, embedding)
                            VALUES (%s, %s, %s)
                            RETURNING id;
                        """, (result['source'], result['drive_file_id'], result['embedding']))

                        new_db_id = cursor.fetchone()[0]
                        
                        try:
                            self.socket.send_pyobj({
                                "command": "add",
                                "id": new_db_id,
                                "vector": result['embedding'],
                                "metadata": {
                                    "name": result['source'],
                                    "drive_id": result['drive_file_id'],
                                    "drive_link": f"https://drive.google.com/uc?id={result['drive_file_id']}"
                                }
                            })
                            self.socket.recv_pyobj()
                        except Exception as zmq_err:
                            logger.exception(
                                "Failed to update the FAISS Index for %s: %s",
                                img_path,
                                zmq_err
                            )
                    else:
                        logger.warning(
                            "Worker failed for %s: %s",
                            img_path,
                            result
                        )
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
        
        # Send to the FAISS service
        try:
            logging.info("Sending a search request to ZMQ...")
            self.socket.send_pyobj({
                "command": "search",
                "vector": query_vector.tolist(),
                "k": k
            })
            response = self.socket.recv_pyobj()

            if response["status"] == "ok":
                return response["results"]
            else:
                logger.error(
                    "FAISS Index error: %s",
                    response.get('msg')
                )
                return []
            
        except Exception as e:
            logger.exception("ZMQ Search Failed: %s", e)
            return []
    

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
        
        logging.info("Adding the uploaded image to the Drive folder...")
        drive_id = upload_bytes_to_folder(
            service=self.drive_service,
            folder_id=self.drive_folder_id,
            file_name=f"{new_uuid}.jpg",
            file_bytes=image_bytes,
            mime_type="image/jpeg",
            uuid_str=new_uuid
        )
        logging.info("Drive upload done")

        if not drive_id:
            raise ValueError("Drive upload failed")

        # Insert DB
        self._get_valid_connection

        source_name = metadata.get('name', 'Unknown')
        cursor = self.conn.cursor()
        try:
            logging.info("Inserting the new data point to PostgreSQL...")
            cursor.execute("""
                INSERT INTO face_data (source, drive_file_id, embedding)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (source_name, drive_id, embedding.tolist()))
            
            new_id = cursor.fetchone()[0]
            
            logging.info("A data point with ID=%s is added to the PostgreSQL", new_id)
            logging.info("Sending an add request to ZMQ...")
            # Update FAISS
            self.socket.send_pyobj({
                "command": "add",
                "id": new_id,
                "vector": embedding.tolist(),
                "metadata": {
                    "name": source_name,
                    "drive_id": drive_id,
                    "drive_link": f"https://drive.google.com/uc?id={drive_id}"
                }
            })

            self.socket.recv_pyobj()
            
            logger.info("Added face: %s (ID: %s)", source_name, new_id)
            return new_uuid
        except Exception as e:
            logger.exception("DB or Faiss Insert failed: %s", e)
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

        self._get_valid_connection
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT id FROM face_data WHERE drive_file_id = %s", (drive_file_id,))
            row = cursor.fetchone()
            
            if not row:
                logger.warning("Drive ID %s not found in Database", drive_file_id)
                return False
            
            db_id = row[0]
            
            # DB Delete
            cursor.execute("DELETE FROM face_data WHERE id = %s", (db_id,))

            # Drive Delete
            self.drive_service.files().delete(fileId=drive_file_id).execute()

            # FAISS Delete
            self.socket.send_pyobj({
                "command": "delete",
                "id": db_id
            })
            
            self.socket.recv_pyobj()
            
            logger.info("Deleted face UUID: %s (DB ID: %s)", uuid_str, db_id)
            return True

        except Exception as e:
            logger.exception("Error deleting %s: %s", uuid_str, e)
            return False
        finally:
            cursor.close()

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

    
    def _get_valid_connection(self):
        """
        Check if connection is alive. If not, reconnect.
        Returns a fresh cursor.
        """
        try:
            # Check if connection exists and is open (0 = open)
            if self.conn is None or self.conn.closed != 0:
                logger.warning("DB Connection was closed. Reconnecting...")
                self._init_db_connection()
            
            # Lightweight check: Try to create a cursor. 
            # Sometimes 'self.conn.closed' is 0 but the socket is dead.
            return self.conn.cursor()
        except Exception:
            # Force a hard reconnect
            logger.warning("DB Socket dead. Forcing hard reconnect...")
            self._init_db_connection()
            return self.conn.cursor()