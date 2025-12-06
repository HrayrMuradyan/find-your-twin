from pathlib import Path
import numpy as np
import logging
import httpx
from typing import Any, Dict, List, Optional, Union
import uuid
import io
import os
from dotenv import load_dotenv
from PIL import Image

# Add the project root to the path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Import helpers
from src.image import read_image
from src.validation import validate_model
from src.model import load_model, read_model_config
from src.google_drive import upload_bytes_to_folder, get_file_by_uuid

logger = logging.getLogger(__name__)
load_dotenv()

# Types
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, np.ndarray]

class DatabaseServiceClient:
    """
    Senior Pattern: Facade / Gateway
    1. Performs compute-heavy ML tasks locally (Edge/Client side).
    2. Delegates data persistence and retrieval to the Backend Service.
    """

    def __init__(self,
                 face_detect_model: Optional[str] = None,
                 embeddings_model: Optional[str] = None,
                 drive_service: Optional[Any] = None,
                 drive_folder_id: Optional[str] = None):
        
        # Service Configuration
        hf_url = os.getenv("DATABASE_SERVICE_URL")
        db_port = os.getenv("DATABASE_PORT")

        if not hf_url and not db_port:
            raise ValueError(
                "You must set either DATABASE_SERVICE_URL (HF) or DATABASE_PORT (localhost:{PORT})."
            )

        # Either locally or HF
        self.db_service_url = hf_url or f"http://host.docker.internal:{db_port}"
        
        # Drive Config
        self.drive_service = drive_service
        self.drive_folder_id = drive_folder_id

        # Models
        self.face_detect_model_path = Path(PROJECT_ROOT) / face_detect_model if face_detect_model else None
        self.embeddings_model_path = Path(PROJECT_ROOT) / embeddings_model if embeddings_model else None
        
        self.face_detector = None
        self.embedder = None
        self.face_detect_model_config = None
        self.embeddings_model_config = None

        self.http_client = httpx.AsyncClient(
            base_url=self.db_service_url,
            timeout=30.0,
            headers={}
        )

    async def get_index_count(self) -> Dict:
        try:
            resp = await self.http_client.get("/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"count": 0}

    async def search_image(self, image: ImageInput, k: int = 5):
        self._ensure_models_loaded()
        
        if isinstance(image, (str, Path)):
            image_arr = read_image(image)
        else:
            image_arr = image
        
        # Face Detect + Embeddings
        face = self.face_detector.detect(image_arr)
        if face is None:
            raise ValueError("No face detected.")
        
        query_vector = self.embedder.compute_embeddings(face)
        
        # Search vor similars through remote FAISS
        try:
            response = await self.http_client.post("/search", json={
                "vector": query_vector.tolist(),
                "k": k
            })
            response.raise_for_status()
            return response.json().get("results", [])
        except httpx.HTTPError as e:
            logger.error(f"Search Service Error: {e}")
            raise e

    async def add_image(self,
                        image: ImageInput,
                        metadata: Dict[str, Any]):

        self._ensure_models_loaded()
        if not (self.drive_service and self.drive_folder_id):
             raise ValueError("Google Drive not configured")

        image_arr = read_image(image) if isinstance(image, (str, Path)) else image
        
        # Detection + Embedding
        face = self.face_detector.detect(image_arr)
        if face is None:
            raise ValueError("No face detected.")
        embedding = self.embedder.compute_embeddings(face)

        # Drive Upload
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

        # We send all necessary data to the backend. 
        # The Backend handles SQL insert AND Faiss update atomically.
        source_name = metadata.get('name', 'Unknown')
        
        try:
            await self.http_client.post("/add", json={
                "vector": embedding.tolist(),
                "drive_file_id": drive_id,  
                "source": source_name,
                "metadata": {
                    "name": source_name,
                    "drive_id": drive_id,
                    "drive_link": f"https://drive.google.com/uc?id={drive_id}"
                }
            })
            logger.info(f"Persisted face: {source_name}")
            return new_uuid
        except Exception as e:
            logger.exception(f"Failed to persist data remotely: {e}")
            raise e

    async def delete_image_by_uuid(self, uuid_str: str) -> bool:
        if not self.drive_service: return False

        file_info = get_file_by_uuid(self.drive_service, uuid_str)
        if not file_info: return False
        drive_file_id = file_info.get('id')

        # Delete from Drive
        try:
            self.drive_service.files().delete(fileId=drive_file_id).execute()
        except Exception as e:
            logger.warning(f"Drive delete failed: {e}")

        # Delete from Database Service (Handles SQL + FAISS)
        try:
            await self.http_client.post("/delete", json={"drive_file_id": drive_file_id})
            return True
        except Exception as e:
            logger.error(f"Remote delete failed: {e}")
            return False

    def _ensure_models_loaded(self):
        if self.face_detector and self.embedder: return
        logger.info("Loading Models...")
        
        if self.face_detect_model_config is None:
            validate_model(self.face_detect_model_path)
            self.face_detect_model_config = read_model_config(self.face_detect_model_path)
        if self.embeddings_model_config is None:
            validate_model(self.embeddings_model_path)
            self.embeddings_model_config = read_model_config(self.embeddings_model_path)

        self.face_detector = load_model(self.face_detect_model_config)
        self.embedder = load_model(self.embeddings_model_config)