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
from fastapi.concurrency import run_in_threadpool

# Import helpers
from find_your_twin.image import read_image
from find_your_twin.validation import validate_model
from find_your_twin.model import load_model, read_model_config
from find_your_twin.google_drive import upload_bytes_to_folder, get_file_by_uuid
from find_your_twin.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
load_dotenv()

# Types
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, np.ndarray]

class DatabaseServiceClient:
    """
    Senior Pattern: Facade / Gateway
    1. Performs compute-heavy ML tasks locally (Edge/Client side) without blocking the event loop.
    2. Delegates data persistence and retrieval to the Backend Service.
    3. Manages distributed transactions between Google Drive and Postgres/FAISS.
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

    async def close(self):
        """Cleanup resources on app shutdown."""
        await self.http_client.aclose()

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
        
        def _process_inference():
            if isinstance(image, (str, Path)):
                image_arr = read_image(image)
            else:
                image_arr = image
            
            face = self.face_detector.detect(image_arr)
            if face is None:
                raise ValueError("No face detected.")
            
            return self.embedder.compute_embeddings(face)

        try:
            query_vector = await run_in_threadpool(_process_inference)
        except ValueError as e:
            raise e
        except Exception as e:
            logger.exception("Inference failed with the following error message: %s", e)
            raise ValueError("Failed to process image") from e
        
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

        def _prepare_data():
            image_arr = read_image(image) if isinstance(image, (str, Path)) else image
            
            face = self.face_detector.detect(image_arr)
            if face is None:
                raise ValueError("No face detected.")
            
            emb = self.embedder.compute_embeddings(face)

            # Convert to bytes for upload
            pil_image = Image.fromarray(image_arr)
            with io.BytesIO() as output:
                pil_image.save(output, format="JPEG")
                img_bytes = output.getvalue()
            
            return emb, img_bytes

        embedding, image_bytes = await run_in_threadpool(_prepare_data)

        new_uuid = str(uuid.uuid4())
        
        def _upload_sync():
            return upload_bytes_to_folder(
                service=self.drive_service,
                folder_id=self.drive_folder_id,
                file_name=f"{new_uuid}.jpg",
                file_bytes=image_bytes,
                mime_type="image/jpeg",
                uuid_str=new_uuid
            )

        drive_id = await run_in_threadpool(_upload_sync)

        if not drive_id:
            raise ValueError("Drive upload failed")

        source_name = metadata.get('name', 'Unknown')
        
        try:
            # Atomic DB Insert (Postgres + FAISS via Remote Service)
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
            logger.info(f"Image successfully saved: {source_name} ({new_uuid})")
            return new_uuid

        except Exception as e:
            logger.exception(f"Failed to persist data remotely: {e}")
            logger.info(f"Initiating Compensating Transaction: Rollback Drive Upload {drive_id}...")

            def _rollback_sync():
                try:
                    self.drive_service.files().delete(fileId=drive_id).execute()
                    logger.info(f"Rollback successful: Deleted orphaned file {drive_id}")
                except Exception as cleanup_error:
                    logger.critical(f"Could not delete {drive_id} during rollback. Manual cleanup required. Error: {cleanup_error}")

            await run_in_threadpool(_rollback_sync)
            
            raise e

    async def delete_image_by_uuid(self, uuid_str: str) -> bool:
        """
        Senior Pattern: Delete from Access Layer (DB) first, then Asset Layer (Drive).
        """
        if not self.drive_service:
            return False

        def _lookup_file():
            return get_file_by_uuid(self.drive_service, uuid_str)
        
        file_info = await run_in_threadpool(_lookup_file)
        
        if not file_info:
            logger.warning("File %s not found in Drive. Assuming already deleted.", uuid_str)
            return False
        
        drive_file_id = file_info.get('id')

        # Delete from PostgreSQL and FAISS first
        # This ensures we never have "Zombie Files" (Files that exist but result in 404s).
        try:
            await self.http_client.post("/delete", json={"drive_file_id": drive_file_id})
            logger.info(f"Database record deleted for {drive_file_id}")
        except Exception as e:
            logger.error(f"Remote delete failed: {e}. Aborting Drive deletion to prevent broken links.")
            return False

        # If this fails, we log an orphan but return True (User intent satisfied).
        def _delete_sync():
            try:
                self.drive_service.files().delete(fileId=drive_file_id).execute()
                return True
            except Exception as e:
                logger.critical(f"Database deleted, but Drive delete failed for {drive_file_id}. Error: {e}")
                return True 

        return await run_in_threadpool(_delete_sync)
        

    def _ensure_models_loaded(self):
        """
        Lazy loading of models.
        """
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