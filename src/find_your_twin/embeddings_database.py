from pathlib import Path
import numpy as np
import logging
import httpx
import asyncio
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
    Main class of the inference service.
    """

    def __init__(self,
                 face_detect_model: Optional[str] = None,
                 embeddings_model: Optional[str] = None,
                 drive_service: Optional[Any] = None,
                 drive_folder_id: Optional[str] = None):
        
        # --- Get the database service URL ---
        # inference service communicates with database service via that URL
        
        hf_url = os.getenv("DATABASE_SERVICE_URL")
        db_port = os.getenv("DATABASE_PORT")

        if not hf_url and not db_port:
            raise ValueError(
                "You must set either DATABASE_SERVICE_URL (HF) or DATABASE_PORT (localhost:{PORT})."
            )

        # Either locally or HF
        self.db_service_url = hf_url or f"http://host.docker.internal:{db_port}"

        # Connect to the service
        self.http_client = httpx.AsyncClient(
            base_url=self.db_service_url,
            timeout=30.0,
            headers={}
        )
        
        # --------------------------------------
        
        # Google Drive Config
        self.drive_service = drive_service
        self.drive_folder_id = drive_folder_id

        # Models
        self.face_detect_model_path = Path(PROJECT_ROOT) / face_detect_model if face_detect_model else None
        self.embeddings_model_path = Path(PROJECT_ROOT) / embeddings_model if embeddings_model else None
        
        self.face_detector = None
        self.embedder = None
        self.face_detect_model_config = None
        self.embeddings_model_config = None


    async def close(self):
        """Cleanup resources on app shutdown."""
        await self.http_client.aclose()

    async def get_index_count(self) -> Dict:
        """Get the number of observations in the FAISS db with Retry Logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await self.http_client.get("/health")
                resp.raise_for_status()

                # Returns: {"status": "ok", "count": N of observations}
                return resp.json()
            
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
                # If the connection was stale, this block catches it and retries
                logger.warning(f"Stale connection on health check (Attempt {attempt+1}/{max_retries}). Retrying...")
                if attempt == max_retries - 1:
                    logger.error("Health check failed after retries.")
                    return {"count": 0}
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.exception("Health check failed: %s", e)
                return {"count": 0}

    async def search_image(self, image: ImageInput, k: int = 5):
        """Search closest matches for an image"""
        self._ensure_models_loaded()
        
        def _process_inference():
            """Given an image detects a face and computes embeddings"""
            if isinstance(image, (str, Path)):
                image_arr = read_image(image)
            else:
                image_arr = image
            
            face = self.face_detector.detect(image_arr)
            if face is None:
                raise ValueError("No face detected.")
            
            return self.embedder.compute_embeddings(face)

        # Run the inference in threadpool
        # So that it doesn't block other requests
        try:
            query_vector = await run_in_threadpool(_process_inference)
        except ValueError as e:
            raise e
        except Exception as e:
            logger.exception("Inference failed with the following error message: %s", e)
            raise ValueError("Failed to process image") from e
        
        # Given the embeddings vector, search for closest matches
        try:
            response = await self.http_client.post("/search", json={
                "vector": query_vector.tolist(),
                "k": k
            })
            response.raise_for_status()

            # Results is a list of closest matches returned from FAISS 
            return response.json().get("results", [])
        
        except httpx.HTTPError as e:
            logger.exception("Search Service Error: %s", e)
            raise e

    async def add_image(self,
                        image: ImageInput,
                        metadata: Dict[str, Any]):
        """Adds an image to the database"""

        self._ensure_models_loaded()

        if not (self.drive_service and self.drive_folder_id):
             raise ValueError("Google Drive not configured")

        def _prepare_data():
            """Preprocesses the image to embeddings (for FAISS) and bytes (for Drive)"""
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

        # Run in threadpool so that it doesn't block other requests
        embedding, image_bytes = await run_in_threadpool(_prepare_data)

        # Generate a new uuid
        new_uuid = str(uuid.uuid4())
        
        # Upload to Drive
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

        # Get the name of the data from the metadata
        # Can be found in the data source json file or if it's a user upload, then it's user
        source_name = metadata.get('name', 'Unknown')
        
        # Adds the data to PostgreSQL and FAISS
        try:
            await self.http_client.post("/add", json={
                "vector": embedding.tolist(),
                "drive_file_id": drive_id,  
                "source": source_name,
                "metadata": {
                    "name": source_name,
                    "filename": new_uuid,
                    "drive_id": drive_id,
                    "drive_link": f"https://drive.google.com/uc?id={drive_id}"
                }
            })

            logger.info("Image successfully saved: %s (%s)", source_name, new_uuid)
            return new_uuid

        except Exception as e:
            logger.exception("Failed to persist data remotely: %s", e)
            logger.info("Rollback Drive Upload %s...", drive_id)

            def _rollback_sync():
                try:
                    self.drive_service.files().delete(fileId=drive_id).execute()
                    logger.info("Rollback successful: Deleted orphaned file %s", drive_id)
                except Exception as cleanup_error:
                    logger.critical(
                        "Could not delete %s during rollback. Manual cleanup required. Error: %s",
                        drive_id, cleanup_error
                    )

            await run_in_threadpool(_rollback_sync)
            
            raise e

    async def delete_image_by_uuid(self, uuid_str: str) -> bool:
        """
        Delete from Access Layer (DB) first, then Asset Layer (Drive).
        """
        # Can't delete if drive_service is None
        if not self.drive_service:
            return False

        # Get the file metadata from Drive
        def _lookup_file():
            return get_file_by_uuid(self.drive_service, uuid_str)
        
        file_info = await run_in_threadpool(_lookup_file)
        
        if not file_info:
            logger.warning("File %s not found in Drive. Assuming already deleted.", uuid_str)
            return False
        
        drive_file_id = file_info.get('id')

        # Delete from PostgreSQL and FAISS first
        # This will ensure there are no zombie files (metadata exists, but image no)
        try:
            await self.http_client.post("/delete", json={"drive_file_id": drive_file_id})
            logger.info("Database record deleted for %s", drive_file_id)
        except Exception as e:
            logger.exception("Remote delete failed: %s. Aborting Drive deletion to prevent broken links.", e)
            return False

        # If this fails, we log an orphan but return True (User intent is satisfied).
        def _delete_sync():
            try:
                self.drive_service.files().delete(fileId=drive_file_id).execute()
                return True
            except Exception as e:
                logger.critical("Database deleted, but Drive delete failed for %s. Error: %s", drive_file_id, e)
                return True 

        return await run_in_threadpool(_delete_sync)
        

    def _ensure_models_loaded(self):
        """
        Lazy loading of models.
        """
        if self.face_detector and self.embedder:
            return
        
        logger.info("Loading Models...")

        # Loads configs first
        if self.face_detect_model_config is None:
            validate_model(self.face_detect_model_path)
            self.face_detect_model_config = read_model_config(self.face_detect_model_path)
            
        if self.embeddings_model_config is None:
            validate_model(self.embeddings_model_path)
            self.embeddings_model_config = read_model_config(self.embeddings_model_path)

        # Loads the models given the config
        self.face_detector = load_model(self.face_detect_model_config)
        self.embedder = load_model(self.embeddings_model_config)