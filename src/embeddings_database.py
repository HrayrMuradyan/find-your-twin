from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import io
import gc 
from PIL import Image

import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from tqdm import tqdm
from src.image import read_image
from src.validation import validate_model
from src.model import load_model, read_model_config
from src.file import read_json, save_json
from src.google_drive import upload_bytes_to_folder

# Setup logger
logger = logging.getLogger(__name__)

# Define common types
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, np.ndarray]
Embedding = Union[np.ndarray, List[float]]

class AutoFaissIndex:
    """
    Manages a FAISS index for face embeddings.
    Updated with pipeline methods for batch processing.
    """

    def __init__(self,
                 index_path: ImagePath = "embeddings_store",
                 face_detect_model: Optional[str] = None,
                 embeddings_model: Optional[str] = None,
                 drive_service: Optional[Any] = None,
                 drive_folder_id: Optional[str] = None):
        
        if not isinstance(index_path, (str, Path)):
            raise TypeError(f"index_path must be a string or Path. Got: {type(index_path)}")
            
        # Store Google Drive info
        self.drive_service = drive_service
        self.drive_folder_id = drive_folder_id

        # Setup the database files' paths
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "faiss_index.faiss"
        self.emb_metadata_file = self.index_path / "emb_metadata.parquet"
        self.index_metadata_file = self.index_path / "metadata.json"

        # Check if all necessary database files exist
        database_necessary_files = [
            self.index_file,
            self.emb_metadata_file,
            self.index_metadata_file
        ]
        database_files_exist = [f.exists() for f in database_necessary_files]

        # If all exist, load the existing database
        # If not, delete partial files and start from scratch
        if all(database_files_exist):
            logger.info(
                "All database files exist at %s, loading...",
                self.index_path.relative_to(PROJECT_ROOT)
            )
            self._load_existing()

            self.face_detect_model_config = None
            self.embeddings_model_config = None
            self.face_detector = None
            self.embedder = None
        else:
            if any(database_files_exist):
                for f in database_necessary_files:
                    if f.exists(): f.unlink()

            if not (face_detect_model and embeddings_model):
                raise ValueError("Both face_detect_model and embeddings_model are required for new index.")

            self.face_detect_model = Path(PROJECT_ROOT) / face_detect_model
            self.embeddings_model = Path(PROJECT_ROOT) / embeddings_model
            self.index_type = "flat"  
            self.face_detector = None
            self.embedder = None

            # Validate and load model configs
            validate_model(self.face_detect_model)
            self.face_detect_model_config = read_model_config(self.face_detect_model)
            validate_model(self.embeddings_model)
            self.embeddings_model_config = read_model_config(self.embeddings_model)

            self.dim = self.embeddings_model_config['parameters']['dim']
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            
            self.emb_metadata = pd.DataFrame({
                "id": pd.Series(dtype="int"),
                "gdrive_id": pd.Series(dtype="string")
            })
            self._save()
            logging.info("New FAISS Index is created at: '%s'", self.index_path)
            logging.info("FAISS Index successfuly loaded")

    # -------------------------------------------------------------------------
    #  PHASE 1: PROCESS LOCAL IMAGES 
    # -------------------------------------------------------------------------
    def process_local_to_parquet(self,
                                 image_paths: List[str],
                                 metadata_list: List[Dict],
                                 output_file: Path,
                                 save_threshold: int = 500):
        """
        Detects faces and computes embeddings locally. 
        Saves results (UUID + Embedding) to a Parquet file. 
        """
        self._ensure_models_loaded()
        
        data_records = []
        
        logger.info("PHASE 1: Processing %s images locally...", len(image_paths))
        
        for i, img_path in enumerate(tqdm(image_paths, desc="PHASE 1 (Compute)")):
            try:
                # Read the image and detect faces
                img_arr = read_image(img_path)
                faces = self.face_detector.detect(img_arr)
                
                # Delete the image and free memory
                del img_arr 
                
                if not faces:
                    logging.warning(
                        "No face found for the image at path: '%s', skipping...",
                        img_path
                    )
                    continue
                    
                # Take the first face and calculate the embeddings
                embedding = self.embedder.compute_embeddings(faces[0])
                
                # Assign a new UUID
                file_uuid = str(uuid.uuid4())
                
                # Store Data
                # Convert embedding to list to store in Parquet/JSON safely
                record = {
                    "uuid": file_uuid,
                    "local_path": str(img_path),
                    "embedding": embedding.tolist(), 
                    **metadata_list[i]
                }

                data_records.append(record)

            except Exception as e:
                logger.warning("Failed to process %s: %s", img_path, e)
                continue
            
            # Save when number of records is higher than save threshold 
            if len(data_records) >= save_threshold:
                self._append_to_parquet(data_records, output_file)
                data_records = []
                gc.collect()

        # Save the last batch
        if data_records:
            self._append_to_parquet(data_records, output_file)
            
        logger.info("PHASE 1 Complete. Intermediate data saved to %s", str(output_file))

    def _append_to_parquet(self,
                           data: List[Dict],
                           filename: Path):
        """Helper to append data to parquet file safely."""

        df = pd.DataFrame(data)
        if filename.is_file():
            try:
                existing_df = pd.read_parquet(filename)
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined.to_parquet(filename, engine='fastparquet', compression='snappy')
            except Exception as e:
                logger.error("Error appending parquet: %s. Saving separate chunk.", e)
                # Fallback if concatenation or reading fails
                df.to_parquet(f"{filename}_{uuid.uuid4()}.parquet")
        else:
            df.to_parquet(filename, engine='fastparquet', compression='snappy')

    # -------------------------------------------------------------------------
    #  PHASE 3: BUILD INDEX FROM PROCESSED DATA
    # -------------------------------------------------------------------------
    def build_index_from_parquet(self, parquet_file: Path):
        """
        Loads the processed parquet (which now contains 'gdrive_id' from PHASE 2) 
        and adds vectors to FAISS.
        """
        if not parquet_file.is_file():
            raise FileNotFoundError(parquet_file)
            
        logger.info("PHASE 3: Loading processed data...")
        df = pd.read_parquet(parquet_file)
        
        # Filter rows that successfully uploaded
        if 'gdrive_id' not in df.columns:
            raise ValueError("Parquet file missing 'gdrive_id'. Did PHASE 2 finish?")
            
        initial_len = len(df)
        df = df.dropna(subset=['gdrive_id'])
        logger.info(
            "Found %s ready items (out of %s processed).",
            len(df),
            initial_len
        )
        
        # Convert the embeddings to a numpy array of shape (n, self.dim)
        embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
        
        # Prepare Metadata (Drop internal columns)
        metadata_list = []
        keys_to_exclude = ['embedding', 'local_path', 'uuid']
        
        # Convert DataFrame to list of dicts for iteration
        records = df.to_dict(orient='records')
        
        for row in records:
            meta = {k: v for k, v in row.items() if k not in keys_to_exclude}
            metadata_list.append(meta)
            
        logger.info("PHASE 3: Adding %s vectors to FAISS...", len(embeddings))
        self.add(embeddings, metadata_list)

    # -------------------------------------------------------------------------
    #  STANDARD METHODS (Preserved)
    # -------------------------------------------------------------------------
    def add(self,
            embeddings: List[Embedding] | np.ndarray,
            metadata: List[Dict[str, Any]]):
        
        if not isinstance(embeddings, (list, np.ndarray)):
            raise TypeError(f"embeddings must be a list or np.ndarray.")
        
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # If it's a single embeddings convert (self.dim,) to (1, self.dim)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        n_new = embeddings_np.shape[0]
        if n_new == 0:
            return

        if self.emb_metadata.empty:
            start_id = 0
        else:
            start_id = int(self.emb_metadata['id'].max()) + 1

        ids = list(range(start_id, start_id + n_new))
        ids_array = np.array(ids, dtype=np.int64)

        self.index.add_with_ids(embeddings_np, ids_array)
        
        df = pd.DataFrame(metadata, copy=False)
        df.insert(0, "id", ids_array.astype("int32"))

        self.emb_metadata = pd.concat([self.emb_metadata, df], ignore_index=True)

        self._save()

    def search_query(self,
                     query: Embedding | np.ndarray,
                     k: int = 5,
                     return_metadata: bool = True):
        
        query_np = np.array(query, dtype=np.float32)

        # Convert the query to (1, self.dim) if it's a single query
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        distances, ids = self.index.search(query_np, k)
        results = None

        if return_metadata:
            results = []
            for row_ids in ids:
                valid_ids = [idx for idx in row_ids if idx != -1]
                if valid_ids:
                    results.append(
                        self.emb_metadata[self.emb_metadata['id'].isin(valid_ids)]
                    )
                else:
                    results.append(pd.DataFrame(columns=self.emb_metadata.columns))
        return distances, ids, results

    def search_image(self,
                     image: ImageInput,
                     k: int = 5,
                     return_metadata: bool = True,
                     return_face: bool = False):
        
        self._ensure_models_loaded()
        if isinstance(image, (str, Path)): image_arr = read_image(image)
        else: image_arr = image
        
        faces = self.face_detector.detect(image_arr)
        if not faces:
            raise ValueError("No face detected.")
        if len(faces) > 1:
            raise ValueError("Multiple faces detected.") 
        
        face = faces[0]
        query = self.embedder.compute_embeddings(face)
        distances, ids, results = self.search_query(query, k, return_metadata)
        
        if return_face:
            return distances, ids, results, face
        
        return distances, ids, results, None

    def add_image(self, image: ImageInput, metadata: Dict[str, Any]):
        """Single image upload helper."""
        self._ensure_models_loaded()

        if not (self.drive_service and self.drive_folder_id):
             raise ValueError("Google Drive service not configured.")
        
        if isinstance(image, (str, Path)):
            image_arr = read_image(image)
        else:
            image_arr = image

        new_uuid = str(uuid.uuid4())
        try:
            pil_image = Image.fromarray(image_arr)
            with io.BytesIO() as output:
                pil_image.save(output, format="JPEG")
                image_bytes = output.getvalue()
            
            uploaded_file_id = upload_bytes_to_folder(
                service=self.drive_service,
                folder_id=self.drive_folder_id,
                file_name=f"{new_uuid}.jpg",
                file_bytes=image_bytes,
                mime_type="image/jpeg",
                uuid_str=new_uuid
            )
            if not uploaded_file_id:
                raise ValueError("Failed to upload.")
        except Exception:
            logger.exception("Upload failed")
            raise

        faces = self.face_detector.detect(image_arr)
        if not faces:
            return
        
        embeddings_batch = []
        metadata_batch = []
        for face in faces:
            embeddings_batch.append(self.embedder.compute_embeddings(face))
            new_metadata = metadata.copy()
            new_metadata['gdrive_id'] = uploaded_file_id
            metadata_batch.append(new_metadata)
        
        self.add(embeddings=embeddings_batch, metadata=metadata_batch)

    def _save(self):
        faiss.write_index(self.index, str(self.index_file))
        for col in self.emb_metadata.select_dtypes(include=["object"]).columns:
            self.emb_metadata[col] = self.emb_metadata[col].astype(str)

        self.emb_metadata.to_parquet(self.emb_metadata_file, index=False, engine="fastparquet")
        
        meta = {
            "index_type": self.index_type, "dim": self.dim,
            "models": {
                "face_detect_model": str(self.face_detect_model),
                "embeddings_model": str(self.embeddings_model)
            },
            "vector_count": self.index.ntotal,
        }
        save_json(meta, self.index_metadata_file)

    def _load_existing(self):
        self.index = faiss.read_index(str(self.index_file))
        self.emb_metadata = pd.read_parquet(self.emb_metadata_file)
        index_metadata = read_json(self.index_metadata_file)
        self.dim = index_metadata["dim"]
        self.index_type = index_metadata["index_type"]
        self.face_detect_model = Path(index_metadata["models"]["face_detect_model"])
        self.embeddings_model = Path(index_metadata["models"]["embeddings_model"])

    def _ensure_models_loaded(self):
        if self.face_detector and self.embedder: return
        
        logger.info("Loading models...")
        if self.face_detect_model_config is None:
            validate_model(self.face_detect_model)
            self.face_detect_model_config = read_model_config(self.face_detect_model)
            
        if self.embeddings_model_config is None:
            validate_model(self.embeddings_model)
            self.embeddings_model_config = read_model_config(self.embeddings_model)

        self.face_detector = load_model(self.face_detect_model_config)
        self.embedder = load_model(self.embeddings_model_config)
        logger.info("Models loaded.")