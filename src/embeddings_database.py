from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import sys
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from tqdm import tqdm
from src.image import read_image
from src.validation import validate_model
from src.model import load_model, read_model_config
from src.file import read_json, save_json

# Setup logger
logger = logging.getLogger(__name__)

# Define common types
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, np.ndarray]
Embedding = Union[np.ndarray, List[float]]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AutoFaissIndex:
    """
    Manages a FAISS index for face embeddings, handling creation,
    loading, saving, and searching.

    This class provides a high-level API to:
    - Initialize a new FAISS index with specific models.
    - Load an existing index from disk.
    - Add new embeddings from images or raw vectors.
    - Search the index using query vectors or query images.
    - Batch-populate the index from a collection of images.
    """

    def __init__(self,
                 index_path: ImagePath = "embeddings_store",
                 face_detect_model: Optional[str] = None,
                 embeddings_model: Optional[str] = None):
        """
        Initializes or loads a FAISS index.

        If the files at `index_path` (index, metadata) exist,
        the index is loaded.
        
        If not, a new index is created, and `face_detect_model`
        and `embeddings_model` must be provided.

        Args:
            index_path: Directory to store/load the index files.
            face_detect_model: Name of the face detection model
                               (required for new index).
            embeddings_model: Name of the embeddings model
                              (required for new index).

        """
        if not isinstance(index_path, (str, Path)):
            raise TypeError(
                f"index_path must be a string or Path. Got: {type(index_path)}"
            )

        # ----------------------------
        # 1. Setup The Paths
        # ----------------------------

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "faiss_index.faiss"
        self.emb_metadata_file = self.index_path / "emb_metadata.parquet"
        self.index_metadata_file = self.index_path / "metadata.json"

        database_necessary_files = [
            self.index_file, self.emb_metadata_file, self.index_metadata_file
        ]
        database_files_exist = [f.exists() for f in database_necessary_files]

        # ------------------------------------------
        # 2. If there is an existing index, load it
        # ------------------------------------------

        if all(database_files_exist):
            logger.info(
                "All database files exist at %s, loading...",
                index_path.relative_to(PROJECT_ROOT)
            )

            try:
                self._load_existing()
                self.face_detect_model_config = None
                self.embeddings_model_config = None
                self.face_detector = None
                self.embedder = None
                logger.info("Successfully loaded existing index.")

            except Exception:
                logger.exception(
                    "Failed to load existing FAISS index at %s",
                    index_path.relative_to(PROJECT_ROOT)
                )
                raise

        # --------------------------------------------------
        # 2. If there is no existing index, create a new one
        # --------------------------------------------------
        else:

            # Check for partial files, if exist, delete them
            if any(database_files_exist):
                logger.warning(
                    "Partial database files detected at %s. "
                    "Deleting all and starting from scratch.",
                    index_path.relative_to(PROJECT_ROOT)
                )

                for f in database_necessary_files:
                    if f.exists():
                        try:
                            f.unlink()
                            logger.info("Deleted partial file: %s", f.relative_to(PROJECT_ROOT))
                        except Exception:
                            logger.exception("Failed to delete partial file: %s", f.relative_to(PROJECT_ROOT))
                            raise

            # Validate inputs for new index
            if not isinstance(face_detect_model, (str, type(None))):
                raise TypeError(
                    f"face_detect_model must be a string or None. Got: {type(face_detect_model)}"
                )
            if not isinstance(embeddings_model, (str, type(None))):
                raise TypeError(
                    f"embeddings_model must be a string or None. Got: {type(embeddings_model)}"
                )

            if not (face_detect_model and embeddings_model):
                logger.error(
                    "To initialize a new FAISS Index, both "
                    "face_detect_model and embeddings_model must be provided."
                )
                raise ValueError(
                    "Both face_detect_model and embeddings_model are required "
                    "to create a new index."
                )

            self.face_detect_model = Path(PROJECT_ROOT) / face_detect_model
            self.embeddings_model = Path(PROJECT_ROOT) / embeddings_model
            self.index_type = "flat"  

            # Initialize models as None (lazy load)
            self.face_detector = None
            self.embedder = None

            # Load the configs of the models
            try:
                validate_model(self.face_detect_model)
                self.face_detect_model_config = read_model_config(
                    self.face_detect_model
                )
                logger.info("Loaded face detection config: %s",
                            self.face_detect_model.relative_to(PROJECT_ROOT))
            except Exception:
                logger.exception(
                    "Failed to load face detection model config for %s",
                    self.face_detect_model.relative_to(PROJECT_ROOT)
                )
                raise

            try:
                validate_model(self.embeddings_model)
                self.embeddings_model_config = read_model_config(
                    self.embeddings_model
                )
                logger.info("Loaded embeddings model config: %s",
                            self.embeddings_model.relative_to(PROJECT_ROOT))
                
            except Exception:
                logger.exception(
                    "Failed to load embeddings model config for %s",
                    self.embeddings_model.relative_to(PROJECT_ROOT)
                )
                raise

            # Set self.dim from the config
            self.dim = self.embeddings_model_config['parameters']['dim']

            # Initialize the FAISS index
            try:
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
                logger.info(
                    "Created new FAISS IndexIDMap (IndexFlatIP) with dimension %d",
                    self.dim
                )
            except Exception:
                logger.exception(
                    "Failed to create FAISS IndexIDMap with dimension %d",
                    self.dim
                )
                raise

            # Initialize metadata
            self.emb_metadata = pd.DataFrame({
                "id": pd.Series(dtype="int"),
                "img_path": pd.Series(dtype="string")
            })
            logger.info("Initialized empty embedding metadata DataFrame.")
            
            # Initial save of empty index
            self._save()

    def add(self,
            embeddings: List[Embedding] | np.ndarray,
            metadata: List[Dict[str, Any]]):
        """
        Adds embeddings and their corresponding metadata to the index.

        Args:
            embeddings: A list of embedding vectors or a NumPy array.
            metadata: A list of metadata dictionaries. Must be the same
                      length as embeddings.

        Returns:
            Nothing
        
        """
        if not isinstance(embeddings, (list, np.ndarray)):
            raise TypeError(
                f"embeddings must be a list or np.ndarray. Got: {type(embeddings)}"
            )
        if not isinstance(metadata, list):
            raise TypeError(
                f"metadata must be a list of dicts. Got: {type(metadata)}"
            )

        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        n_new = embeddings_np.shape[0]
        if n_new != len(metadata):
            raise ValueError(
                f"Number of embeddings ({n_new}) does not match "
                f"number of metadata entries ({len(metadata)})."
            )
        
        if n_new == 0:
            logger.info("Add called with no embeddings. Skipping.")
            return

        # Generate new IDs
        if self.emb_metadata.empty:
            start_id = 0
        else:
            start_id = int(self.emb_metadata['id'].max()) + 1

        ids = list(range(start_id, start_id + n_new))
        ids_array = np.array(ids, dtype=np.int64)

        # Add vectors to FAISS index
        try:
            self.index.add_with_ids(embeddings_np, ids_array)
            logger.info("Added %d new vectors to FAISS index.", n_new)
        except Exception:
            logger.exception("Failed to add vectors to FAISS index.")
            raise

        # Add metadata to DataFrame
        try:
            df = pd.DataFrame(metadata, copy=False)
            df.insert(0, "id", ids_array.astype("int32"))
            self.emb_metadata = pd.concat(
                [self.emb_metadata, df], ignore_index=True
            )
            logger.info("Updated embedding metadata DataFrame.")
        except Exception:
            logger.exception("Failed to update metadata DataFrame.")
            raise

        # Save updated index and metadata
        self._save()

    def search_query(
        self,
        query: Embedding | np.ndarray,
        k: int = 5,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[pd.DataFrame]]]:
        """
        Searches the index for the k-nearest neighbors to the query vector(s).

        Args:
            query: A single query vector or a 2D array of query vectors.
            k: The number of neighbors to retrieve.
            return_metadata: Whether to fetch and return metadata for found IDs.

        Returns:
            A tuple containing:
            - distances (np.ndarray): Distances to the k-nearest neighbors.
            - ids (np.ndarray): FAISS IDs of the k-nearest neighbors.
            - results (Optional[List[pd.DataFrame]]): A list of DataFrames
              (one per query) containing metadata, or None.

        """
        if not isinstance(query, (list, np.ndarray)):
            raise TypeError(
                f"query must be a list or np.ndarray. Got: {type(query)}"
            )
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer. Got: {k}")
        if not isinstance(return_metadata, bool):
            raise TypeError(
                f"return_metadata must be a boolean. Got: {type(return_metadata)}"
            )

        query_np = np.array(query, dtype=np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        logger.info(
            "Searching index for %d query vector(s) with k=%d",
            query_np.shape[0], k
        )
        distances, ids = self.index.search(query_np, k)

        results = None
        if return_metadata:
            results = []
            for row_ids in ids:
                valid_ids = [idx for idx in row_ids if idx != -1]
                if valid_ids:
                    found = self.emb_metadata[
                        self.emb_metadata['id'].isin(valid_ids)
                    ]
                    results.append(found)
                else:

                    results.append(
                        pd.DataFrame(columns=self.emb_metadata.columns)
                    )
            logger.info("Fetched metadata for search results.")

        return distances, ids, results

    def search_image(
        self,
        image: ImageInput,
        k: int = 5,
        return_metadata: bool = True,
        return_face: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[pd.DataFrame]], Optional[np.ndarray]]:
        """
        Detects a face in an image, computes its embedding, and searches
        the index.

        Args:
            image: Image file path or a NumPy array (image).
            k: The number of neighbors to retrieve.
            return_metadata: Whether to fetch metadata for found IDs.
            return_face: Whether to return the detected face image crop.

        Returns:
            A tuple containing:
            - distances (np.ndarray): Distances to the k-nearest neighbors.
            - ids (np.ndarray): FAISS IDs of the k-nearest neighbors.
            - results (Optional[List[pd.DataFrame]]): Metadata results.
            - face (Optional[np.ndarray]): The detected face crop, if
              `return_face=True`, otherwise None.
        
        """
        self._ensure_models_loaded()

        if not isinstance(image, (str, Path, np.ndarray)):
            raise TypeError(
                f"image must be str, Path, or np.ndarray. Got: {type(image)}"
            )
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer. Got: {k}")
        if not isinstance(return_metadata, bool):
            raise TypeError(
                f"return_metadata must be a boolean. Got: {type(return_metadata)}"
            )
        if not isinstance(return_face, bool):
            raise TypeError(
                f"return_face must be a boolean. Got: {type(return_face)}"
            )

        if isinstance(image, (str, Path)):
            logger.info("Reading image from path: %s", image)
            try:
                image_arr = read_image(image)
            except Exception:
                logger.exception("Failed to read image at %s", image)
                raise
        else:
            image_arr = image

        # Detect Face
        try:
            faces = self.face_detector.detect(image_arr)
            n_faces = len(faces)
            logger.info("Face detection found %d face(s).", n_faces)
        except Exception:
            logger.exception("Failed to detect face in the provided image.")
            raise

        if n_faces > 1:
            logger.error("More than 1 face (%d) detected.", n_faces)
            raise ValueError("Multiple faces detected; expected exactly one.")
        if n_faces == 0:
            logger.error("No face detected.")
            raise ValueError("No face detected; expected exactly one.")

        face = faces[0]

        # Compute Embedding
        try:
            query = self.embedder.compute_embeddings(face)
            logger.info("Computed embeddings for the detected face.")
        except Exception:
            logger.exception("Failed to compute embeddings for the face.")
            raise

        # Search Index
        distances, ids, results = self.search_query(query, k, return_metadata)

        if return_face:
            return distances, ids, results, face
        else:
            return distances, ids, results, None

    def add_image(self, image: ImageInput, metadata: Dict[str, Any]):
        """
        Detects face(s) in an image, computes embeddings, and adds them
        to the index with the given metadata.

        If multiple faces are detected, all are added with the *same* metadata.

        Args:
            image: Image file path or a NumPy array (image).
            metadata: A *single* metadata dictionary to associate with
                      all detected faces from this image.
        
        """
        self._ensure_models_loaded()

        if not isinstance(image, (str, Path, np.ndarray)):
            raise TypeError(
                f"image must be str, Path, or np.ndarray. Got: {type(image)}"
            )
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict. Got: {type(metadata)}"
            )

        if isinstance(image, (str, Path)):
            logger.info("Reading image from path: %s", image)
            try:
                image_arr = read_image(image)
            except Exception:
                logger.exception("Failed to read image at %s", image)
                raise
        else:
            image_arr = image

        # Detect Faces
        try:
            faces = self.face_detector.detect(image_arr)
            logger.info("Detected %d face(s) in image.", len(faces))
        except Exception:
            logger.exception("Failed to detect faces in the image.")
            raise

        if not faces:
            logger.info("No faces detected in image, nothing to add.")
            return

        embeddings_batch = []
        metadata_batch = []

        # Compute Embeddings for each face
        for face in faces:
            try:
                embedding = self.embedder.compute_embeddings(face)
                embeddings_batch.append(embedding)
                # --- FIX: Use .copy() to avoid mutable reference bug ---
                metadata_batch.append(metadata.copy()) 
            except Exception:
                logger.exception(
                    "Failed to compute embeddings for a detected face."
                )

                raise

        # Add to Index
        try:
            self.add(embeddings=embeddings_batch, metadata=metadata_batch)
            logger.info(
                "Successfully added %d embeddings for the image.",
                len(embeddings_batch)
            )
        except Exception:
            logger.exception(
                "Failed to add embeddings to the FAISS Index."
            )
            raise

    def populate_images(self, image_data: Dict[str, List[Any]], batch_size: int = 32):
        """
        Populates the index from a dictionary of image paths and metadata.

        The `image_data` dict must contain an 'img_path' key with a list
        of image paths. Other keys are treated as metadata lists, aligned
        by index with 'img_path'.

        Example:
            image_data = {
                "img_path": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
                "name": ["Alice", "Bob"],
                "source": ["doc1", "doc2"]
            }

        Args:
            image_data: Dictionary mapping metadata keys to lists of values.
                        'img_path' is required.
            batch_size: Batch size for populating images (e.g. batch_size=32, 32 images are uploaded at a time)

        """
        self._ensure_models_loaded()

        if not isinstance(image_data, dict):
            raise TypeError(
                f"image_data must be a dict. Got: {type(image_data)}"
            )
        
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an integer. Got: {type(batch_size)}"
            )
        
        img_paths = image_data.get("img_path", [])
        
        if not (batch_size > 0):
            raise ValueError(f"batch_size should be larger than 0. You have {batch_size}")
        
        # Ensure batch_size is not larger than total images, if img_paths is not empty
        if img_paths:
            batch_size = min(batch_size, len(img_paths))
        elif batch_size != 32: # if img_paths is empty, use default or provided
            pass

        embeddings_batch = []
        metadata_batch = []

        if not img_paths:
            raise ValueError(
                "`image_data` must contain a non-empty 'img_path' list."
            )

        # Prepare metadata keys (exclude img_path)
        metadata_keys = [k for k in image_data.keys() if k != "img_path"]

        logger.info(
            "Starting to populate FAISS Index from %d images...",
            len(img_paths)
        )
        for i, img_path in enumerate(
            tqdm(img_paths, desc="Populating FAISS index", unit="image")
        ):
            img_path_str = str(img_path)
            
            try:
                img = read_image(img_path)
            except Exception:
                logger.warning(
                    "Failed to read image at %s. Skipping.", img_path_str
                )
                continue

            try:
                faces = self.face_detector.detect(img)
            except Exception:
                logger.warning(
                    "Failed to detect faces in %s. Skipping.", img_path_str
                )
                continue

            if not faces:
                logger.info("No faces detected for image: %s", img_path_str)
                continue

            # Gather metadata for this image
            metadata = {k: image_data[k][i] for k in metadata_keys}

            for face in faces:
                try:
                    embeddings = self.embedder.compute_embeddings(face)
                except Exception:
                    logger.warning(
                        "Failed to calculate embeddings for face in %s. Skipping face.",
                        img_path_str
                    )
                    continue 

                embeddings_batch.append(embeddings)
                metadata_batch.append({"img_path": img_path_str, **metadata})

                # Batch insert
                if len(embeddings_batch) >= batch_size:
                    logger.info(
                        "Inserting batch of %d embeddings...",
                        len(embeddings_batch)
                    )
                    try:
                        self.add(
                            embeddings=embeddings_batch,
                            metadata=metadata_batch
                        )
                    except Exception:
                        logger.exception("Failed to insert batch. Stopping.")
                        raise
                    embeddings_batch.clear()
                    metadata_batch.clear()

        # Add any remaining items
        if embeddings_batch:
            logger.info(
                "Inserting final batch of %d embeddings...",
                len(embeddings_batch)
            )
            try:
                self.add(
                    embeddings=embeddings_batch, metadata=metadata_batch
                )
            except Exception:
                logger.exception("Failed to insert final batch.")
                raise
        
        logger.info("FAISS Index population complete.")

    def _save(self):
        """Save the FAISS index and metadata to disk."""
        # Save FAISS Index
        try:
            faiss.write_index(self.index, str(self.index_file))
            logger.info("FAISS Index saved to %s", self.index_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception("Failed to save FAISS Index at %s",
                             self.index_file.relative_to(PROJECT_ROOT))
            raise

        # Save Metadata Parquet
        # Ensure object columns are strings to avoid parquet errors
        for col in self.emb_metadata.select_dtypes(include=["object"]).columns:
            self.emb_metadata[col] = self.emb_metadata[col].astype(str)

        try:
            self.emb_metadata.to_parquet(
                self.emb_metadata_file,
                index=False,
                engine="fastparquet" 
            )
            logger.info("Embedding metadata saved to %s",
                        self.emb_metadata_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception(
                "Failed to save embedding metadata at %s",
                self.emb_metadata_file.relative_to(PROJECT_ROOT)
            )
            raise

        # Save Index Metadata JSON
        meta = {
            "index_type": self.index_type,
            "dim": self.dim,
            "models": {
                "face_detect_model": str(self.face_detect_model),
                "embeddings_model": str(self.embeddings_model)
            },
            "vector_count": self.index.ntotal,
        }

        try:
            save_json(meta, self.index_metadata_file)
            logger.info("Index metadata saved to %s",
                        self.index_metadata_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception(
                "Failed to save index metadata at %s",
                self.index_metadata_file.relative_to(PROJECT_ROOT)
            )
            raise

    def _load_existing(self):
        """Load an existing FAISS index and metadata from disk."""
        logger.info("Loading existing FAISS Index...")
        # Load FAISS Index
        try:
            self.index = faiss.read_index(str(self.index_file))
            logger.info("FAISS Index loaded from - %s", self.index_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception("Failed to load FAISS Index at %s",
                             self.index_file.relative_to(PROJECT_ROOT))
            raise

        # Load Metadata Parquet
        try:
            self.emb_metadata = pd.read_parquet(self.emb_metadata_file)
            logger.info("Embedding metadata loaded from - %s",
                        self.emb_metadata_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception(
                "Failed to load embedding metadata at %s",
                self.emb_metadata_file.relative_to(PROJECT_ROOT)
            )
            raise

        # Load Index Metadata JSON
        try:
            index_metadata = read_json(self.index_metadata_file)
            logger.info("Index metadata loaded from - %s",
                        self.index_metadata_file.relative_to(PROJECT_ROOT))
        except Exception:
            logger.exception(
                "Failed to load index metadata at %s",
                self.index_metadata_file.relative_to(PROJECT_ROOT)
            )
            raise

        # Set attributes
        self.dim = index_metadata["dim"]
        self.index_type = index_metadata["index_type"]
        self.face_detect_model = Path(index_metadata["models"]["face_detect_model"])
        self.embeddings_model = Path(index_metadata["models"]["embeddings_model"])

        logger.info(
            "Loaded index with %d vectors, dim=%d",
            self.index.ntotal, self.dim
        )
        logger.info(
            "Face detection model -> %s",
            self.face_detect_model.relative_to(PROJECT_ROOT)
        )

        logger.info(
            "Embeddings model -> %s",
            self.embeddings_model.relative_to(PROJECT_ROOT)
        )

    def _load_models(self):
        """
        Lazily loads the face detection and embedding models if they
        are not already loaded.
        
        Also loads their configurations if they haven't been loaded yet
        (e.g., when loading an existing index).
        """
        # Check if models are already loaded
        if self.face_detector and self.embedder:
            return

        logger.info("Loading models (face detector and embedder)...")

        # --- FIX: Lazily load configs if they are None ---
        # This handles the case where we loaded an existing index
        try:
            if self.face_detect_model_config is None:
                validate_model(self.face_detect_model)
                self.face_detect_model_config = read_model_config(
                    self.face_detect_model
                )
                logger.info("Lazily loaded face detect config.")
        except Exception:
            logger.exception(
                "Failed to load config for face_detect_model: %s",
                self.face_detect_model.relative_to(PROJECT_ROOT)
            )
            raise

        try:
            if self.embeddings_model_config is None:
                validate_model(self.embeddings_model)
                self.embeddings_model_config = read_model_config(
                    self.embeddings_model
                )
                logger.info("Lazily loaded embeddings model config.")
        except Exception:
            logger.exception(
                "Failed to load config for embeddings_model: %s",
                self.embeddings_model.relative_to(PROJECT_ROOT)
            )
            raise
        # --- END FIX ---


        # Check if configs are set (they should be after the block above)
        if not (self.face_detect_model_config and self.embeddings_model_config):
            raise ValueError(
                "Cannot load models: face_detect_model_config and/or "
                "embeddings_model_config are not set."
            )

        # Load models
        try:
            if self.face_detector is None:
                self.face_detector = load_model(self.face_detect_model_config)
                logger.info("Face detection model loaded.")
        except Exception:
            logger.exception("Failed to load the face detection model.")
            raise

        try:
            if self.embedder is None:
                self.embedder = load_model(self.embeddings_model_config)
                logger.info("Embeddings model loaded.")
        except Exception:
            logger.exception("Failed to load the embeddings model.")
            raise

    def _ensure_models_loaded(self):
        """
        Checks if models are loaded and calls _load_models() if not.
        """
        if self.face_detector is None or self.embedder is None:
            logger.info(
                "Models not yet loaded. Triggering model load..."
            )
            self._load_models()