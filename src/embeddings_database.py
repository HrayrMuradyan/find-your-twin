from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from tqdm import tqdm
from src.image import read_image
from src.validation import validate_model
from src.model import load_model, read_model_config

class AutoFaissIndex:
    def __init__(self,
                 index_path="embeddings_store",
                 face_detect_model=None,
                 embeddings_model=None,
                 n_threshold=100_000):

        # Input validation
        if not isinstance(face_detect_model, (str, None)):
            raise TypeError(f"Variable face_detect_model should be a string or None. You have {type(face_detect_model)}")
        if not isinstance(embeddings_model, (str, None)):
            raise TypeError(f"Variable embeddings_model should be a string or None. You have {type(embeddings_model)}")
        if not isinstance(n_threshold, int):
            raise TypeError(f"Variable n_threshold should be an integer. You have {type(n_threshold)}")
        if not isinstance(index_path, (str, Path)):
            raise TypeError(f"Variable index_path should be a string or Path object. You have {type(index_path)}")
        
        self.face_detect_model = face_detect_model
        self.embeddings_model = embeddings_model

        if face_detect_model and embeddings_model:
            self.face_detector = None
            self.embedder = None

            validate_model(face_detect_model)
            validate_model(embeddings_model)

            self.face_detect_model_config = read_model_config(face_detect_model)
            self.embeddings_model_config = read_model_config(embeddings_model)
            
        self.dim = self.embeddings_model_config['parameters']['dim']
        self.n_threshold = n_threshold
        self.index_type = "flat"

        # Setup storage paths
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "faiss_index.faiss"
        self.emb_metadata_file = self.index_path / "emb_metadata.parquet"
        self.index_metadata_file = self.index_path / "metadata.json"

        database_necessary_files = [self.index_file, self.emb_metadata_file, self.index_metadata_file]
        database_files_exist = [f.exists() for f in database_necessary_files]
        
        # Load existing or initialize
        if all(database_files_exist):
            self._load_existing()
        
        else:
            if any(database_files_exist):
                print("Detected partial index files — resetting everything.")
                for f in database_necessary_files:
                    if f.exists():
                        f.unlink()

            # Initialize clean state
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            self.emb_metadata = pd.DataFrame({
                "id": pd.Series(dtype="int"),
                "img_path": pd.Series(dtype="string")
            })
    
    def add(self, embeddings, metadata):
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_new = embeddings.shape[0]

        if self.emb_metadata.empty:
            start_id = 0
        else:
            start_id = int(self.emb_metadata['id'].max()) + 1
            
        ids = list(range(start_id, start_id + n_new))
        ids_array = np.array(ids, dtype=np.int64)

        # Add vectors to FAISS index
        self.index.add_with_ids(embeddings, ids_array)

        # Add metadata to DataFrame
        df = pd.DataFrame(metadata, copy=False)
        df.insert(0, "id", ids_array.astype("int32"))
        self.emb_metadata = pd.concat([self.emb_metadata, df], ignore_index=True)

        # Save updated index and metadata
        self._save()

    def search_query(self, query, k: int = 5, return_metadata: bool = True):
        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, ids = self.index.search(query, k) 

        if return_metadata:
            results = []
            for row_ids in ids:
                results.append(self.emb_metadata[self.emb_metadata['id'].isin(row_ids)])
            return distances, ids, results
        else:
            return distances, ids

    def search_image(self,
                     image,
                     k: int = 5,
                     return_metadata: bool = True,
                     return_face: np.ndarray = False):
        self._ensure_models_loaded()
        
        if isinstance(image, (str, Path)):
            image = read_image(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Argument image should be string, Path object or a numpy array. You have {type(image)}")

        if not isinstance(k, int):
            raise TypeError(f"Argument k should be an integer. You have {type(k)}")

        if not isinstance(return_metadata, bool):
            raise TypeError(f"Argument return_metadata should be a boolean. You have {type(return_metadata)}")
        
        face = self.face_detector.detect(image)
        n_faces = len(face)
        if n_faces > 1:
            raise ValueError(f"There was more than 1 face detected in the image.")
        elif n_faces == 0:
            raise ValueError(f"No face was detected.")
        
        face = face[0]

        query = self.embedder.compute_embeddings(face)
        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, ids = self.index.search(query, k) 

        if return_metadata:
            results = []
            for row_ids in ids:
                results.append(self.emb_metadata[self.emb_metadata['id'].isin(row_ids)])
            
            if return_face:
                return distances, ids, results, face
            else:
                return distances, ids, results
        else:
            return distances, ids

    def add_image(self, image, metadata):

        self._ensure_models_loaded()
        
        if isinstance(image, (str, Path)):
            image = read_image(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Argument image should be string, Path object or a numpy array. You have {type(image)}")

        if not isinstance(metadata, dict):
            raise TypeError(f"Argument metadata should be a dictionary. You have {type(metadata)}")

        faces = self.face_detector.detect(image)

        for face in faces:
            embeddings = self.embedder.compute_embedding(face)
            self.add(embeddings=embeddings, metadata=metadata)

    def populate_images(self, image_data: dict):
        self._ensure_models_loaded()
       
        batch_size = 32
        embeddings_batch = []
        metadata_batch = []
        
        img_paths = image_data.get("img_path", [])
        if not img_paths:
            raise ValueError("`image_data` must contain a non-empty 'img_path' key.")
        
        # Prepare metadata keys excluding image paths
        metadata_keys = [k for k in image_data.keys() if k != "img_path"]
        
        for i, img_path in enumerate(tqdm(img_paths, desc="Populating FAISS index", unit="image")):
            img = read_image(img_path)
            faces = self.face_detector.detect(img)
        
            if not faces:
                continue
        
            # Gather metadata for this image
            metadata = {k: image_data[k][i] for k in metadata_keys}
        
            for face in faces:
                embeddings = self.embedder.compute_embeddings(face)
                embeddings_batch.append(embeddings)
                metadata_batch.append({"img_path": img_path, **metadata})
        
                # Batch insert
                if len(embeddings_batch) >= batch_size:
                    self.add(embeddings=embeddings_batch, metadata=metadata_batch)
                    embeddings_batch.clear()
                    metadata_batch.clear()
        
        # Add any remaining items
        if embeddings_batch:
            self.add(embeddings=embeddings_batch, metadata=metadata_batch)

    def _save(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_file))
        
        for col in self.emb_metadata.select_dtypes(include=["object"]).columns:
            self.emb_metadata[col] = self.emb_metadata[col].astype(str)
            
        self.emb_metadata.to_parquet(
            self.emb_metadata_file,
            index=False,
            engine="fastparquet"
        )

        meta = {
            "index_type": self.index_type,
            "dim": self.dim,
            "threshold": self.n_threshold,
            "vector_count": self.index.ntotal,
        }

        with self.index_metadata_file.open("w") as f:
            json.dump(meta, f, indent=2)

    def _load_existing(self):
        """Load an existing FAISS index and metadata from disk."""
        print("Loading existing FAISS index...")
        self.index = faiss.read_index(str(self.index_file))
        print(type(self.index))
        self.emb_metadata = pd.read_parquet(self.emb_metadata_file)

        with self.index_metadata_file.open() as f:
            meta = json.load(f)
        self.index_type = meta["index_type"]

    def _load_models(self):
        if not (self.face_detect_model and self.embeddings_model):
            raise ValueError(f"The AutoFaissIndex had to be initialized with the face and embeddings models.\
            You have face_detect_model: {self.face_detect_model}, embeddings_model: {self.embeddings_model}.\
            Thus, you can't load the models.")

        if not (self.face_detector and self.embedder):
            print("[INFO] Initializing the models...")
            
            self.face_detector = load_model(self.face_detect_model_config)
            self.embedder = load_model(self.embeddings_model_config)

    def _ensure_models_loaded(self):
        if self.face_detector is None or self.embedder is None:
            self._load_models()