from typing import Union, Optional
from pathlib import Path
import numpy as np
from deepface import DeepFace
from deepface.modules import preprocessing
from PIL import Image
import logging

# Assuming project structure context for imports
from find_your_twin.image import read_image

logger = logging.getLogger(__name__)

class DeepFaceEmbedder:
    """
    A wrapper class for generating face embeddings using DeepFace models.
    """

    def __init__(
        self,
        model_name: str = "ArcFace",
        enforce_detection: bool = False,
        target_size: list[int, int] = [112, 112],
        normalization: str = "base",
        dim: int = 512
    ):
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.target_size = target_size
        self.normalization = normalization
        self.dim = dim

        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Loads and returns the specified DeepFace model."""
        model = None
        try:
            logging.info("Loading DeepFace model: %s", model_name)
            model = DeepFace.build_model(model_name)
            logging.info("DeepFace Embedding Model '%s' loaded successfully", model_name)

        except Exception as e:
            logging.exception(
                "There was an error building DeepFace embedding model with name '%s'", model_name
            )

        return model

    def compute_embeddings(self, img: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """
        Computes the face embedding for the given image.
        """
        if self.model is None:
            logging.error("Model not loaded. Cannot compute embeddings.")
            return None

        if isinstance(img, (str, Path)):
            image = read_image(img)
        elif isinstance(img, np.ndarray):
            image = img.copy()
        elif isinstance(img, Image.Image):
            image = np.asarray(img)
        else:
            raise TypeError(
                f"Argument 'img' must be str, Path, or np.ndarray, got {type(img)}."
            )

        try:
            # Resize and normalize the image
            resized_img = preprocessing.resize_image(
                img=image,
                target_size=self.target_size
            )
            normalized_img = preprocessing.normalize_input(
                img=resized_img,
                normalization=self.normalization
            )

            # Get the raw embedding, squeeze, normalize
            embedding = np.array(self.model.forward(normalized_img))  
            embedding = embedding.squeeze()
            norm = np.linalg.norm(embedding)
            epsilon = np.finfo(embedding.dtype).eps
            normalized_embedding = embedding / (norm + epsilon)
        
            return normalized_embedding.astype(np.float32)

        except Exception as e:
            logging.warning("Failed to compute embedding: %s", e)
            return None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"enforce_detection={self.enforce_detection}, "
            f"target_size={self.target_size}, "
            f"normalization='{self.normalization}', "
            f"dim={self.dim})"
        )