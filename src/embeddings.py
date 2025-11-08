from typing import Union, Optional
from pathlib import Path
import numpy as np
from deepface import DeepFace
from src.image import read_image
from deepface.modules import preprocessing


class DeepFaceEmbedder:
    """
    A wrapper class for generating face embeddings using DeepFace models.

    Features:
    - Supports multiple DeepFace models (e.g., Facenet, VGG-Face, ArcFace, etc.)
    - Loads and reuses models efficiently
    - Converts image paths or numpy arrays into embeddings
    """

    def __init__(
        self,
        model_name: str = "ArcFace",
        enforce_detection: bool = False,
        target_size: list[int, int] = [112, 112],
        normalization: str = "base",
        dim: int = 512
    ):
        """
        Initializes the DeepFaceEmbedder.

        Args:
            model_name (str): The name of the DeepFace model to use.
            enforce_detection (bool): If True, raises an error if no face is detected.
            target_size (list[int, int]): Target size for face resizing [height, width].
            normalization (str): Normalization method for input image.
            dim (int): The dimension of the embeddings.
        """
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.target_size = target_size
        self.normalization = normalization
        self.dim = dim
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Loads and returns the specified DeepFace model."""
        print(f"[INFO] Loading DeepFace model: {model_name} ...")
        model = DeepFace.build_model(model_name)
        print(f"[INFO] Model '{model_name}' loaded successfully.")
        return model

    def compute_embeddings(self, img: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """
        Computes the face embedding for the given image.

        Args:
            img (str, Path, or np.ndarray): Path to the image file, Path object, or a numpy array.

        Returns:
            np.ndarray | None: 1D face embedding vector, or None if embedding fails.
        """
        if isinstance(img, (str, Path)):
            image = read_image(img)
        elif isinstance(img, np.ndarray):
            image = img.copy()
        else:
            raise TypeError(f"Argument 'img' must be str, Path, or np.ndarray, got {type(img)}.")

        try:
            resized_img = preprocessing.resize_image(
                img=image,
                target_size=self.target_size
            )
            normalized_img = preprocessing.normalize_input(
                img=resized_img,
                normalization=self.normalization
            )
            return np.array(self.model.forward(normalized_img))

        except Exception as e:
            print(f"[WARN] Failed to compute embedding: {e}")
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
