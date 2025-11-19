from pathlib import Path
import mediapipe as mp
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))
from src.image import read_image

class MPFaceDetector:
    """
    A simple wrapper around MediaPipe's FaceDetection solution for extracting
    face crops from images.

    Parameters
    ----------
    model_selection : int, optional
        Which MediaPipe face detection model variant to use.
        - 0: short-range model (2 meters)
        - 1: full-range model (5 meters)
        Default is 1.
    min_detection_confidence : float, optional
        Minimum confidence threshold for a detection to be considered valid.
        Default is 0.5.
    """

    def __init__(self,
                 model_selection=1,
                 min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, img):
        """
        Detect faces in an image and return cropped face regions.

        Parameters
        ----------
        img : str, pathlib.Path, or numpy.ndarray
            Input image. If a path is provided, the image is read from disk.
            If a numpy array is provided, it is used directly.

        Returns
        -------
        list of numpy.ndarray
            A list of cropped face images (each as an array).
            Returns an empty list if no faces are detected.
        """
        image = read_image(img) if isinstance(img, (str, Path)) else img
        results = self.face_detector.process(image)
        faces = []

        if results.detections:
            ih, iw, _ = image.shape
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
                x2, y2 = x1 + int(bbox.width * iw), y1 + int(bbox.height * ih)

                # Clamp and crop
                faces.append(
                    image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                )

        return faces


