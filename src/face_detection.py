from pathlib import Path
import mediapipe as mp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from src.image import read_image

class MPFaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, img):
        image = read_image(img) if isinstance(img, (str, Path)) else img
        results = self.face_detector.process(image)
        faces = []
        if results.detections:
            ih, iw, _ = image.shape
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
                x2, y2 = x1 + int(bbox.width * iw), y1 + int(bbox.height * ih)
                faces.append(image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)])
        return faces

