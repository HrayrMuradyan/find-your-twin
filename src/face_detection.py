from pathlib import Path
from PIL import ImageDraw
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))
from src.image import read_image
import math
import numpy as np

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
        
        import mediapipe as mp
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



class SCR_Face_Detector:
    def __init__(self,
                 model_path,
                 detection_probability=0.4,
                 rotation_angles=None,
                 rotate_when_score_lt=None):
        from scrfd import SCRFD, Threshold

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        self.face_detector = SCRFD.from_path(self.model_path)
        self.probability = detection_probability
        self.threshold = Threshold(probability=self.probability)
        self.rotation_angles = rotation_angles
        self.rotate_when_score_lt = rotate_when_score_lt 

    def _load_image(self, img):
        from PIL import Image
        if isinstance(img, (str, Path)):
            return read_image(img, return_numpy=False) 
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return img

    def extract(self, img):
        return self.face_detector.detect(img, threshold=self.threshold)

    def _get_max_score(self, results):
        if not results:
            return 0.0
        return max([res.probability for res in results])

    def _select_primary_face(self, results, img_size):
        """
        Selects the single best face from a list of detections.
        Strategy: Largest Bounding Box Area.
        """
        if not results:
            return None
        
        def get_area(res):
            w = res.bbox.lower_right.x - res.bbox.upper_left.x
            h = res.bbox.lower_right.y - res.bbox.upper_left.y
            return w * h

        # Sort descending (largest first)
        sorted_results = sorted(results, key=get_area, reverse=True)
        
        # Return the largest face
        return sorted_results[0]

    def detect(self, img):
        """
        Detects, crops and aligns the key image in the photo
        """
        if (self.rotation_angles or self.rotate_when_score_lt) and not (self.rotation_angles and self.rotate_when_score_lt):
            raise ValueError(
                f"Both rotation_angles and rotate_when_score_lt should be provided. "
                f"You provided only one of them. "
                f"You have: rotation_angles = {self.rotation_angles}, rotate_when_score_lt = {self.rotate_when_score_lt}"
            )
            
        image = self._load_image(img)

        # Extract the face (with no rotation)
        best_results = self.extract(image)
        best_score = self._get_max_score(best_results)
        working_image = image

        # If the highest confidence score is lower than the threshold, try rotating the image
        if self.rotation_angles and best_score < self.rotate_when_score_lt:
            for angle in self.rotation_angles:
                rot_img = image.rotate(angle, expand=True)
                res = self.extract(rot_img)
                score = self._get_max_score(res)

                if score > self.rotate_when_score_lt and score > best_score:
                    best_score = score
                    best_results = res
                    working_image = rot_img

        if best_results:
            # Pick only the largest face (key image)
            final_result = self._select_primary_face(best_results, working_image.size)
        else:
            return None

        # Align and Crop
        aligned_face = self._align_single_face(working_image, final_result)         

        return aligned_face

    def _align_single_face(self, full_image, res):
        """
        Internal helper to crop and rotate a single face result.
        """
        kps = res.keypoints       
        bbox = res.bbox

        # Calculate Rotation Angle
        dx = kps.right_eye.x - kps.left_eye.x
        dy = kps.right_eye.y - kps.left_eye.y
        angle = math.degrees(math.atan2(dy, dx))
        
        # Orientation Check
        # this mostly handles minor head tilts.
        eye_center_x = (kps.left_eye.x + kps.right_eye.x) / 2
        eye_center_y = (kps.left_eye.y + kps.right_eye.y) / 2
        
        nose_dx = kps.nose.x - eye_center_x
        nose_dy = kps.nose.y - eye_center_y

        theta_rad = math.radians(-angle)
        nose_rotated_y = nose_dx * math.sin(theta_rad) + nose_dy * math.cos(theta_rad)

        if nose_rotated_y < 0:
            angle += 180

        # Crop
        w_img, h_img = full_image.size
        bbox_coords = (
            max(0, int(bbox.upper_left.x)),
            max(0, int(bbox.upper_left.y)),
            min(w_img, int(bbox.lower_right.x)),
            min(h_img, int(bbox.lower_right.y))
        )
        
        cropped_face = full_image.crop(bbox_coords)
        
        # Rotate
        w_crop, h_crop = cropped_face.size
        center = (w_crop / 2, h_crop / 2)
        
        aligned_face = cropped_face.rotate(angle, center=center, expand=True)
        
        return aligned_face

    def visualize(self, img, detect_result, save_path=None):
        """
        Draws bounding boxes and keypoints on the original image.
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        
        kps = detect_result.keypoints
        bbox = detect_result.bbox

        # Draw Box
        draw.rectangle(
            [(bbox.upper_left.x, bbox.upper_left.y), (bbox.lower_right.x, bbox.lower_right.y)],
            outline="red", width=3
        )

        # Draw Points
        # Map generic names to the SCRFD keypoints
        points_map = {
            "L_Eye": kps.left_eye,
            "R_Eye": kps.right_eye,
            "Nose": kps.nose,
            "M_L": kps.left_mouth,
            "M_R": kps.right_mouth
        }

        for name, point in points_map.items():
            x, y = point.x, point.y
            # Draw circle
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill="green", outline="black")
            # Draw label
            draw.text((x+5, y-5), name, fill="yellow", stroke_width=1, stroke_fill="black")

        if save_path:
            img.save(save_path)
            print(f"Saved visualization to {save_path}")

        return img

