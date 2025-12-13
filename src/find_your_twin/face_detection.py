from pathlib import Path
import math
import numpy as np
from PIL import Image

from find_your_twin.image import read_image

class SCR_Face_Detector:
    def __init__(self,
                 model_path,
                 detection_probability=0.4,
                 rotation_angles=None,
                 rotate_when_score_lt=None):
        """
        Initialize the SCRFD face detector wrapper.

        Args:
            model_path (str/Path): Path to SCRFD model.
            detection_probability (float): Base probability threshold for detection.
            rotation_angles (list[int/float]): Angles (deg) to try when rotation-based fallback is enabled.
            rotate_when_score_lt (float): Rotate image if best score is below this value.
        """
        from scrfd import SCRFD, Threshold

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load SCRFD face detector model
        self.face_detector = SCRFD.from_path(self.model_path)

        self.probability = detection_probability

        # Detection threshold object used internally by SCRFD
        self.threshold = Threshold(probability=self.probability)

        # Rotation-based fallback parameters
        self.rotation_angles = rotation_angles
        self.rotate_when_score_lt = rotate_when_score_lt

    def _load_image(self, img):
        """
        Normalize input into a PIL Image.

        Accepts:
          - file path (string or Path)
          - numpy array
          - already loaded PIL Image
        """
        
        if isinstance(img, (str, Path)):
            return read_image(img, return_numpy=False) 
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            raise ValueError(
                f"The argument img should be one of the following (str, Path, np.ndarray). You have {type(img)}"
            )
        
        # Returns a PIL.Image
        return img

    def extract(self, img):
        """
        Run face detection on a PIL image and return full detection results.
        """
        return self.face_detector.detect(img, threshold=self.threshold)

    def _get_max_score(self, results):
        """
        Return highest probability among detections.
        If none present, return 0.0.
        """
        if not results:
            return 0.0
        return max([res.probability for res in results])

    def _select_primary_face(self, results):
        """
        Select the single most important face.

        Strategy:
            - Pick the face with the largest bounding box area.

        Args:
            results: List of SCRFD detection results.
        """
        if not results:
            return None
        
        # Compute area of each detected bbox
        def get_area(res):
            w = res.bbox.lower_right.x - res.bbox.upper_left.x
            h = res.bbox.lower_right.y - res.bbox.upper_left.y
            return w * h

        # Sort detections by area (descending)
        sorted_results = sorted(results, key=get_area, reverse=True)
        
        return sorted_results[0]

    def detect(self, img):
        """
        Main detection pipeline:
        
        Steps:
        1. Load image.
        2. Detect faces without rotation.
        3. If detection score is weak, optionally try rotated versions.
        4. Pick the largest (primary) face.
        5. Align and return cropped face image.

        Returns:
            PIL Image of aligned face, or None if no face detected.
        """

        # Both rotation params must be provided together
        if (self.rotation_angles or self.rotate_when_score_lt) and not (self.rotation_angles and self.rotate_when_score_lt):
            raise ValueError(
                f"Both rotation_angles and rotate_when_score_lt should be provided. "
                f"You provided only one of them. "
                f"You have: rotation_angles = {self.rotation_angles}, rotate_when_score_lt = {self.rotate_when_score_lt}"
            )
            
        image = self._load_image(img)

        # Run initial detection
        best_results = self.extract(image)
        best_score = self._get_max_score(best_results)
        working_image = image

        # Fallback: rotate image if initial detection is weak
        if self.rotation_angles and best_score < self.rotate_when_score_lt:
            for angle in self.rotation_angles:
                rot_img = image.rotate(angle, expand=True)
                res = self.extract(rot_img)
                score = self._get_max_score(res)

                # Keep rotated result only if it's better
                if score > self.rotate_when_score_lt and score > best_score:
                    best_score = score
                    best_results = res
                    working_image = rot_img

        if best_results:
            # Choose the best face among all detections
            final_result = self._select_primary_face(best_results)
        else:
            return None

        # Align and crop the selected face
        aligned_face = self._align_single_face(working_image, final_result)         

        return aligned_face

    def _align_single_face(self, full_image, res):
        """
        Crop and align a single detected face.

        Steps:
        1. Estimate tilt angle from eye positions.
        2. Rotate 180° if the nose orientation suggests the face is upside down.
        3. Crop bounding box.
        4. Rotate cropped face for final alignment.

        Returns:
            PIL Image of aligned face.
        """
        kps = res.keypoints
        bbox = res.bbox

        # Compute raw head tilt angle from left/right eye
        dx = kps.right_eye.x - kps.left_eye.x
        dy = kps.right_eye.y - kps.left_eye.y
        angle = math.degrees(math.atan2(dy, dx))
        
        # Compute eye center for orientation correction
        eye_center_x = (kps.left_eye.x + kps.right_eye.x) / 2
        eye_center_y = (kps.left_eye.y + kps.right_eye.y) / 2
        
        # Check if face is upside down using nose relative to eye center
        nose_dx = kps.nose.x - eye_center_x
        nose_dy = kps.nose.y - eye_center_y

        theta_rad = math.radians(-angle)
        # Rotate nose position into a normalized frame
        nose_rotated_y = nose_dx * math.sin(theta_rad) + nose_dy * math.cos(theta_rad)

        # If nose rotated to negative Y, the face is upside down -> rotate 180 degrees
        if nose_rotated_y < 0:
            angle += 180

        # Crop the bounding box from the full image
        w_img, h_img = full_image.size
        bbox_coords = (
            max(0, int(bbox.upper_left.x)),
            max(0, int(bbox.upper_left.y)),
            min(w_img, int(bbox.lower_right.x)),
            min(h_img, int(bbox.lower_right.y))
        )
        
        cropped_face = full_image.crop(bbox_coords)
        
        # Align face by rotating around center
        w_crop, h_crop = cropped_face.size
        center = (w_crop / 2, h_crop / 2)
        
        aligned_face = cropped_face.rotate(angle, center=center, expand=True)
        
        return aligned_face
