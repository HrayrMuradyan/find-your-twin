import pytest
import sys
import math
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
from pathlib import Path

mock_scrfd_module = MagicMock()
sys.modules["scrfd"] = mock_scrfd_module

from find_your_twin.face_detection import SCR_Face_Detector

def create_fake_result(probability, bbox_area, keypoints=None):
    """Creates a mock result object mimicking SCRFD output."""
    res = MagicMock()
    res.probability = probability
    
    w, h = bbox_area
    res.bbox.upper_left.x = 0
    res.bbox.upper_left.y = 0
    res.bbox.lower_right.x = w
    res.bbox.lower_right.y = h
    
    if keypoints:
        res.keypoints = keypoints
    else:
        res.keypoints.left_eye.x = 10
        res.keypoints.left_eye.y = 10
        res.keypoints.right_eye.x = 20
        res.keypoints.right_eye.y = 10
        res.keypoints.nose.x = 15
        res.keypoints.nose.y = 15 

    return res

@pytest.fixture
def mock_model_path(tmp_path):
    """Creates a dummy model file."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "fake_model.onnx"
    p.touch()
    return str(p)

@pytest.fixture
def detector(mock_model_path):
    """Returns an instance of SCR_Face_Detector."""
    mock_scrfd_module.reset_mock()
    inner_detector = MagicMock()
    mock_scrfd_module.SCRFD.from_path.return_value = inner_detector
    
    detector_instance = SCR_Face_Detector(model_path=mock_model_path)
    return detector_instance

def test_init_validates_path():
    """Verifies ValueError if model path does not exist."""
    with pytest.raises(ValueError, match="Model path does not exist"):
        SCR_Face_Detector(model_path="non_existent_file.onnx")

def test_init_loads_scrfd(mock_model_path):
    """Verifies that SCRFD.from_path is called."""
    mock_scrfd_module.reset_mock()
    SCR_Face_Detector(model_path=mock_model_path)
    mock_scrfd_module.SCRFD.from_path.assert_called_once()

def test_load_image_numpy(detector):
    """Verifies numpy to PIL conversion."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = detector._load_image(arr)
    assert isinstance(img, Image.Image)

def test_load_image_path(detector):
    """Verifies path loading uses read_image."""
    with patch("find_your_twin.face_detection.read_image") as mock_read:
        mock_read.return_value = Image.new("RGB", (10, 10))
        img = detector._load_image("some/path.jpg")
        mock_read.assert_called_with("some/path.jpg", return_numpy=False)
        assert isinstance(img, Image.Image)

def test_load_image_invalid_type(detector):
    """Verifies invalid types raise ValueError."""
    with pytest.raises(ValueError, match="should be one of the following"):
        detector._load_image(12345)

def test_select_primary_face(detector):
    """Verifies selection of largest bounding box."""
    res1 = create_fake_result(0.9, (10, 10)) 
    res2 = create_fake_result(0.8, (50, 50)) 
    res3 = create_fake_result(0.95, (20, 20))

    results = [res1, res2, res3]
    winner = detector._select_primary_face(results)
    assert winner == res2

def test_detect_returns_none_if_no_faces(detector):
    """Verifies None return on empty detection."""
    detector.face_detector.detect.return_value = []
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result = detector.detect(img)
    assert result is None

def test_detect_rotation_config_validation(mock_model_path):
    """Verifies rotation config validation happens inside detect()."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    det1 = SCR_Face_Detector(mock_model_path, rotation_angles=[90])
    with pytest.raises(ValueError, match="Both rotation_angles and rotate_when_score_lt"):
        det1.detect(img)

    det2 = SCR_Face_Detector(mock_model_path, rotate_when_score_lt=0.5)
    with pytest.raises(ValueError, match="Both rotation_angles and rotate_when_score_lt"):
        det2.detect(img)

def test_detect_uses_rotation_fallback(mock_model_path):
    """Verifies rotation logic is triggered on low score."""
    inner_detector = MagicMock()
    mock_scrfd_module.SCRFD.from_path.return_value = inner_detector
    
    det = SCR_Face_Detector(mock_model_path, rotation_angles=[90, 180], rotate_when_score_lt=0.5)

    res_bad = [create_fake_result(0.3, (10,10))]
    res_worse = [create_fake_result(0.2, (10,10))]
    res_good = [create_fake_result(0.8, (10,10))]
    
    det.extract = MagicMock(side_effect=[res_bad, res_worse, res_good])
    det._align_single_face = MagicMock(return_value="Success")

    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_pil_image = MagicMock()
    mock_pil_image.rotate.return_value = mock_pil_image
    det._load_image = MagicMock(return_value=mock_pil_image)

    result = det.detect(img_arr)
    
    assert result == "Success"
    assert mock_pil_image.rotate.call_count == 2 
    args, _ = det._align_single_face.call_args
    assert args[1] == res_good[0] 

def test_align_single_face_geometry(detector):
    """Verifies rotation angle calculation."""
    full_image = MagicMock(spec=Image.Image)
    full_image.size = (100, 100)
    cropped_mock = MagicMock(spec=Image.Image)
    cropped_mock.size = (50, 50)
    full_image.crop.return_value = cropped_mock
    
    res = MagicMock()
    res.bbox.upper_left.x = 0
    res.bbox.upper_left.y = 0
    res.bbox.lower_right.x = 50
    res.bbox.lower_right.y = 50
    
    res.keypoints.left_eye.x = 0
    res.keypoints.left_eye.y = 0
    res.keypoints.right_eye.x = 10
    res.keypoints.right_eye.y = 10
    
    res.keypoints.nose.x = 5
    res.keypoints.nose.y = 15 
    
    detector._align_single_face(full_image, res)
    
    angle_arg = cropped_mock.rotate.call_args[0][0]
    assert math.isclose(angle_arg, 45.0, abs_tol=0.1)

def test_align_single_face_upside_down_correction(detector):
    """Verifies 180 flip for upside down faces."""
    full_image = MagicMock(spec=Image.Image)
    full_image.size = (100, 100)
    cropped_mock = MagicMock(spec=Image.Image)
    cropped_mock.size = (50, 50)
    full_image.crop.return_value = cropped_mock

    res = MagicMock()
    res.keypoints.left_eye.x = 0
    res.keypoints.left_eye.y = 0
    res.keypoints.right_eye.x = 10
    res.keypoints.right_eye.y = 0
    
    res.keypoints.nose.x = 5
    res.keypoints.nose.y = -10 

    res.bbox.upper_left.x = 0
    res.bbox.upper_left.y = 0
    res.bbox.lower_right.x = 50
    res.bbox.lower_right.y = 50

    detector._align_single_face(full_image, res)

    angle_arg = cropped_mock.rotate.call_args[0][0]
    assert math.isclose(angle_arg, 180.0, abs_tol=0.1)


def test_select_primary_face_returns_none_for_empty_results(detector):
    """Verifies that None is returned if the results list is empty."""
    result = detector._select_primary_face([])
    assert result is None