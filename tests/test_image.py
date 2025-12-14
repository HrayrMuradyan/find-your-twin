import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
import find_your_twin.image as image
import sys
import importlib

@patch("find_your_twin.image.cv2.imread")
def test_read_image_success_numpy(mock_imread, tmp_path):
    """Test that read_image successfully returns a numpy array for a valid path."""
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    
    result = image.read_image(tmp_path / "test.jpg")
    
    assert isinstance(result, np.ndarray)
    mock_imread.assert_called_with(str(tmp_path / "test.jpg"))

@patch("find_your_twin.image.cv2.imread")
def test_read_image_success_pil(mock_imread):
    """Test that read_image returns a PIL Image when return_numpy is False."""
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    
    result = image.read_image("test.jpg", return_numpy=False)
    
    assert isinstance(result, Image.Image)

def test_read_image_invalid_type():
    """Test that read_image raises TypeError when path is not a string or Path object."""
    with pytest.raises(TypeError):
        image.read_image(123)

@patch("find_your_twin.image.cv2.imread")
def test_read_image_file_not_found(mock_imread):
    """Test that read_image raises FileNotFoundError when cv2 cannot load the image."""
    mock_imread.return_value = None
    
    with pytest.raises(FileNotFoundError):
        image.read_image("nonexistent.jpg")

@patch("find_your_twin.image.plt.show")
@patch("find_your_twin.image.plt.imshow")
def test_show_image_numpy(mock_imshow, mock_show):
    """Test that show_image calls matplotlib functions correctly for a numpy array input."""
    img_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    image.show_image(img_arr)
    
    mock_imshow.assert_called_with(img_arr)
    mock_show.assert_called_once()

@patch("find_your_twin.image.read_image")
@patch("find_your_twin.image.plt.show")
@patch("find_your_twin.image.plt.imshow")
def test_show_image_path(mock_imshow, mock_show, mock_read_image):
    """Test that show_image loads the image and calls matplotlib functions for a path input."""
    mock_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_read_image.return_value = mock_arr
    
    image.show_image("test.jpg")
    
    mock_read_image.assert_called_with("test.jpg")
    mock_imshow.assert_called_with(mock_arr)

def test_show_image_invalid_type():
    """Test that show_image raises TypeError for inputs that are not str, Path, or numpy array."""
    with pytest.raises(TypeError):
        image.show_image(123)

def test_resize_image_no_change():
    """Test that resize_image returns the original image if it is smaller than max size."""
    img = Image.new('RGB', (100, 100))
    result = image.resize_image(img, 200)
    
    assert result.size == (100, 100)
    assert result is img

def test_resize_image_width_dominant():
    """Test that resize_image correctly resizes when width exceeds max size."""
    img = Image.new('RGB', (400, 200))
    result = image.resize_image(img, 200)
    
    # 400x200 -> max 200 -> scale factor 0.5 -> 200x100
    assert result.size == (200, 100)

def test_resize_image_height_dominant():
    """Test that resize_image correctly resizes when height exceeds max size."""
    img = Image.new('RGB', (200, 400))
    result = image.resize_image(img, 200)
    
    # 200x400 -> max 200 -> scale factor 0.5 -> 100x200
    assert result.size == (100, 200)

def test_resize_image_boundary():
    """Test that resize_image does not resize if the largest dimension exactly matches max size."""
    img = Image.new('RGB', (200, 150))
    result = image.resize_image(img, 200)
    
    assert result.size == (200, 150)
    assert result is img


def test_show_image_handles_missing_matplotlib(capsys):
    """
    Verifies that show_image prints a warning and returns safely
    if plt is None (simulating matplotlib not installed).
    """
    with patch("find_your_twin.image.plt", None):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Call the function
        image.show_image(img)
        
        # Capture stdout to verify the print statement
        captured = capsys.readouterr()
        assert "Matplotlib is not installed. Skipping visualization." in captured.out

def test_optional_import_sets_plt_to_none():
    """
    Verifies that the module sets 'plt' to None if importing matplotlib fails.
    This requires reloading the module in a controlled environment.
    """
    # Explicitly remove matplotlib to force a re-import attempt,
    with patch.dict(sys.modules, {"matplotlib.pyplot": None, "matplotlib": None}):
        
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("Simulated Matplotlib Missing")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Run the top-level code in image.py again
            importlib.reload(image)
            
            # Verify plt is None
            assert image.plt is None

    # Reload the module again normally to restore it for other tests
    importlib.reload(image)
    
    # Verify it's back
    assert image.plt is not None