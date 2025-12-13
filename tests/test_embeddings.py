import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import find_your_twin.embeddings as embeddings
from find_your_twin.embeddings import DeepFaceEmbedder

class MockDeepFaceModel:
    """Simulates the DeepFace model object."""
    def forward(self, img_input):
        return [np.ones(512) * 10] 

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks DeepFace, internal image reading, and preprocessing."""
    
    # Mock Build Model
    def _mock_build(model_name):
        return MockDeepFaceModel()
    monkeypatch.setattr("deepface.DeepFace.build_model", _mock_build)

    # Mock read_image (from src.image)
    def _mock_read(path):
        return np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(embeddings, "read_image", _mock_read)

    # Mock Preprocessing
    def _resize(img, target_size): 
        return img
    def _normalize(img, normalization): 
        return img
    monkeypatch.setattr("deepface.modules.preprocessing.resize_image", _resize)
    monkeypatch.setattr("deepface.modules.preprocessing.normalize_input", _normalize)


def test_init_success(mock_dependencies):
    """Verify successful model loading and attribute assignment."""
    embedder = DeepFaceEmbedder(model_name="TestModel")
    assert embedder.model_name == "TestModel"
    assert isinstance(embedder.model, MockDeepFaceModel)
    assert "DeepFaceEmbedder" in repr(embedder)

def test_init_load_model_failure(monkeypatch, caplog):
    """Verify that if build_model fails, self.model becomes None (no crash)."""
    def _fail_build(model_name):
        raise ValueError("Download failed")
    
    monkeypatch.setattr("deepface.DeepFace.build_model", _fail_build)

    embedder = DeepFaceEmbedder(model_name="BadModel")
    
    assert embedder.model is None
    assert "There was an error building DeepFace embedding model" in caplog.text

def test_compute_embeddings_no_model(monkeypatch, caplog):
    """Verify compute_embeddings returns None immediately if model failed to load."""
    # Force load failure
    monkeypatch.setattr("deepface.DeepFace.build_model", lambda x: (_ for _ in ()).throw(Exception))
    
    embedder = DeepFaceEmbedder()
    result = embedder.compute_embeddings("test.jpg")
    
    assert result is None
    assert "Model not loaded" in caplog.text

def test_compute_embeddings_input_types(mock_dependencies):
    """Verify support for str, Path, np.ndarray, and PIL.Image inputs."""
    embedder = DeepFaceEmbedder()
    
    # String
    assert isinstance(embedder.compute_embeddings("img.jpg"), np.ndarray)
    # Path
    assert isinstance(embedder.compute_embeddings(Path("img.jpg")), np.ndarray)
    # Numpy
    assert isinstance(embedder.compute_embeddings(np.zeros((50,50,3))), np.ndarray)
    # PIL
    assert isinstance(embedder.compute_embeddings(Image.new('RGB', (50,50))), np.ndarray)

def test_compute_embeddings_invalid_type(mock_dependencies):
    """Verify TypeError is raised for unsupported input types."""
    embedder = DeepFaceEmbedder()
    with pytest.raises(TypeError, match="must be str, Path, or np.ndarray"):
        embedder.compute_embeddings(123)

def test_compute_embeddings_calculation(mock_dependencies):
    """Verify embedding calculation and L2 normalization."""
    embedder = DeepFaceEmbedder(dim=512)
    result = embedder.compute_embeddings(np.zeros((100,100,3)))
    
    assert result.shape == (512,)
    assert result.dtype == np.float32
    assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)

def test_compute_embeddings_processing_error(mock_dependencies, monkeypatch, caplog):
    """Verify that exceptions during processing (e.g., resize) are caught and logged."""
    embedder = DeepFaceEmbedder()
    
    # Force error in resizing
    def _crash(*args, **kwargs):
        raise RuntimeError("Resize crashed")
    monkeypatch.setattr("deepface.modules.preprocessing.resize_image", _crash)

    result = embedder.compute_embeddings(np.zeros((100,100,3)))
    
    assert result is None
    assert "Failed to compute embedding" in caplog.text