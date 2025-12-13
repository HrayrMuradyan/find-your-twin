import pytest
import json
from unittest.mock import patch
import find_your_twin.file as file

@pytest.fixture(autouse=True)
def mock_project_root(tmp_path):
    with patch("find_your_twin.file.PROJECT_ROOT", tmp_path):
        yield

def test_read_json_success(tmp_path):
    """Test that read_json correctly parses a real JSON file created in tmp_path."""
    data = {"key": "value", "number": 123}
    p = tmp_path / "valid.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    result = file.read_json(p)
    assert result == data

def test_read_json_invalid_type():
    """Test that read_json raises TypeError if input is not a string or Path."""
    with pytest.raises(TypeError):
        file.read_json(123)

def test_read_json_file_not_found(tmp_path):
    """Test that read_json raises FileNotFoundError if the file does not exist on disk."""
    p = tmp_path / "ghost.json"
    with pytest.raises(FileNotFoundError):
        file.read_json(p)

def test_read_json_invalid_extension(tmp_path):
    """Test that read_json raises ValueError if the file has the wrong extension."""
    p = tmp_path / "data.txt"
    p.touch() 
    
    with pytest.raises(ValueError):
        file.read_json(p)

def test_read_json_decode_error(tmp_path):
    """Test that read_json raises JSONDecodeError for a file with corrupt content."""
    p = tmp_path / "corrupt.json"
    p.write_text("{ this is not json }", encoding="utf-8")
    
    with pytest.raises(json.JSONDecodeError):
        file.read_json(p)

def test_save_json_success(tmp_path):
    """Test that save_json creates a real file with the correct content."""
    data = {"saved": True, "list": [1, 2]}
    p = tmp_path / "subdir" / "output.json"
    
    file.save_json(data, p)
    
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == data

def test_save_json_invalid_type():
    """Test that save_json raises TypeError if the path argument is invalid."""
    with pytest.raises(TypeError):
        file.save_json({}, 123)

def test_save_json_serialization_error(tmp_path):
    """Test that save_json fails when trying to save non-serializable data (like a set)."""
    p = tmp_path / "bad_data.json"
    bad_data = {"myset": {1, 2, 3}}
    
    with pytest.raises(TypeError):
        file.save_json(bad_data, p)

@patch("pathlib.Path.open")
def test_save_json_write_error(mock_open, tmp_path):
    """Test that save_json handles OS errors (permissions, etc.) by re-raising them."""

    mock_open.side_effect = OSError("Disk full")
    p = tmp_path / "fail.json"
    
    with pytest.raises(OSError):
        file.save_json({"a": 1}, p)