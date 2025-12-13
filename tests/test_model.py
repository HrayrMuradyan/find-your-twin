import pytest
from unittest.mock import MagicMock, patch
import find_your_twin.model as model

def test_read_model_config_invalid_type():
    """Test that read_model_config raises TypeError when model_path is not a string or Path."""
    with pytest.raises(TypeError):
        model.read_model_config(123)

def test_read_model_config_file_not_found(tmp_path):
    """Test that read_model_config raises ValueError when config.json is missing."""
    with pytest.raises(ValueError):
        model.read_model_config(tmp_path)

@patch("find_your_twin.model.read_json")
@patch("find_your_twin.model.PROJECT_ROOT")
def test_read_model_config_success(mock_project_root, mock_read_json, tmp_path):
    """Test that read_model_config correctly parses a valid config file and updates path parameters."""
    config_file = tmp_path / "config.json"
    config_file.touch()
    
    mock_project_root.return_value = tmp_path.parent
    model.PROJECT_ROOT = tmp_path.parent

    mock_data = {
        "parameters": {
            "model_path": "sub_folder/model.bin",
            "learning_rate": 0.01
        },
        "nested_config": {
            "option_a": "value_a"
        },
        "simple_param": "simple_value"
    }
    mock_read_json.return_value = mock_data

    result = model.read_model_config(tmp_path)

    assert result["parameters"]["model_path"] == str(tmp_path / "sub_folder/model.bin")
    assert result["simple_param"] == "simple_value"

def test_load_model_invalid_type():
    """Test that load_model raises TypeError when model_config is not a dictionary."""
    with pytest.raises(TypeError):
        model.load_model("not a dict")

@patch("find_your_twin.model.import_attr")
def test_load_model_success(mock_import_attr):
    """Test that load_model correctly imports the class and returns an instance with provided parameters."""
    mock_class = MagicMock()
    mock_import_attr.return_value = mock_class
    
    config = {
        "module": "src.models",
        "class": "MyModel",
        "parameters": {"param1": 10, "param2": 20}
    }

    result = model.load_model(config)

    mock_import_attr.assert_called_once_with("src.models", "MyModel")
    mock_class.assert_called_once_with(param1=10, param2=20)
    assert result == mock_class.return_value