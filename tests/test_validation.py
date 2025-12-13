import pytest
import json
import find_your_twin.validation as validator

REQUIRED_KEYS = ["model_name", "class", "module", "parameters"]

def test_validate_model_invalid_type():
    """Ensures TypeError is raised if the model argument is not a string or Path object."""
    with pytest.raises(TypeError, match="has to be a string or a Path"):
        validator.validate_model(123)

def test_validate_model_directory_not_found(tmp_path):
    """Ensures TypeError is raised if the provided path is not an existing directory."""
    non_existent_path = tmp_path / "ghost_dir"
    with pytest.raises(TypeError, match="has to be a directory"):
        validator.validate_model(non_existent_path)

def test_validate_model_missing_config_file(tmp_path):
    """Ensures TypeError is raised if the model directory exists but does not contain config.json."""
    with pytest.raises(TypeError, match="should contain 'config.json'"):
        validator.validate_model(tmp_path)

def test_validate_model_invalid_keys(tmp_path):
    """Ensures ValueError is raised if the config contains keys not present in REQUIRED_KEYS."""
    config_file = tmp_path / "config.json"
    
    content = {"model_name": "test", "random_bad_key": "bad"}
    config_file.write_text(json.dumps(content))

    with pytest.raises(ValueError, match="Not all required keys exist"):
        validator.validate_model(tmp_path)

def test_validate_model_success(tmp_path):
    """Verifies that the function returns True when the directory exists, config exists, and keys are valid."""
    config_file = tmp_path / "config.json"
    
    valid_content = {
        "model_name": "TestModel", 
        "class": "TestClass",
        "module": "TestModule",
        "parameters": {}
    }
    config_file.write_text(json.dumps(valid_content))

    assert validator.validate_model(tmp_path) is True