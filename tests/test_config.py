import pytest
import yaml

from find_your_twin import config

@pytest.fixture
def mock_root(tmp_path, monkeypatch):
    """Sets the PROJECT_ROOT inside src.config to a temporary directory."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path

def test_get_config_path_success(mock_root):
    """Verifies that get_config_path correctly reads the config path from a mocked pyproject.toml."""
    toml_path = mock_root / "pyproject.toml"
    
    content = """
    [tool.find_your_twin]
    config_path = "configs/test_config.yaml"
    """
    with open(toml_path, "w") as f:
        f.write(content)

    assert config.get_config_path() == "configs/test_config.yaml"

def test_get_config_path_missing_file(mock_root):
    """Ensures a ValueError is raised when pyproject.toml is missing from the project root."""
    with pytest.raises(ValueError, match="pyproject.toml file doesn't exist"):
        config.get_config_path()

def test_not_str_or_path_provided(mock_root):
    """Ensures if the config path is provided it has to be a string or Path"""
    wrong_path = 12354
    with pytest.raises(TypeError, match="Variable config_path has to be a string"):
        config.load_config(wrong_path)

def test_load_config_malformed_yaml(mock_root):
    """
    Triggers the RuntimeError by creating a file with a valid extension 
    but invalid YAML syntax (parsing failure).
    """
    bad_file = mock_root / "broken.yaml"
    
    with open(bad_file, "w") as f:
        f.write("key: [this_list_is_never_closed") 

    with pytest.raises(RuntimeError, match="Failed to open or parse"):
        config.load_config("broken.yaml")

def test_get_config_path_missing_key(mock_root):
    """Ensures a KeyError is raised when the required tool.find_your_twin section is missing."""
    toml_path = mock_root / "pyproject.toml"
    
    content = """
    [tool.other_tool]
    stuff = "irrelevant"
    """
    with open(toml_path, "w") as f:
        f.write(content)

    with pytest.raises(KeyError, match="Missing 'tool.find_your_twin.config_path'"):
        config.get_config_path()

def test_load_config_from_explicit_path(mock_root):
    """Verifies that load_config correctly loads a YAML file when an explicit path is provided."""
    config_file = mock_root / "custom.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"app": {"title": "Test App"}}, f)

    result = config.load_config("custom.yaml")
    assert result["app"]["title"] == "Test App"

def test_load_config_invalid_extension(mock_root):
    """Ensures a ValueError is raised when providing a file with a non-YAML extension."""
    bad_file = mock_root / "config.txt"
    bad_file.touch()

    with pytest.raises(ValueError, match="should refer to a file with a .yml or .yaml extension"):
        config.load_config("config.txt")

def test_load_config_file_not_found(mock_root):
    """Ensures a ValueError is raised when the specified configuration file does not exist."""
    with pytest.raises(ValueError, match="config file provided .* doesn't exist"):
        config.load_config("ghost.yaml")

def test_integration_files_exist_on_disk():
    """Checks that the actual pyproject.toml and configs/main.yaml files exist in the real project root."""
    real_root = config.PROJECT_ROOT
    assert (real_root / "pyproject.toml").exists()
    assert (real_root / "configs/main.yaml").exists()

def test_integration_load_real_config():
    """Loads the actual project configuration and verifies specific key values match expectations."""
    settings = config.load_config()
    
    assert settings["app"]["title"] == "Visual Search API"
    assert settings["models"]["paths"]["face_detect_model"] == "models/face_detect/scrfd"
    assert settings["search"]["retrieval_similarity_threshold"] == 30
    assert settings["image"]["max_size"] == 1280