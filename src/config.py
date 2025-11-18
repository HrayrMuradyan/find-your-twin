from pathlib import Path
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.path import get_project_root_path
import tomli

def get_config_path() -> str:
    """
    Load the path to the main project config file from pyproject.toml.

    This function assumes that the `pyproject.toml` file exists in the project root,
    and that it contains a section like:

        [tool.find_your_twin]
        config_path = "configs/main.yaml"

    Returns:
        str: The relative or absolute path to the main config file,
             as specified under [tool.find_your_twin] in pyproject.toml.

    """
    # Get the project root path (1 levels above this file)
    project_root_path = get_project_root_path(1)

    # Path to pyproject.toml
    pyproject_path = project_root_path / "pyproject.toml"

    # Check if pyproject.toml exists
    if not pyproject_path.exists():
        raise ValueError(f"The pyproject.toml file doesn't exist in the root of the project. Expected to have it here: {str(pyproject_path)}")

    # Load the TOML data
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # Check if tool.find_your_twin.config_path exists
    try:
        return data["tool"]["find_your_twin"]["config_path"]
    except KeyError as e:
        raise KeyError(
            "Missing 'tool.find_your_twin.config_path' in pyproject.toml"
        ) from e

def load_config(config_path=None):
    """
    Loads a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the configuration file. If not provided, the default path from pyproject.toml is used.

    Returns
    -------
    dict
        Parsed YAML configuration as a dictionary.

    Raises
    ------
    TypeError
        If the input is not a string or Path.
    ValueError
        If the path does not exist, is not a file, or does not have a .yml or .yaml extension.
    RuntimeError
        If there is an error opening or reading the file.
    """

    # Get config path from argument or function
    if config_path is None:
        config_path = get_config_path()
    else:
        if not isinstance(config_path, (str, Path)):
            raise TypeError(f"Variable config_path has to be a string or a Path object. You have {type(config_path)}")

    # Define the project root path
    project_root_path = get_project_root_path(1)

    # Convert to Path
    path = project_root_path / config_path 

    # Verify if the argument is a file and it exists
    if not path.is_file():
        raise ValueError(f"The config file provided in the pyproject.toml doesn't exist. You provided: {path}")

    # Ensure YAML extension
    if path.suffix not in [".yml", ".yaml"]:
        raise ValueError(f"Path '{path}' should refer to a file with a .yml or .yaml extension.")

    # Try to open and parse
    try:
        with open(path, "r") as f:
            main_config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to open or parse the config file '{path}': {e}")

    return main_config

    


