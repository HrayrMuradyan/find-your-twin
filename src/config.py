from pathlib import Path
import yaml
import tomli

PROJECT_ROOT = Path(__file__).parent.parent

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

    # Get the path to pyproject.toml from the root
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    # Pyproject.toml file should exist
    if not pyproject_path.exists():
        raise ValueError(f"The pyproject.toml file doesn't exist in the root of the project. Expected to have it here: {str(pyproject_path)}")

    # Load the TOML data
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # tool.find_your_twin.config_path should exist, get it
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

    """

    # Config path is either an argument to the function, or get it from the toml file
    if config_path is None:
        config_path = get_config_path()
    else:
        if not isinstance(config_path, (str, Path)):
            raise TypeError(f"Variable config_path has to be a string or a Path object. You have {type(config_path)}")

    # Get the config path from the root
    path = PROJECT_ROOT / config_path 

    # Verify that the config path is file
    if not path.is_file():
        raise ValueError(f"The config file provided in the pyproject.toml doesn't exist. You provided: {path}")

    # Config file should have yml or yaml extension
    if path.suffix not in [".yml", ".yaml"]:
        raise ValueError(f"Path '{path}' should refer to a file with a .yml or .yaml extension.")

    # Try to open the config file and parse it
    try:
        with open(path, "r") as f:
            main_config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to open or parse the config file '{path}': {e}")

    return main_config

    


