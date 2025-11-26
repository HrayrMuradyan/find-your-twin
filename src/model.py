from pathlib import Path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))
from src.file import read_json
from src.utils import import_attr
import logging
logger = logging.getLogger(__name__)

def read_model_config(model_path):
    """
    Reads and returns the configuration of a model from a given directory.

    The function expects a model directory containing a `config.json` file. 
    It validates the input path and ensures the configuration file exists,
    then reads and returns the JSON content as a dictionary.

    Args:
        model_path (str or Path): Path to the model directory containing `config.json`.

    Raises:
        TypeError: If `model_path` is not a string.
        ValueError: If the `config.json` file is missing or the model directory is invalid.

    Returns:
        dict: The contents of the model's `config.json` file.
    """
    if not isinstance(model_path, (str, Path)):
        raise TypeError(
            f"model_path argument should be a string or a Path object. You have {type(model_path)}"
        )

    model_path = Path(model_path)
    model_config_file_path = model_path / "config.json"

    if not model_config_file_path.is_file():
        raise ValueError(f"Model {model_path} is corrupt. Use a different model.")

    model_config = read_json(model_config_file_path)

    logging.info(
        "Model %s config loaded.",
        model_path.relative_to(PROJECT_ROOT)
    )
    logging.info("Config:")
    for parameter, value in model_config.items():
        if isinstance(value, dict):
            logging.info("   %s:", parameter.capitalize())
            for p, v in value.items():
                logging.info("      %s -> %s", p, v)
        else:
            logging.info("   %s -> %s", parameter, value)

    if "model_path" in model_config["parameters"]:
        model_config["parameters"]["model_path"] = str(model_path / model_config["parameters"]["model_path"])

    return model_config

def load_model(model_config):
    """
    Initializes and returns a model instance from a configuration dictionary.

    The function expects a dictionary containing the model's module, class, 
    and initialization parameters. It dynamically imports the specified class 
    and creates an instance using the provided parameters.

    Args:
        model_config (dict): A dictionary containing the keys:
            - "module" (str): Python module path where the model class is defined.
            - "class" (str): Name of the class to instantiate.
            - "parameters" (dict): Keyword arguments to pass to the class constructor.

    Raises:
        TypeError: If `model_config` is not a dictionary.
        KeyError: If required keys ("module", "class", or "parameters") are missing.
        ImportError: If the specified module or class cannot be imported.
        Exception: For any issues during model instantiation.

    Returns:
        object: An instance of the specified model class.

    """
    if not isinstance(model_config, dict):
        raise TypeError(f"model_config argument should be a dict. You have {type(model_config)}")

    model_class_str = model_config["class"]
    model_module_str = model_config["module"]

    model_class = import_attr(model_module_str, model_class_str)
    
    return model_class(**model_config["parameters"])

