from pathlib import Path

from find_your_twin.file import read_json

REQUIRED_KEYS = ["model_name", "class", "module", "parameters"]

def validate_model(model):
    """
    Validates that the provided model path is a valid model directory.

    This function performs several checks to ensure the model directory 
    and configuration are properly structured:
      - The `model` argument is a string or Path object.
      - The directory exists and contains a `config.json` file.
      - The `config.json` includes all required keys defined in `REQUIRED_KEYS`.

    It raises descriptive errors if any of these conditions fail.

    Args:
        model (str or pathlib.Path): Path to the model directory to validate.

    """
    if not isinstance(model, (str, Path)):
        raise TypeError(f"Argument model has to be a string or a Path object. You have {type(model)}")

    model = Path(model)
    if not model.is_dir():
        raise TypeError(f"model has to be a directory. You have {model}")

    model_config_file = model / "config.json"
    if not model_config_file.is_file():
        raise TypeError(f"model folder should contain 'config.json'.")

    model_config = read_json(model_config_file)

    if not all([key.lower() in REQUIRED_KEYS for key in model_config.keys()]):
        raise ValueError(
            "Not all required keys exist in the model config. "
            "This means the model is corrupt and can't be initialized."
        )

    return True
