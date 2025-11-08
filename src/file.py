import json
from pathlib import Path
from typing import Any, Union

def read_json(file_path: Union[str, Path]) -> Any:
    """
    Read and parse a JSON file safely.

    Args:
        file_path (str or Path): Path to the JSON file.

    Returns:
        Any: Parsed Python object (dict, list, etc.) loaded from the JSON file.

    Raises:
        TypeError: If file_path is not a str or Path.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not '.json'.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    # Type check
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be str or Path, got {type(file_path).__name__}")
    
    path = Path(file_path)

    # Existence check 
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Extension check 
    if path.suffix.lower() != ".json":
        raise ValueError(f"File must have a '.json' extension, got '{path.suffix}'")

    # Read and parse JSON 
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data
