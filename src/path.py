from pathlib import Path

def get_project_root_path(levels_up):
    """
    Return the path to the project root by traversing up a given number of directory levels
    from the location of the current file.

    Parameters:
        levels_up (int): The number of parent directories to go up from the current file.

    Returns:
        Path: The resolved absolute path to the computed root directory.

    """
    if not isinstance(levels_up, int):
        raise ValueError(f"Variable levels_up should be an integer. You have {type(levels_up)}")

    # Get the absolute path of the file
    path = Path(__file__).resolve()

    # If the number of levels is larger than the number of parents raise an error
    if levels_up >= len(path.parents):
        raise ValueError(f"levels_up={levels_up} is too high for path {path}")
    
    # Get the project root
    return path.parents[levels_up]