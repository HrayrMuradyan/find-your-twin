import importlib

def import_attr(module_name: str, attr_name: str):
    """
    Dynamically imports an attribute (class, function, or variable) from a given module.

    Args:
        module_name (str): The full dotted path of the module (e.g. "src.face_detection").
        attr_name (str): The name of the attribute to import (e.g. "MPFaceDetector").

    Returns:
        Any: The imported attribute (class, function, or variable).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute is not found in the module.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr_name}'") from e
    

def blur_str(input_str, perc=0.8):
    """
    Obscure a portion of a string by replacing the last characters with asterisks.

    Parameters
    ----------
    input_str : str
        The original string to be partially blurred.
    perc : float, optional (default=0.8)
        The percentage of the string to blur. 
        For example, perc=0.8 means 80% of the characters (from the end) 
        will be replaced with "*".

    Returns
    -------
    str
        The blurred string where the last `perc * len(input_str)` characters
        are replaced with asterisks.

    Examples
    --------
    >>> blur_str("abcdef", perc=0.5)
    'abc***'
    >>> blur_str("123456789", perc=0.8)
    '1********'
    """
    input_len = len(input_str)
    threshold = int(input_len * (1 - perc))
    return input_str[:threshold] + "*" * int(input_len - threshold)

