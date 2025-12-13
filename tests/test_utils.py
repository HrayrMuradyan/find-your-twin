import pytest
import math
import find_your_twin.utils as utils

def test_import_attr_success():
    """Verifies that a valid attribute is correctly imported from an existing standard library module."""
    func = utils.import_attr("math", "sqrt")
    assert func == math.sqrt

def test_import_attr_module_not_found():
    """Ensures ImportError is raised with a specific message when the target module does not exist."""
    with pytest.raises(ImportError, match="Could not import module"):
        utils.import_attr("fake_module_xyz_123", "attr")

def test_import_attr_attribute_not_found():
    """Ensures AttributeError is raised when the module exists but the requested attribute is missing."""
    with pytest.raises(AttributeError, match="has no attribute"):
        utils.import_attr("math", "non_existent_function")

def test_blur_str_standard():
    """Verifies that the string is correctly obscured based on the default or provided percentage."""
    assert utils.blur_str("abcdef", perc=0.5) == "abc***"

def test_blur_str_full_blur():
    """Checks that setting percentage to 1.0 results in a completely obscured string."""
    assert utils.blur_str("secret", perc=1.0) == "******"

def test_blur_str_no_blur():
    """Checks that setting percentage to 0.0 returns the original string unchanged."""
    assert utils.blur_str("visible", perc=0.0) == "visible"

def test_blur_str_empty():
    """Ensures that passing an empty string returns an empty string without errors."""
    assert utils.blur_str("", perc=0.5) == ""