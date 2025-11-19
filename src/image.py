from pathlib import Path
from typing import Union
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(img_path: Union[str, Path]) -> np.ndarray:
    """
    Reads an image from a file path and converts it to RGB format.

    Args:
        img_path (str or Path): Path to the image file.

    Returns:
        np.ndarray: The image in RGB format.

    """
    if not isinstance(img_path, (str, Path)):
        raise TypeError(f"img_path should be a str or Path, got {type(img_path)}")

    img_path = str(img_path)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"No image found at path: {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def show_image(img: Union[str, Path, np.ndarray]):
    """
    Displays an image using matplotlib.

    Args:
        img (str, Path, or np.ndarray): Image to display. Can be a file path or an array.
    """
    if isinstance(img, (str, Path)):
        img = read_image(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError(f"Argument img should be a str, Path, or np.ndarray, got {type(img)}")

    plt.imshow(img)
    plt.axis('off')  
    plt.show()
