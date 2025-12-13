from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image

# Import matplotlib only if it is installed
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import logging
logger = logging.getLogger(__name__)

def read_image(img_path: Union[str, Path], return_numpy=True) -> np.ndarray:
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
    return image_rgb if return_numpy else Image.fromarray(image_rgb)


def show_image(img: Union[str, Path, np.ndarray]):
    """
    Displays an image using matplotlib.

    Args:
        img (str, Path, or np.ndarray): Image to display. Can be a file path or an array.
    """
    if plt is None:
        print("Matplotlib is not installed. Skipping visualization.")
        return

    if isinstance(img, (str, Path)):
        img = read_image(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError(f"Argument img should be a str, Path, or np.ndarray, got {type(img)}")

    plt.imshow(img)
    plt.axis('off')  
    plt.show()


def resize_image(image, image_max_size):
    """
    Resizes the image to fit within IMAGE_MAX_SIZE while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input PIL Image object.

    Returns:
        PIL.Image.Image: The resized image if larger than max size, 
                         otherwise the original image.
    """
    if image.height > image_max_size or image.width > image_max_size:
        if image.height > image.width:
            new_height = image_max_size
            new_width = int(image.width * (image_max_size / image.height))
        else:
            new_width = image_max_size
            new_height = int(image.height * (image_max_size / image.width))

        logging.info(
            "Resizing image from %sx%s to %sx%s", 
            image.width, image.height, new_width, new_height
        )
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logging.info("Resizing successful")
        
    return image
