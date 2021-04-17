from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_rgb(image_path: Union[Path, str], lib: str = "cv2") -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2`
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            raise NotImplementedError("Only cv2 is supported.")
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_rgba(image_path: Union[Path, str], lib: str = "cv2") -> np.array:
    """Load RGBA image from path. If it is a 3-channel image, include dummy alpha channel.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2`
    Returns: 4-channel array with RGBA image
    """
    if Path(image_path).is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if image.shape[2] == 3:
                dummy_alpha = np.full(shape=(image.shape[0], image.shape[1]), fill_value=255, dtype=np.uint8)
                image = np.dstack([image, dummy_alpha])

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        else:
            raise NotImplementedError("Only cv2 is supported.")
        return image

    raise FileNotFoundError(f"File not found {image_path}")
