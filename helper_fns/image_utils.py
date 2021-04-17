import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def overlay_image_alpha(
    background: np.ndarray,
    foreground: np.ndarray,
    pos: Tuple[int, int],
    alpha_mask: Optional[np.ndarray] = None,  # alpha channel rescaled to [0, 1]
    opacity: float = 1.0,
) -> np.ndarray:
    """Overlay a four-channel image onto a background.

    Args:
        background (np.ndarray): 3 or 4 channels.
        foreground (np.ndarray): 4 channels.
        pos (Tuple[int, int]): top left, bottom right pixel position to paste fg on.
        alpha_mask (Optional[np.ndarray], optional): Optional alpha mask. Defaults to None.

    Returns:
        np.ndarray: Composited image.
    """
    x, y = pos
    if alpha_mask is None:
        alpha_mask = foreground[:, :, 3] / 255.0

    # Image ranges
    y1, y2 = max(0, y), min(background.shape[0], y + foreground.shape[0])
    x1, x2 = max(0, x), min(background.shape[1], x + foreground.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(foreground.shape[0], background.shape[0] - y)
    x1o, x2o = max(0, -x), min(foreground.shape[1], background.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background

    channels = background.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o] * opacity
    alpha_inv = 1.0 - alpha

    composite = background.copy()
    for c in range(channels):
        composite[y1:y2, x1:x2, c] = alpha * foreground[y1o:y2o, x1o:x2o, c] + alpha_inv * background[y1:y2, x1:x2, c]

    return composite


def show(image: Union[np.array, Path, str], transparency: bool = False) -> None:
    """Plots an image.

    Args:
        image (Union[np.array, Path, str]): cv2 image or path-like.
        transparency (bool, optional): If True, respects the alpha channel. Defaults to True.
    """

    if isinstance(image, str) or isinstance(image, Path):
        if transparency:
            image = load_rgba(image)
        else:
            image = load_rgb(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots()
    ax.axis("off")

    if transparency:
        ax.imshow(image, alpha=image[:, :, 3])

    else:
        ax.imshow(image)


def get_image_files(folder: Union[Path, str], extensions: List[str] = [".jpeg", ".jpg", ".png"]) -> List[Path]:
    """Recursively retrieve a list of image files from the root folder and subfolders.

    Args:
        folder (Union[Path, str]): Root directory.
        extensions (List[str], optional):  Defaults to [".jpeg", ".jpg", ".png"].

    Returns:
        List[Path]: List of image paths.
    """
    files = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            if Path(file).suffix in extensions:
                files.append(Path(os.path.join(dirpath, file)))

    return files
