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


def show(image: Union[np.array, Path, str], axes: str = "off") -> None:
    """Plots an image.

    Args:
        image (Union[np.array, Path, str]): cv2 image or path-like.
        axes str: Whether to keep matplotlib axes "on" or "off"
    """

    if isinstance(image, str) or isinstance(image, Path):
        image = load_rgb(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots()
    ax.axis(axes)

    ax.imshow(image)


def show_transparent_image(image: Union[np.array, Path, str], color: List = [52, 235, 232], axes: str = "off") -> None:
    """Plots a transparent image on a solid background.

    Args:
        image (Union[np.array, Path, str]): cv2 image or path-like.
        color (List, optional): Background RGB color. Defaults to [52, 235, 232].
        axes (str, optional): Whether to keep matplotlib axes "on" or "off"
    """
    if isinstance(image, str) or isinstance(image, Path):
        image = load_rgba(image)

    assert image.ndim == 4, "Image does not contain an alpha channel."

    solid_bg = np.ones_like(image[:, :, 0:3])
    solid_bg[:, :, 2], solid_bg[:, :, 1], solid_bg[:, :, 0] = color  # RGB
    rasterized = overlay_image_alpha(solid_bg, image, (0, 0))

    show(rasterized, axes)


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


def get_bbox(four_channel_image: np.ndarray, alpha_thresh: int = 125) -> Tuple[int, int, int, int]:
    """Get the opaque-pixels' bounding box for a four channel image.

    Args:
        four_channel_image (np.ndarray): Input image.
        alpha_thresh (int, optional): Defaults to 125.

    Returns:
        Tuple[int, int, int, int]: (x_min, y_min, x_max, y_max)
    """
    assert four_channel_image.shape[2] == 4
    mask = np.array(four_channel_image[:, :, 3] >= alpha_thresh, np.uint8)
    y, x = mask.nonzero()
    x_min, y_min, x_max, y_max = np.min(x), np.min(y), np.max(x), np.max(y)
    return (x_min, y_min, x_max, y_max)


def tight_crop(four_channel_image: np.ndarray, alpha_thresh: int = 0) -> np.ndarray:
    """Returns a four-channel image with all edge rows and columns removed, that are entirely transparent.

    Args:
        four_channel_image (np.ndarray): Input image.
        alpha_thresh (int, optional): Alpha channel threshold. Defaults to 0.

    Returns:
        np.ndarray: Tight-cropped image.
    """
    x_min, y_min, x_max, y_max = get_bbox(four_channel_image, alpha_thresh=alpha_thresh)
    four_channel_image = four_channel_image[y_min:y_max, x_min:x_max]
    return four_channel_image
