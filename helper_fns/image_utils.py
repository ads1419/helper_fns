import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import urlopen

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


def load_image_url(url: str, read_flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Download the image, convert it to a NumPy array, and then read it into OpenCV format

    Args:
        url (str): Image URL
        read_flag (int, optional): OpenCV image reading flag.
            See https://docs.opencv.org/master/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80

    Returns:
        np.ndarray: cv2 image in RGB format.
    """

    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, read_flag)

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return the image
    return image


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


def show(image: Union[np.array, Path, str], axes: str = "off", title: str = None) -> None:
    """Plots an image.

    Args:
        image (Union[np.array, Path, str]): cv2 image or path-like.
        axes str: Whether to keep matplotlib axes "on" or "off"
        title str: Optionally, show image title
    """

    if isinstance(image, str) or isinstance(image, Path):
        image = load_rgb(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots()
    ax.axis(axes)
    if title:
        ax.set_title(title)

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

    assert image.ndim == 3 and image.shape[2] == 4, "Image does not contain an alpha channel."

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


def crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    image = image[y_min:y_max, x_min:x_max]
    return image


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


def rotate_bound(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by angle degrees while making sure that the object does not leave image frame.
    Resizes the frame to accommodate the rotated object.
    From https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py

    Args:
        image (np.ndarray): cv2 image
        angle (int): Angle in degrees

    Returns:
        np.ndarray: Rotated image
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def resize(image: np.ndarray, width: int = None, height: int = None, inter: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize an image to specific width or height while maintaining aspect ratio.
    From https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py

    Args:
        image (np.ndarray): Input image
        width (int, optional): Image width. Defaults to None.
        height (int, optional): Image height. Defaults to None.
        inter (int, optional): OpenCV interpolation methods. See
            https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Returns:
        np.ndarray: Resized image
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
