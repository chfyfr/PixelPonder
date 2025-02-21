import cv2
import numpy as np
from PIL import Image

from .annotator.dwpose import DWposeDetector
from .annotator.mlsd import MLSDdetector
from .annotator.canny import CannyDetector
from .annotator.midas import MidasDetector
from .annotator.hed import HEDdetector
from .annotator.tile import TileDetector
from .annotator.zoe import ZoeDetector

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

class Annotator:
    def __init__(self, name: str, device: str, local_dir=None):
        if name == "canny":
            processor = CannyDetector()
        elif name == "openpose":
            processor = DWposeDetector(device, local_dir)
        elif name == "depth":
            processor = MidasDetector(local_dir)
        elif name == "hed":
            processor = HEDdetector(local_dir)
        elif name == "hough":
            processor = MLSDdetector(local_dir)
        elif name == "tile":
            processor = TileDetector()
        elif name == "zoe":
            processor = ZoeDetector(local_dir)
        self.name = name
        self.processor = processor

    def __call__(self, image: Image, width: int, height: int):
        image = np.array(image)
        detect_resolution = max(width, height)
        image, remove_pad = resize_image_with_pad(image, detect_resolution)

        image = np.array(image)
        if self.name == "canny":
            result = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.name == "hough":
            result = self.processor(image, thr_v=0.05, thr_d=5)
        elif self.name == "depth":
            result = self.processor(image)
            result, _ = result
        else:
            result = self.processor(image)

        result = HWC3(remove_pad(result))
        result = cv2.resize(result, (width, height))
        return result

