# MLSD Line Detection
# From https://github.com/navervision/mlsd
# Apache-2.0 license

import cv2
import numpy as np
import torch
import os

from einops import rearrange
from huggingface_hub import hf_hub_download
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

from ...annotator.util import annotator_ckpts_path


class MLSDdetector:
    def __init__(self, local_dir=None):
        model_path = os.path.join(annotator_ckpts_path, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            model_path = hf_hub_download(repo_id="lllyasviel/Annotators", filename="mlsd_large_512_fp32.pth", local_dir=local_dir)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.cuda().eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass
        return img_output[:, :, 0]
