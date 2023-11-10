import logging
import warnings
from pathlib import Path

# filter torchscript warning, see function to_torchscript
warnings.filterwarnings('ignore')

import math

import cv2
import numpy as np
import torch
import torchvision
from mmcv import Config

from models import build_model
from models.utils import fuse_module, rep_model_convert


class FASTWrapper(torch.nn.Module):
    '''Wrapper for FAST model for it to be convertible to torchscript.
    
    The original FAST model can not be converted to torchscript because of the
    decoding step: after the model generates a segmentation mask/class score,
    the model uses opencv functions such as cv2.connectedComponents and
    cv2.minAreaRect to separate different text objects and generate bounding
    boxes. These opencv functions are not traceble thus the model can not be
    converted to torchscript.
    
    A wrapper is created to separate the decoding step from the model. An torch
    traceable cv2.connectedComponents (from kornia) is used, thus the
    torchscript model now output (B, 2, H, W) tensor where the 1st channel is
    the segmentation mask (with integer labels for each pixel to group them
    into separate text instances) and the 2nd channel is the class score. 
    The decoding step then involves finding the minimum bouding box (or polygon)
    for each label.
    
    Args:
        '''
    def __init__(self, config_path:str):
        '''
        Args:
            config_path (str): path to config file, expect config/fast/*/*.py
        '''
        super().__init__()
        # Load config
        cfg = Config.fromfile(config_path)
        # if has test_cfg, use the conf_thresh and min_area from it
        if hasattr(cfg, "test_cfg"):
            self.conf_thresh = cfg.test_cfg.min_score
            self.min_area = cfg.test_cfg.min_area
        else:
            self.conf_thresh = 0.88
            self.min_area = 250
        # Empty model
        model = build_model(cfg.model)
        # Download pretrained if doesn't exist
        pretrained_path = Path("pretrained") / (Path(config_path).stem + ".pth")
        if not pretrained_path.exists():
            # download
            url = f"https://github.com/czczup/FAST/releases/download/release/{pretrained_path.name}"
            logging.warning(url)
            torch.hub.download_url_to_file(url, str(pretrained_path))
        else:
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['ema']
            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        
        model = rep_model_convert(model)
        # fuse conv and bn
        model = fuse_module(model)

        # Attributes used in custom model
        self.cfg = cfg
        self.model = model
        self.model_size = cfg.data.train.img_size
    
    def forward(self, img):
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        input_dict = dict(
            imgs=img,
            img_metas=dict(
                org_img_size = [[self.model_size, self.model_size]],
                img_size = [[self.model_size, self.model_size]]
            )
        )
        input_dict.update(dict(cfg=self.cfg))
        return self.model(**input_dict)

    def module_str(self):
        return "fast_wrapper"


def fast_decode(label, score, min_score, min_area, bbox_type):
        keys = torch.unique(label, sorted=True)
        label_num = len(keys)
        bboxes = []
        scores = []
        for index in range(1, label_num):
            i = keys[index]
            ind = (label == i)
            ind_np = ind.data.cpu().numpy()

            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < min_score:
                label[ind] = 0
                continue
            
            if bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
                rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
                bbox = cv2.boxPoints(rect)

            elif bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind_np] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0]
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1).tolist())
            scores.append(score_i)
        return bboxes, scores