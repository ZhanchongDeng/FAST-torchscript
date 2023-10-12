from pathlib import Path
import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from mmcv.cnn import get_model_complexity_info
import logging
import warnings
warnings.filterwarnings('ignore')
import json
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from models.utils import generate_bbox
import math

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
    def __init__(self, args):
        super().__init__()
        cfg = Config.fromfile(args.config)
        cfg.test_cfg = dict()
        # using ICDAR2015
        cfg.test_cfg.min_score=args.conf_thresh
        cfg.test_cfg.min_area=args.min_area
        cfg.test_cfg.bbox_type='rect'
        cfg.test_cfg.result_path='outputs/submit_ic15.zip'

        for d in [cfg, cfg.data.test]:
            d.update(dict(
                report_speed=args.report_speed,
            ))
        if args.conf_thresh is not None:
            cfg.test_cfg.min_score = args.conf_thresh
        if args.min_area is not None:
            cfg.test_cfg.min_area = args.min_area

        cfg.batch_size = args.batch_size

        
        model = build_model(cfg.model)
        if args.checkpoint is not None:
            if os.path.isfile(args.checkpoint):
                print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
                logging.info("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
                sys.stdout.flush()
                checkpoint = torch.load(args.checkpoint)
                
                if not args.ema:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint['ema']

                d = dict()
                for key, value in state_dict.items():
                    tmp = key.replace("module.", "")
                    d[tmp] = value
                model.load_state_dict(d)
            else:
                print("No checkpoint found at '{}'".format(args.checkpoint))
                raise
        
        model = rep_model_convert(model)

        # fuse conv and bn
        model = fuse_module(model)

        self.cfg = cfg
        self.model = model
    
    def forward(self, img):
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        input_dict = dict(
            imgs=img,
            img_metas=dict(
                org_img_size = [args.model_size],
                img_size = [args.model_size]
            )
        )
        input_dict.update(dict(cfg=self.cfg))
        return self.model(**input_dict)

    def module_str(self):
        return "fast_wrapper"

    def generate_bbox(keys, label, score, min_score, min_area, bbox_type):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config',help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str)
    parser.add_argument('--report-speed', action='store_true')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--conf-thresh', default=0.88, type=float)
    parser.add_argument('--min-area', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--ema', action='store_false', default = True)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--image-name')
    parser.add_argument('--model-size', type = int)

    args = parser.parse_args()

    # tiny ic17mlt 640
    # model_size = 640
    # args.config = "/workspace/bobby/FAST/config/fast/ic17mlt/fast_tiny_ic17mlt_640.py"
    # args.checkpoint = "/workspace/bobby/FAST/pretrained/fast_tiny_ic17mlt_640.pth"
    # base total_text 800
    # model_size = 800
    # args.config = "/workspace/bobby/FAST/config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py"
    # args.checkpoint = "/workspace/bobby/FAST/pretrained/fast_base_tt_800_finetune_ic17mlt.pth"
    config_name = os.path.basename(args.config)
    args.model_size = [args.model_size, args.model_size]
    logging.basicConfig(filename=f'log.txt', level=logging.INFO)

    main(args)