import argparse
from pathlib import Path
import logging
import warnings
# filter torchscript warning, see function to_torchscript
warnings.filterwarnings('ignore')

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

    def to_torchscript(self, model_dir, device, image = None, validate = False):
        '''
        Convert model to torchscript and save to [model config stem]/fast_{device}.torchscript

        Note: if you uncomment filter.warning at the start of the file you may see a warning
        about "Converting a tensor to a Python boolean..." and show the code with
            if len(image.shape) < 3 or image.shape[-3] != 1:
        This is fine because we should expect the input image to be correct, thus the input
        to connectedComponents will also be fine.
        
        Args:
            model_dir(str): directory to save the torchscript model
            device (str): device to run the model on [cuda:[n], cpu, cuda]
            image (str): path to image to trace the model on
            '''
        self.model.to(device)
        self.model.eval()
        if image is not None:
            img = torchvision.io.read_image(image)
            img = img / 255.0
            img = torchvision.transforms.Resize([self.model_size, self.model_size])(img)
            img = img.unsqueeze(0)
        else:
            img = torch.randn(1, 3, self.model_size, self.model_size)
        
        ts_model = torch.jit.trace(self, img.to(device))
        model_path = Path(model_dir) / f"fast_{device}.torchscript"
        ts_model.save(str(model_path))

        # Random test
        if validate:
            rand_input = torch.randn(1, 3, self.model_size, self.model_size)
            output_gt = self.forward(rand_input.to(device))
            output_ts = ts_model(rand_input.to(device))
            assert torch.allclose(output_gt[0], output_ts[0], atol=1e-05)
            logging.info("Torchscript model validated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FAST model to torchscript')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--image', help='test image path', default = None)
    parser.add_argument('--validate', help='Run image for both model and compare results to ensure integrity of the torchscript model', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    wrapper_model = FASTWrapper(args.config)
    wrapper_model.to_torchscript("build", "cuda", args.image, args.validate)