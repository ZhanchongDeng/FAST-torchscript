import logging
import warnings
from pathlib import Path

import torch
import torchvision

# filter torchscript warning, see function to_torchscript
warnings.filterwarnings('ignore')


def to_torchscript(output_model_dir:str, model:torch.nn.Module, model_size:int, device:str, image = None, validate = False):
    '''
    Convert model to torchscript and save to [model config stem]/fast_{device}.torchscript

    Note: if you uncomment filter.warning at the start of the file you may see a warning
    about "Converting a tensor to a Python boolean..." and show the code with
        if len(image.shape) < 3 or image.shape[-3] != 1:
    This is fine because we should expect the input image to be correct, thus the input
    to connectedComponents will also be fine.
    
    Args:
        output_model_dir (str): directory to save the torchscript model
        model (torch.nn.Module): fast model to convert to torchscript
        device (str): device to run the model on [cuda:[n], cpu, cuda]
        image (str): path to image to trace the model on
        validate (bool): whether to validate the torchscript model by comparing
            the output of the torchscript model and the original model on a
            random input
        '''
    model.to(device)
    model.eval()
    if image is not None:
        img = torchvision.io.read_image(image)
        img = img / 255.0
        img = torchvision.transforms.Resize(model_size)(img)
        img = img.unsqueeze(0)
    else:
        img = torch.randn(1, 3, model_size, model_size)
    
    ts_model = torch.jit.trace(model, img.to(device))
    model_path = Path(output_model_dir) / f"fast_{device}.torchscript"
    ts_model.save(str(model_path))

    # Random test
    if validate:
        rand_input = torch.randn(1, 3, model_size, model_size)
        output_gt = model(rand_input.to(device))
        output_ts = ts_model(rand_input.to(device))
        assert torch.allclose(output_gt[0], output_ts[0], atol=1e-05)
        logging.info("Torchscript model validated")