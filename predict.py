import argparse

import cv2
import numpy as np
import torch
import torchvision

from fast_wrapper import FASTWrapper, fast_decode


def load_image(image_path, model_size):
    img = torchvision.io.read_image(image_path)
    img = img / 255.0
    img = torchvision.transforms.functional.resize(img, (model_size, model_size))
    img = img.unsqueeze(0)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with FAST Model')
    parser.add_argument('image', help='image path')
    parser.add_argument('--wrapper', help='use wrapper model', action='store_true', default=False)
    args = parser.parse_args()

    device = "cuda"
    if args.wrapper:
        # Load Wrapper
        fast_config_path = "/workspace/bobby/FAST/config/fast/ic17mlt/fast_base_ic17mlt_640.py"
        model = FASTWrapper(fast_config_path)
        model_size = model.model_size
        conf_thresh = model.conf_thresh
        min_area = model.min_area
    else:
        # Load Torchscript
        ts_model_path = f"build/fast_{device}.torchscript"
        model = torch.jit.load(ts_model_path, map_location=torch.device(device))
        model_size = 512
        conf_thresh = 0.88
        min_area = 250
    
    model.to(device)
    model.eval()
    # Load Image
    image = load_image(args.image, model_size)
    # Run Model
    with torch.no_grad():
        out = model(image.to(device))
        out = out[0]
    # Decode Output
    bboxes, scores = fast_decode(out[0], out[1], conf_thresh, min_area, "rect")
    # visualize
    cv_image = cv2.imread(args.image)
    original_image_size = cv_image.shape[:2][::-1]
    for bbox in bboxes:
        bbox = np.array(bbox).reshape(-1, 2)
        bbox = bbox * np.array(original_image_size) / model_size
        bbox = bbox.astype(np.int32)
        cv2.drawContours(cv_image, [bbox], 0, (0, 255, 0), 2)
        cv2.putText(cv_image, f"{scores[0]:.2f}", tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("build/output.jpg", cv_image)