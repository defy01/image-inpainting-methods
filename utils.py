import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--resume_training', action='store_true', help='specify whether to resume training from a saved checkpoint')
opt = parser.parse_args()
print(vars(opt))

def create_center_mask(image_size, mask_size):
    mask = torch.ones(image_size)
    start_h = (image_size[0] - mask_size) // 2
    start_w = (image_size[1] - mask_size) // 2
    mask[start_h:start_h+mask_size, start_w:start_w+mask_size] = 0
    return mask
