import argparse
import random
import torch

import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='resume training from a saved checkpoint')
parser.add_argument('--eval', action='store_true',
                    help='run evaluation of the model from a saved checkpoint')
opt = parser.parse_args()
print(vars(opt))


class DiceCoef(nn.Module):
    def __init__(self):
        super(DiceCoef, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        """
        Compute the Dice Coefficient.

        Args:
            inputs (torch.Tensor): Predicted (inpainted) image.
            targets (torch.Tensor): Ground truth image.
            smooth (float): A small constant to prevent division by zero.

        Returns:
            torch.Tensor: Dice Coefficient.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(targets * inputs)
        union = torch.sum(targets) + torch.sum(inputs)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        """
        Compute the combined Dice Loss and Binary Cross-Entropy Loss.

        Args:
            inputs (torch.Tensor): Predicted (inpainted) image.
            targets (torch.Tensor): Ground truth image.
            smooth (float): A small constant to prevent division by zero.

        Returns:
            torch.Tensor: Combined loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(targets * inputs)
        union = torch.sum(targets) + torch.sum(inputs)
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceMSELoss(nn.Module):
    def __init__(self):
        super(DiceMSELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        """
        Compute the combined Dice Loss and Mean Squared Error (MSE) Loss.

        Args:
            inputs (torch.Tensor): Predicted (inpainted) image.
            targets (torch.Tensor): Ground truth image.
            smooth (float): A small constant to prevent division by zero.

        Returns:
            torch.Tensor: Combined loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(targets * inputs)
        union = torch.sum(targets) + torch.sum(inputs)
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        MSE = F.mse_loss(inputs, targets)
        Dice_MSE = MSE + dice_loss

        return Dice_MSE


def create_mask(image_size, mask_size):
    """
    Creates a binary mask with a specified size.

    Args:
        image_size (tuple): Size of the image.
        mask_size (int): Size of the mask.

    Returns:
        torch.Tensor: Binary mask.
    """
    mask = torch.ones(image_size)

    for _ in range(8):
        start_h = random.randint(0, image_size[0] - mask_size)
        start_w = random.randint(0, image_size[1] - mask_size)
        mask[start_h:start_h+mask_size, start_w:start_w+mask_size] = 0

    return mask
