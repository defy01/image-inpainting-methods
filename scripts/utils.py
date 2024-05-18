"""
Author: Tomáš Rajsigl
Email: xrajsi01@stud.fit.vutbr.cz
Filename: utils.py

The following file defines various utilities in the forms of classes
and functions used in the training and evaluation process.
"""

import os
import random
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from model import VGG19


class MaskDataset(Dataset):
    """
    A PyTorch Dataset class for loading mask images from a specified directory.

    Attributes:
        root_dir (str): The root directory containing the mask images.
        transform (callable or None): Optional transform to be applied to the images.
        images (list): List of filenames of mask images in the root directory.

    Methods:
        __len__(): Returns the total number of mask images in the dataset.
        __getitem__(idx): Retrieves a mask image and its associated metadata from the dataset.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class Visualizer:
    """
    A class for visualizing image inpainting results in the form of an image grid.

    Attributes:
        plot_metrics (bool): Whether to plot evaluation metrics PSNR and SSIM alongside images.

    Methods:
        plot(): Plots the images and saves them to the specified path.
    """

    def __init__(self, plot_metrics=False):
        self.plot_metrics = plot_metrics

    def plot(self, inputs, masked_inputs, outputs, masks, save_path):
        """
        Plots the original, masked, and inpainted images in a grid format.

        Args:
            inputs (torch.Tensor): Tensor containing original images.
            masked_inputs (torch.Tensor): Tensor containing masked images.
            outputs (torch.Tensor): Tensor containing inpainted outputs.
            masks (torch.Tensor): Tensor containing mask images.
            save_path (str): Path to save the plot.
        """
        fig, axs = plt.subplots(3, 5, figsize=(11, 7))
        for i in range(5):
            # Plot original images
            input_image = inputs[i].permute(1, 2, 0).cpu().numpy()
            axs[0, i].imshow(input_image)

            # Plot masked images
            masked_image = masked_inputs[i].permute(1, 2, 0).cpu().numpy()
            axs[1, i].imshow(masked_image)

            # Plot reconstructed images
            generated_result = outputs[i] * masks[i] + inputs[i] * (1 - masks[i])
            inpainted_image = generated_result.permute(1, 2, 0).detach().cpu().numpy()
            axs[2, i].imshow(inpainted_image)

            # Add PSNR and SSIM scores for evaluation
            if self.plot_metrics:
                ssim_score = ssim(
                    input_image,
                    inpainted_image,
                    win_size=1,
                    channel_axis=2,
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
                psnr_score = psnr(input_image, inpainted_image, data_range=1)
                axs[2, i].set_xlabel(f"PSNR/SSIM: {psnr_score:.2f}/{ssim_score:.2f}")

        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])

        axs[0, 0].set_ylabel("Original", fontsize=12)
        axs[1, 0].set_ylabel("Masked", fontsize=12)
        axs[2, 0].set_ylabel("Inpainted", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class FeatureExtractor(nn.Module):
    """
    A class for Perceptual and Style loss computations using a VGG19 network.

    Calculates both losses between two sets of feature maps extracted
    from images using a pre-trained VGG19 network.

    Attributes:
        vgg (VGG19): A pre-trained VGG19 network used for feature extraction.
        criterion (torch.nn.L1Loss): The criterion used for calculating the perceptual and style losses. L1 loss is used in this case.

    Methods:
        compute_gram(x): Computes the Gram matrix of the input feature map.
        forward(x, y): Performs forward pass of the module, computing Perceptual and Style loss.
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = VGG19().to("cuda")
        self.criterion = torch.nn.L1Loss().to("cuda")

    def compute_gram(self, x):
        B, C, H, W = x.size()
        f = x.view(B, C, W * H)
        gram = torch.bmm(f, f.transpose(1, 2)) / (H * W * C)
        return gram

    def forward(self, x, y):
        """
        Performs forward pass of the module, computing Perceptual and Style loss.
        The layers used are relu1_2, relu2_1, relu3_1, relu4_1 and relu5_1 for both terms.

        Args:
            x (torch.Tensor): Input image tensor.
            y (torch.Tensor): Target image tensor.

        Returns:
            A tuple containing the Perceptual and Style losses.
        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0
        style_loss = 0

        for i, j in [(1, 2), (2, 1), (3, 1), (4, 1), (5, 1)]:
            perceptual_loss += self.criterion(x_vgg[f"relu{i}_{j}"], y_vgg[f"relu{i}_{j}"])
            gram_x = self.compute_gram(x_vgg[f"relu{i}_{j}"])
            gram_y = self.compute_gram(y_vgg[f"relu{i}_{j}"])
            style_loss += self.criterion(gram_x, gram_y)

        return perceptual_loss, style_loss


class EarlyStopping:
    """
    Early stopping regularization to avoid overfitting, this class is entirely
    based upon: https://github.com/Bjarten/early-stopping-pytorch

    Original author: Bjarte Mehus Sunde, 2018
    Original author's mail: BjarteSunde@outlook.com

    Licence:
    MIT License

    Copyright (c) 2018 Bjarte Mehus Sunde

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path="early_stop.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class DiceCoef(nn.Module):
    """
    A class for computing the Dice Coefficient.

    Methods:
        forward(inputs, targets, smooth=1e-5): Compute the Dice Coefficient.

    """

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


def prepare_transforms(opt):
    """
    Prepares image and mask transforms based on the specified options.

    Args:
        opt: Distinguishes between training and evaluation mode.

    Returns:
        transform: Image transforms.
        mask_transform: Mask transforms.
        shuffle_flag: Flag indicating whether to shuffle the data.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mask_test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    mask_train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=(0, 180), translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ]
    )

    if opt.eval:
        return transform, mask_test_transform, False
    else:
        return transform, mask_train_transform, True


def prepare_data(transform, mask_transform, shuffle_flag, batch_size):
    """
    Prepares training and test data loaders.

    Args:
        transform: Transformations to be applied to the images.
        mask_transform: Transformations to be applied to the masks.
        shuffle_flag: Flag indicating whether to shuffle the data.
        batch_size: The specified batch size.

    Returns:
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test data.
        mask_dataloader: DataLoader for mask data.
    """
    mask_data = MaskDataset(root_dir="data/mask_dataset", transform=mask_transform)
    training_data = datasets.Places365(root="data", split="train-standard", small=True, download=True, transform=transform)
    test_data = datasets.Places365(root="data", split="val", small=True, download=True, transform=transform)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_flag, num_workers=8, pin_memory=True)
    mask_dataloader = DataLoader(mask_data, batch_size=batch_size, shuffle=shuffle_flag, num_workers=8, pin_memory=True)

    return train_dataloader, test_dataloader, mask_dataloader


def create_mask(image_size, mask_size):
    """
    Creates a binary mask with a specified size of squares.

    Args:
        image_size (tuple): Size of the image.
        mask_size (int): Size of the mask.

    Returns:
        torch.Tensor: Binary mask.
    """
    mask = torch.zeros(image_size)

    for _ in range(8):
        x = random.randint(0, image_size[1] - mask_size)
        y = random.randint(0, image_size[0] - mask_size)
        mask[y : y + mask_size, x : x + mask_size] = torch.ones(mask_size)

    return mask


def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from keys in a model's state dict.
    Used for cases when the model has been trained using multiple GPUs.

    Args:
        state_dict: Model's state dictionary.

    Returns:
        dict: Updated state dictionary.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
