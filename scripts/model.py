"""
Author: Tomáš Rajsigl
Email: xrajsi01@stud.fit.vutbr.cz
Filename: model.py

The original U-Net paper: https://arxiv.org/abs/1505.04597.
The ResNet paper: https://arxiv.org/abs/1512.03385.
The PatchGAN paper: https://arxiv.org/abs/1611.07004.

This implementation contains both a standard and a modified U-Net architecture
with residual blocks from ResNet as the generator network and a PatchGAN discriminator.
A standalone version utilising the partial convolutional layer is also implemented.
For the calculation of the Perceptual and Style loss terms, a pretrained VGG19 is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from partialconv import PartialConv2d
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    """A residual block module."""
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", dilation=1, padding_mode="zeros", norm="none"
    ):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        if norm == "spectral":
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Residual shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        if hasattr(self, "norm"):
            out = self.norm(out)
        out = self.relu(out)

        out = self.conv2(out)
        if hasattr(self, "norm"):
            out = self.norm(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResidualPConvBlock(nn.Module):
    """A residual block utilizing partial convolutions."""
    def __init__(self, in_channels, out_channels, stride=1, padding="same"):
        super(ResidualPConvBlock, self).__init__()
        self.conv1 = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            return_mask=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PartialConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            return_mask=True,
        )

        self.shortcut = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
            return_mask=False,
        )

    def forward(self, x, mask_in):
        out, mask = self.conv1(x, mask_in)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        shortcut_out = self.shortcut(x, mask_in)

        out += shortcut_out
        out = self.relu(out)
        return out, mask


class UNet(nn.Module):
    """
    U-Net architecture implementation inspired by:
    https://github.com/Mostafa-wael/U-Net-in-PyTorch.
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.e11 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        xout = self.outconv(xd42)
        out = F.sigmoid(xout)

        return out


class ResidualUNet(nn.Module):
    """The modified U-Net architecture with additional residual blocks."""
    def __init__(self):
        super(ResidualUNet, self).__init__()

        # Encoder
        self.e11 = ResidualBlock(4, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = ResidualBlock(512, 1024)

        # Decoder
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(1024, 512, kernel_size=2, padding="same")
        )
        self.d11 = ResidualBlock(1024, 512)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(512, 256, kernel_size=2, padding="same")
        )
        self.d21 = ResidualBlock(512, 256)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(256, 128, kernel_size=2, padding="same")
        )
        self.d31 = ResidualBlock(256, 128)

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(128, 64, kernel_size=2, padding="same")
        )
        self.d41 = ResidualBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, 3, kernel_size=1, padding="same")

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        # Encoder
        x_E1 = self.e11(x)
        x_P1 = self.pool1(x_E1)

        x_E2 = self.e21(x_P1)
        x_P2 = self.pool2(x_E2)

        x_E3 = self.e31(x_P2)
        x_P3 = self.pool3(x_E3)

        x_E4 = self.e41(x_P3)
        x_P4 = self.pool4(x_E4)

        x_E5 = self.e51(x_P4)

        # Decoder
        x_U1 = self.upconv1(x_E5)
        x_D1 = self.d11(torch.cat([x_U1, x_E4], dim=1))

        x_U2 = self.upconv2(x_D1)
        x_D2 = self.d21(torch.cat([x_U2, x_E3], dim=1))

        x_U3 = self.upconv3(x_D2)
        x_D3 = self.d31(torch.cat([x_U3, x_E2], dim=1))

        x_U4 = self.upconv4(x_D3)
        x_D4 = self.d41(torch.cat([x_U4, x_E1], dim=1))

        # Output layer
        xout = self.outconv(x_D4)
        out = F.sigmoid(xout)

        return out


class ResidualUNet2(nn.Module):
    """A deeper variant of the Res U-Net architecture."""
    def __init__(self):
        super(ResidualUNet2, self).__init__()

        # Encoder
        self.e11 = ResidualBlock(4, 64, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = ResidualBlock(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = ResidualBlock(128, 256, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = ResidualBlock(256, 512, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = ResidualBlock(512, 1024, kernel_size=3)

        # Decoder
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(1024, 512, kernel_size=2, padding="same")
        )
        self.d11 = ResidualBlock(1024, 512, kernel_size=3)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, kernel_size=2, padding="same")
        )
        self.d21 = ResidualBlock(512, 256, kernel_size=3)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, kernel_size=2, padding="same")
        )
        self.d31 = ResidualBlock(256, 128, kernel_size=3)

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 64, kernel_size=2, padding="same")
        )
        self.d41 = ResidualBlock(128, 64, kernel_size=3)

        # Output layer
        self.outconv = nn.Conv2d(64, 3, kernel_size=3, padding="same")

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        # Encoder
        x_E1 = self.e11(x, mask)
        x_P1 = self.pool1(x_E1)

        x_E2 = self.e21(x_P1)
        x_P2 = self.pool2(x_E2)

        x_E3 = self.e31(x_P2)
        x_P3 = self.pool3(x_E3)

        x_E4 = self.e41(x_P3)
        x_P4 = self.pool4(x_E4)

        x_E5 = self.e51(x_P4)

        # Decoder
        x_U1 = self.upconv1(x_E5)
        x_D1 = self.d11(torch.cat([x_U1, x_E4], dim=1))

        x_U2 = self.upconv2(x_D1)
        x_D2 = self.d21(torch.cat([x_U2, x_E3], dim=1))

        x_U3 = self.upconv3(x_D2)
        x_D3 = self.d31(torch.cat([x_U3, x_E2], dim=1))

        x_U4 = self.upconv4(x_D3)
        x_D4 = self.d41(torch.cat([x_U4, x_E1], dim=1))

        # Output layer
        xout = self.outconv(x_D4)
        out = F.sigmoid(xout)

        return out


class ResidualPConvUNet(nn.Module):
    """A Res U-Net architecture utilizing partial convolutions in the encoder."""
    def __init__(self):
        super(ResidualPConvUNet, self).__init__()

        # Encoder
        self.e11 = ResidualPConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = ResidualPConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = ResidualPConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = ResidualPConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = ResidualPConvBlock(512, 1024)

        # Decoder
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(1024, 512, kernel_size=2, padding="same")
        )
        self.d11 = ResidualBlock(1024, 512)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(512, 256, kernel_size=2, padding="same")
        )
        self.d21 = ResidualBlock(512, 256)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(256, 128, kernel_size=2, padding="same")
        )
        self.d31 = ResidualBlock(256, 128)

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(128, 64, kernel_size=2, padding="same")
        )
        self.d41 = ResidualBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, 3, kernel_size=1, padding="same")

    def forward(self, x, mask):
        mask = 1 - mask
        # Encoder
        x_E1, mask_E1 = self.e11(x, mask)
        x_P1 = self.pool1(x_E1)
        mask_P1 = self.pool1(mask_E1)

        x_E2, mask_E2 = self.e21(x_P1, mask_P1)
        x_P2 = self.pool2(x_E2)
        mask_P2 = self.pool1(mask_E2)

        x_E3, mask_E3 = self.e31(x_P2, mask_P2)
        x_P3 = self.pool3(x_E3)
        mask_P3 = self.pool1(mask_E3)

        x_E4, mask_E4 = self.e41(x_P3, mask_P3)
        x_P4 = self.pool4(x_E4)
        mask_P4 = self.pool1(mask_E4)

        x_E5, _ = self.e51(x_P4, mask_P4)

        # Decoder
        x_U1 = self.upconv1(x_E5)
        x_D1 = self.d11(torch.cat([x_U1, x_E4], dim=1))

        x_U2 = self.upconv2(x_D1)
        x_D2 = self.d21(torch.cat([x_U2, x_E3], dim=1))

        x_U3 = self.upconv3(x_D2)
        x_D3 = self.d31(torch.cat([x_U3, x_E2], dim=1))

        x_U4 = self.upconv4(x_D3)
        x_D4 = self.d41(torch.cat([x_U4, x_E1], dim=1))

        # Output layer
        xout = self.outconv(x_D4)
        out = F.sigmoid(xout)

        return out


class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class VGG19(nn.Module):
    """Pretrained VGG19 for the computation of Perceptual and Style loss."""
    def __init__(self):
        super(VGG19, self).__init__()

        self.features = models.vgg19(weights="IMAGENET1K_V1").features

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Define mean and std for normalization
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to("cuda:0")
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to("cuda:0")

        # Define names for ReLU layers
        self.relu_names = [f"relu{pre}_{pos}" for pre, pos in zip(
            [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        )]

        self.layer_indices = [
            [0, 1],
            [2, 3],
            [4, 5, 6],
            [7, 8],
            [9, 10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19, 20],
            [21, 22],
            [23, 24],
            [25, 26],
            [27, 28, 29],
            [30, 31],
            [32, 33],
            [34, 35]
        ]

        # Initialize VGG layers for the desired output after ReLU activation
        for relu_name, indices in zip(self.relu_names, self.layer_indices):
            setattr(self, relu_name, nn.Sequential())
            for idx in indices:
                getattr(self, relu_name).add_module(str(idx), self.features[idx])

    def forward(self, x):
        # Preprocessing steps - resize and normalize the input as explained in:
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)

        # Store the respective layers' features
        features = {}
        for relu_name in self.relu_names:
            x = getattr(self, relu_name)(x)
            features[relu_name] = x

        return features
