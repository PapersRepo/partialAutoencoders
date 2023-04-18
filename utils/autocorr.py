# libraries
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# torchvision
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision.transforms.functional import resize
from torchvision.transforms.transforms import Resize, ToTensor, InterpolationMode


def autocorr_explicit_pad(x):
    """
    Compute autocorrelation for padded image
    """

    # fourier transform plus shift
    x_pad = torch.zeros((2 * x.shape[0] - 1, 2 * x.shape[1] - 1))
    x_pad[0 : x.shape[0], 0 : x.shape[1]] = x
    f = torch.fft.fftn(x_pad, dim=(0, 1), norm="backward")

    # autocorrelation
    AC = torch.fft.ifftn(torch.abs(f) ** 2, dim=(0, 1), norm="backward")
    AC_shift = torch.fft.fftshift(AC).real

    return AC_shift


def autocorr(x):
    """
    Compute autocorrelation for padded image
    """

    # fourier transform plus shift
    f = torch.fft.fftn(x, s=(2 * x.shape[0] - 1, 2 * x.shape[1] - 1), dim=(0, 1), norm="backward")

    # autocorrelation
    AC = torch.fft.ifftn(torch.abs(f) ** 2, dim=(0, 1), norm="backward")
    AC_shift = torch.fft.fftshift(AC).real

    return AC_shift


def autocorr_batch(x):
    """
    Compute autocorrelation for padded image
    """

    # fourier transform
    f = torch.fft.fftn(x, s=(2 * x.shape[2] - 1, 2 * x.shape[3] - 1), dim=(2, 3), norm="backward")

    # autocorrelation plus shift
    AC = torch.fft.ifftn(torch.abs(f) ** 2, dim=(2, 3), norm="backward")
    AC_shift = torch.fft.fftshift(AC).real

    return AC_shift


def autocorr_manual_pad(x, im_res=64):
    """
    Compute autocorrelation for padded image
    """

    # fourier transform plus shift
    f = torch.fft.fftn(x, dim=(0, 1), norm="backward")

    # autocorrelation
    AC = torch.fft.ifftn(torch.abs(f) ** 2, dim=(0, 1), norm="backward")
    AC_shift = torch.fft.fftshift(AC).real

    # pad stuff
    pad_size = int(im_res / 2 - 1)
    AC_shift = F.pad(AC_shift, (pad_size, pad_size, pad_size, pad_size))
    AC_shift = F.pad(AC_shift, (0, 1, 0, 1))

    return AC_shift
