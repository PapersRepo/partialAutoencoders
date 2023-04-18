# libraries
import numpy as np
from PIL import Image
from utils.autocorr import autocorr
import matplotlib.pylab as plt

# torch
import torch
import torch.nn.functional as F
import torch.utils.data

# torchvision
import torchvision
import torchvision.datasets
from torchvision.transforms.transforms import (
    Resize,
    ToTensor,
    InterpolationMode,
    RandomRotation,
)

# params
im_res = 64
sing_im_res = 28
ConvertToTensor = ToTensor()
f_autocorr = autocorr


def create_single_image_simple(im_i, name='OOD', folder='./check_dataset/val/'):
    """
    Create an autocorrelation image pair
    from a single image, in this case I create
    a single out of distribution example with two letters
    that were never seen from the network
    """

    # get image
    im_i = ConvertToTensor(im_i).squeeze()

    # padded array
    pad_size = int(np.floor((im_res - im_i.shape[0]) // 2))
    im_i = F.pad(im_i, (pad_size, pad_size, pad_size, pad_size))

    # normalize array
    im_i = (im_i - im_i.min()) / im_i.max()

    # compute autocorrelation
    corr_i = f_autocorr(im_i)
    corr_i[corr_i < 0] = 0

    # normalize autocorrelation
    corr_i = 255 * corr_i / corr_i.max()

    # pad original image
    pad_size = int(im_res / 2 - 1)
    im_i = F.pad(im_i, (pad_size, pad_size, pad_size, pad_size))
    im_i = F.pad(im_i, (0, 1, 0, 1))

    im_i = im_i / im_i.max()
    corr_i = corr_i / corr_i.max()
    AB = torch.zeros((2 * im_res - 1, 2 * (2 * im_res - 1)))
    AB[0:2 * im_res - 1, 0:2 * im_res - 1] = corr_i
    AB[0:2 * im_res - 1, 2 * im_res - 1:2 * (2 * im_res - 1)] = im_i

    # save array
    filename = folder + name + ".png"
    print(filename)
    plt.imsave(filename, AB.numpy(), cmap="gray")


# get image
im_i = Image.open('./paper_data/outOfDistribution.png')
im_i = np.array(im_i.getdata())[:,0].reshape(im_res,im_res)

create_single_image_simple(im_i)
