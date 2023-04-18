# libraries
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

from utils.autocorr import autocorr
import concurrent.futures

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--kind", type=str, default="train", help="create train or test image")
parser.add_argument("--test", dest="test", action="store_true")
parser.add_argument("--no-test", dest="test", action="store_false")
parser.add_argument("--train_samples", type=int, default=300000, help="number of training images")
parser.add_argument("--test_samples", type=int, default=50000, help="number of test images")
parser.set_defaults(test=False)
opt = parser.parse_args()

# params
im_res = 64
sing_im_res = 28

# create datasets from mnist
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

testset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

# convert image to tensor
ConvertToTensor = ToTensor()
ResizeImage = Resize((int(im_res / 2), int(im_res / 2)), InterpolationMode.NEAREST)
ResizeSingImage = Resize((sing_im_res, sing_im_res), InterpolationMode.NEAREST)
RotateSingImage = RandomRotation(45.0)

# select dataset, autocorrelation and folder
if opt.kind == "train":
    chosen_set = "train"
else:
    chosen_set = "test"
train_samples = opt.train_samples
test_samples = opt.test_samples
if chosen_set == "train":

    # select dataset
    dataset = trainset
    folder = "./datasets/mixed/train/"

    # define random vectors
    v_random1 = np.random.randint(0, 60000, size=(train_samples))
    v_random2 = np.random.randint(0, 60000, size=(train_samples))
    v_random_sing = np.random.randint(0, 60000, size=(train_samples))

else:

    # select dataset
    dataset = testset
    folder = "./datasets/mixed/test/"

    # define random vectors
    v_random1 = np.random.randint(0, 10000, size=(test_samples))
    v_random2 = np.random.randint(0, 10000, size=(test_samples))
    v_random_sing = np.random.randint(0, 10000, size=(test_samples))

f_autocorr = autocorr
len_vrandom = len(v_random1)


def pair_xy(x_min, x_max, y_min, y_max, d_min):

    d = 0.0

    while d < d_min:

        # first coord
        x1 = np.random.randint(x_min, x_max)
        y1 = np.random.randint(y_min, y_max)

        # second coord
        x2 = np.random.randint(x_min, x_max)
        y2 = np.random.randint(y_min, y_max)

        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return x1, y1, x2, y2


def create_single_image_mnist(n):
    """
    Create an autocorrelation image pair
    from a mnist image
    """

    # get image
    i = v_random_sing[n]
    im_i, label = dataset[i]
    im_i = ResizeImage(im_i)
    im_i = RotateSingImage(im_i)
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
    AB[0 : 2 * im_res - 1, 0 : 2 * im_res - 1] = corr_i
    AB[0 : 2 * im_res - 1, 2 * im_res - 1 : 2 * (2 * im_res - 1)] = im_i

    # save array
    filename = folder + "AB_mnist_" + str(n) + ".png"
    print(filename)
    plt.imsave(filename, AB.numpy(), cmap="gray")


def create_pair_image_mnist(n):
    """
    Create an autocorrelation image pair
    from 2 mnist images
    """

    # get images
    i, j = v_random1[n], v_random2[n]
    im_i, label = dataset[i]
    im_j, label = dataset[j]
    im_i, im_j = ResizeSingImage(im_i), ResizeSingImage(im_j)
    im_i, im_j = RotateSingImage(im_i), RotateSingImage(im_j)
    im_i, im_j = ConvertToTensor(im_i).squeeze(), ConvertToTensor(im_j).squeeze()

    # build single image
    im_tot = torch.zeros((im_res, im_res))
    x1, y1, x2, y2 = pair_xy(
        int(sing_im_res / 2), im_res - int(sing_im_res / 2), int(sing_im_res / 2), im_res - int(sing_im_res / 2), 35,
    )

    im_tot[
        x1 - int(sing_im_res / 2) : x1 + int(sing_im_res / 2), y1 - int(sing_im_res / 2) : y1 + int(sing_im_res / 2),
    ] = im_i
    im_tot[
        x2 - int(sing_im_res / 2) : x2 + int(sing_im_res / 2), y2 - int(sing_im_res / 2) : y2 + int(sing_im_res / 2),
    ] = im_j
    im_i = im_tot

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
    AB[0 : 2 * im_res - 1, 0 : 2 * im_res - 1] = corr_i
    AB[0 : 2 * im_res - 1, 2 * im_res - 1 : 2 * (2 * im_res - 1)] = im_i

    # save array
    filename = folder + "AB_mnist_" + str(n) + ".png"
    print(filename)
    plt.imsave(filename, AB.numpy(), cmap="gray")


def create_image(n):
    if n % 2 == 0:
        create_single_image_mnist(n)
    else:
        create_pair_image_mnist(n)


if opt.test is True:
    for i in range(100):
        create_image(i)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(create_image, range(len(v_random1)))
