# libraries
from utils.autocorr import autocorr_batch
import numpy as np
import torch


class BaseCorrLoss:
    """
    Class for the autocorrelation loss function
    """

    def __init__(self, h, w, radius=None, mask=False, inner_weight=1.0, outer_weight=20.0):
        """
        mask = boolean; flag for the weighting of the loss function
        """
        self.mask = mask
        self.h = h
        self.w = w
        self.radius = radius
        if self.mask is True:
            # input is usually 128x128
            self.inner_mask_input, self.outer_mask_input = self.__create_circular_mask(h, w, self.radius)
            # input is usually 127x127
            self.inner_mask_loss, self.outer_mask_loss = self.__create_circular_mask(h - 1, w - 1, self.radius)
            self.inner_weight = inner_weight
            self.outer_weight = outer_weight

    def __create_circular_mask(self, h, w, radius=None, center=None):
        """
        Creates a circular mask for a [h,w] image

        Inputs
        ----------
        'h' = int; image height
        'w' = int; image width

        Outputs
        -------
        'inner_mask' = ndarray; mask selecting only the center of the image
        'outer_mask' = ndarray; complementary to center_mask
        """

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        # create masks
        inner_mask = dist_from_center <= radius
        outer_mask = dist_from_center > radius

        return torch.from_numpy(inner_mask), torch.from_numpy(outer_mask)

    def update_circular_mask(self, h, w, radius=None, center=None):
        """
        Updates the circular mask for the [h,w] image

        Inputs
        ----------
        'h' = int; image height
        'w' = int; image width

        Outputs
        -------
        'inner_mask' = ndarray; mask selecting only the center of the image
        'outer_mask' = ndarray; complementary to center_mask
        """

        self.h = h
        self.w = w
        self.radius = radius
        if self.mask is True:
            # input is usually 128x128
            self.inner_mask_input, self.outer_mask_input = self.__create_circular_mask(h, w, self.radius)
            # output autocorr is usually 127x127
            self.inner_mask_loss, self.outer_mask_loss = self.__create_circular_mask(h - 1, w - 1, self.radius)

    def L1(self, img_recon, img_gt, corr_flag="corr"):
        """
        Computes the L1 loss function from the reconstructed
        and ground truth images

        Inputs
        ----------
        'img_recon' = tensor;  reconstructed image
        'img_gt' = tensor; ground truth image
        'corr_flag' = str; corr or image loss

        Outputs
        -------
        'loss' = float; loss
        """

        # get proper img reconstruction and ground truth by carving where needed and shifting by
        # one because of the img normalization and the last tanh layer
        img_gt_center = img_gt[:, :, (self.h // 4) : ((self.h * 3) // 4), (self.w // 4) : ((self.w * 3) // 4)] + 1.0
        img_recon_center = img_recon + 1.0

        if corr_flag == "corr":

            # compute loss with autocorrelation
            corr_gt = autocorr_batch(img_gt_center)
            corr_recon = autocorr_batch(img_recon_center)
            if self.mask is True:
                corr_diff = corr_gt - corr_recon
                inner_loss = (corr_diff * self.inner_mask_loss.to(corr_diff.device)).abs().sum()
                outer_loss = (corr_diff * self.outer_mask_loss.to(corr_diff.device)).abs().sum()
                loss = (inner_loss * self.inner_weight + outer_loss * self.outer_weight) / (
                    self.inner_weight + self.outer_weight
                )
            else:
                loss = 0.5 * (corr_gt - corr_recon).abs().sum()
        else:

            # compute loss with reconstructed images
            loss = (img_gt_center - img_recon_center).abs().sum()

        return loss

    def L2(self, img_recon, img_gt, corr_flag="corr"):
        """
        Computes the L1 loss function from the reconstructed
        and ground truth images

        Inputs
        ----------
        'img_recon' = tensor;  reconstructed image
        'img_gt' = tensor; ground truth image
        'corr_flag' = str; corr or image loss

        Outputs
        -------
        'loss' = float; loss
        """

        # get proper img reconstruction and ground truth by carving where needed and shifting by
        # one because of the img normalization and the last tanh layer
        img_gt_center = img_gt[:, :, (self.h // 4) : ((self.h * 3) // 4), (self.w // 4) : ((self.w * 3) // 4)] + 1.0
        img_recon_center = img_recon + 1.0

        if corr_flag == "corr":

            # compute loss with autocorrelation
            corr_gt = autocorr_batch(img_gt_center)
            corr_recon = autocorr_batch(img_recon_center)
            if self.mask is True:
                corr_diff = corr_gt - corr_recon
                inner_loss = (corr_diff * self.inner_mask_loss.to(corr_diff.device)).square().sum()
                outer_loss = (corr_diff * self.outer_mask_loss.to(corr_diff.device)).square().sum()
                loss = (inner_loss * self.inner_weight + outer_loss * self.outer_weight) / (
                    self.inner_weight + self.outer_weight
                )
            else:
                loss = 0.5 * (corr_gt - corr_recon).square.sum()
        else:

            # compute loss with reconstructed images
            loss = (img_gt_center - img_recon_center).square().sum()

        return loss
