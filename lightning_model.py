# torch
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
import pytorch_lightning as pl

# other libraries
import argparse
import os
import matplotlib.pylab as plt
import numpy as np

# layers and models
from utils.autocorr import autocorr_batch
from losses import BaseCorrLoss


def weights_init_normal(m):
    """
    Initializes network weights
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class BaseCorrModel(pl.LightningModule):
    """
    Pytorch lightning class for the autocorrelation
    inversion problem
    """
    def __init__(
        self,
        Network,
        lr=1e-4,
        b1=0.5,
        b2=0.999,
        imagePath="Progress/",
        partial=False,
        zoom=False,
        radius=None,
        optim="adam",
        norm="l1",
        loss="corr",
        weight=1.0,
        shrink_mask=False,
        shrink_epoch=500,
        shrink_step=10,
        radius_step=1,
        scheduler_milestone=None,
        scheduler_gamma=0.1,
        one_cycle_lr=None
    ):
        """
        Instantiate the the pytorch lighting model for the
        autocorrelation inversion problem

        Inputs
        ----------
        'Network' = nn.Module;  used neural network
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = Network
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.partial = partial
        self.zoom = zoom
        self.radius = radius
        self.optim = optim
        self.norm = norm
        self.loss = loss
        self.weight = weight
        self.imagePath = imagePath
        self.shrink_mask = shrink_mask
        self.shrink_epoch = shrink_epoch
        self.shrink_step = shrink_step
        self.radius_step = radius_step
        self.stored_epoch = self.current_epoch
        self.scheduler_milestone = scheduler_milestone
        self.scheduler_gamma = scheduler_gamma
        self.one_cycle_lr = one_cycle_lr

        # initialize network weights
        self.net.apply(weights_init_normal)

        # initialize loss
        self.CorrLoss = BaseCorrLoss(128,
                                     128,
                                     self.radius,
                                     mask=self.partial,
                                     outer_weight=self.weight)

    def forward(self, corr_gt):
        """
        Computes the inference for the autocorrelation problem

        Inputs
        ----------
        'corr_gt' = tensor;  the autocorrelation ground truth image

        Outputs
        -------
        'img_recon' = tensor; the image reconstructed from the autocorrelation
        """
        img_recon = self.net(corr_gt)
        return img_recon

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step

        Inputs
        ----------
        'batch' = tensor;  a single iteration of the dataloader
        'batch_idx' = int;  iteration index
        """

        # map batch to images
        (imgs, _) = batch
        batch_size = imgs.shape[0]

        # Get network inputs and ground truths
        corr_gt = imgs[:, 0:1, :, 0:128]
        img_gt = imgs[:, 0:1, :, 128:]

        # Mask input if necessary
        if self.CorrLoss.mask is True:

            # shrink radius if necessary
            if ((self.shrink_mask is True)
                    and (self.current_epoch >= self.shrink_epoch)
                    and ((self.current_epoch + 1) % self.shrink_step == 0)
                    and self.current_epoch != self.stored_epoch
                    and self.radius > 26):
                self.radius = self.radius - self.radius_step
                self.CorrLoss.update_circular_mask(128,
                                                   128,
                                                   radius=self.radius)
                self.stored_epoch = self.current_epoch
            corr_gt = (corr_gt + 1.0) * self.CorrLoss.inner_mask_input.to(
                self.device) - 1.0

        # zoom input if necessary
        if self.zoom is True:
            corr_gt = corr_gt[:, :, 32:96, 32:96]
            corr_gt = transforms.Resize(
                (128, 128), interpolation=InterpolationMode.BICUBIC)(corr_gt)

        # Forward pass
        img_recon = self(corr_gt)

        # Compute loss
        if self.norm == "l1":
            loss = self.CorrLoss.L1(img_recon, img_gt,
                                    corr_flag=self.loss) / batch_size
        else:
            loss = self.CorrLoss.L2(img_recon, img_gt,
                                    corr_flag=self.loss) / batch_size

        self.log("training_loss", loss)
        if self.shrink_mask is True:
            self.log("mask_radius", self.radius)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step

        Inputs
        ----------
        'batch' = tensor;  a single iteration of the dataloader
        'batch_idx' = int;  iteration index
        """

        # map batch to images
        (imgs, _) = batch
        batch_size = imgs.shape[0]

        # Get network inputs and ground truths
        corr_gt = imgs[:, 0:1, :, 0:128]
        img_gt = imgs[:, 0:1, :, 128:]

        # Mask input if necessary
        if self.CorrLoss.mask is True:
            corr_gt = (corr_gt + 1.0) * self.CorrLoss.inner_mask_input.to(
                self.device) - 1.0

        # zoom input if necessary
        if self.zoom is True:
            corr_gt = corr_gt[:, :, 32:96, 32:96]
            corr_gt = transforms.Resize(
                (128, 128), interpolation=InterpolationMode.BICUBIC)(corr_gt)

        # Forward pass
        img_recon = self(corr_gt)

        # Compute loss
        if self.norm == "l1":
            loss = self.CorrLoss.L1(img_recon, img_gt,
                                    corr_flag=self.loss) / batch_size
        else:
            loss = self.CorrLoss.L2(img_recon, img_gt,
                                    corr_flag=self.loss) / batch_size

        # log images every 10 batches
        if batch_idx % 4000 == 0:

            # create images for logging
            corr_recon = autocorr_batch(img_recon + 1.0)
            img_gt_log = img_gt.cpu().data.numpy()[0, 0:1, :, :]
            corr_gt_log = corr_gt.cpu().data.numpy()[0, 0:1, :, :]
            img_recon_log = img_recon.cpu().data.numpy()[0, 0:1, :, :]
            corr_recon_log = corr_recon.cpu().data.numpy()[int(batch_size //
                                                               2), 0:1, :, :]

            img_gt_log = img_gt_log - img_gt_log.min()
            corr_gt_log = corr_gt_log - corr_gt_log.min()
            img_recon_log = img_recon_log - img_recon_log.min()
            corr_recon_log = corr_recon_log - corr_recon_log.min()

            img_gt_log = img_gt_log / img_gt_log.max()
            corr_gt_log = corr_gt_log / corr_gt_log.max()
            img_recon_log = img_recon_log / img_recon_log.max()
            corr_recon_log = corr_recon_log / corr_recon_log.max()

            # logging
            logger = self.logger.experiment
            logger.add_image("image ground truth", img_gt_log, batch_idx)
            logger.add_image("autocorrelation ground truth", corr_gt_log,
                             batch_idx)
            logger.add_image("image reconstruction", img_recon_log, batch_idx)
            logger.add_image("autocorrelation reconstruction", corr_recon_log,
                             batch_idx)
            save_image(
                torch.from_numpy(img_gt_log),
                self.imagePath + "train_img_gt_log_" +
                str(self.current_epoch) + "_" + str(batch_idx) + ".png",
            )
            save_image(
                torch.from_numpy(corr_gt_log),
                self.imagePath + "train_corr_gt_log_" +
                str(self.current_epoch) + "_" + str(batch_idx) + ".png",
            )
            save_image(
                torch.from_numpy(img_recon_log),
                self.imagePath + "train_img_recon_log_" +
                str(self.current_epoch) + "_" + str(batch_idx) + ".png",
            )
            save_image(
                torch.from_numpy(corr_recon_log),
                self.imagePath + "train_corr_recon_log_" +
                str(self.current_epoch) + "_" + str(batch_idx) + ".png",
            )

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):

        # choose optimizer
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         betas=(self.b1, self.b2))
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9)

        # define OneCycleLR scheduler
        if self.one_cycle_lr is not None:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.one_cycle_lr,
                div_factor=1e4,
                final_div_factor=1e5,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3)
            lr_scheduler = {"scheduler": lr_scheduler, "interval":"step"}
            self.scheduler_milestone = None

        # define multi step lr_scheduler
        if self.scheduler_milestone is not None:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.scheduler_milestone],
                gamma=self.scheduler_gamma)

        return [optimizer], [lr_scheduler]
