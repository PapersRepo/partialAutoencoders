# torch
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# other libraries
import argparse
import os
import matplotlib.pylab as plt

# models and utils
from deeplab import DeepLab
from utils.autocorr import autocorr_batch

# pytorch lighting
import pytorch_lightning as pl
from lightning_model import BaseCorrModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# numpy
import numpy as np

# torch flags
torch.backends.cudnn.benchmark = True


def main(opt):

    print("Is torch available: ", torch.cuda.is_available())

    # initialize network
    Net = DeepLab(1, in_channels=1, pretrained=False, backbone="resnet101")

    # check optimizer type
    assert opt.optimizer == "adam" or opt.optimizer == "sgd", "optimizer is adam or sgd."

    # check norm type
    assert opt.norm == "l1" or opt.norm == "l2", "loss norm is l1 or l2."

    # check loss type
    assert opt.loss == "corr" or opt.loss == "image", "loss type is corr or image."

    # create training dataset
    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root="./working_training_dataset/",
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((128, 256),
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.n_workers,
    )

    # create validation dataset
    val_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root="./working_validation_dataset/",
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((128, 256),
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.n_workers,
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.checkdir,
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        filename=opt.backend + "-{epoch:02d}-{val_loss:.2f}",
    )
    checkpoint_periodic_callback = ModelCheckpoint(
        dirpath=opt.checkdir,
        every_n_epochs=25,
        save_top_k=-1,
        filename=opt.backend + "-periodic-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # logging
    logger = TensorBoardLogger(opt.logdir, name=opt.backend)

    # instantiate model
    if opt.checkpoint is None:
        CorrModel = BaseCorrModel(
            Net,
            lr=opt.lr,
            b1=opt.b1,
            b2=opt.b2,
            # checkpointPath=opt.checkpoint,
            imagePath=opt.imagedir,
            partial=opt.partial,
            radius=opt.radius,
            zoom=opt.zoom,
            shrink_mask=opt.shrink,
            shrink_epoch=opt.shrink_epoch,
            shrink_step=opt.shrink_step,
            radius_step=opt.radius_step,
            scheduler_milestone=opt.scheduler_milestone,
            scheduler_gamma=opt.scheduler_gamma,
            one_cycle_lr=opt.one_cycle_lr)
    else:

        # get correct device
        print("Loading model from checkpoint")
        torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        CorrModel = BaseCorrModel.load_from_checkpoint(
            opt.checkpoint,
            Network=Net,
            map_location=torch.device(torch_device),
            lr=opt.lr,
            b1=opt.b1,
            b2=opt.b2,
            checkpointPath=opt.checkpoint,
            imagePath=opt.imagedir,
            partial=opt.partial,
            radius=opt.radius,
            zoom=opt.zoom,
            shrink_mask=opt.shrink,
            shrink_epoch=opt.shrink_epoch,
            shrink_step=opt.shrink_step,
            radius_step=opt.radius_step,
            scheduler_milestone=opt.scheduler_milestone,
            scheduler_gamma=opt.scheduler_gamma,
            one_cycle_lr=opt.one_cycle_lr)

    # train
    trainer = pl.Trainer(
        auto_lr_find=opt.lr_find,
        devices=opt.n_gpus,
        num_nodes=opt.n_nodes,
        logger=logger,
        min_epochs=int(opt.n_epochs // 2),
        max_epochs=opt.n_epochs,
        log_every_n_steps=50,
        callbacks=[
            checkpoint_callback, checkpoint_periodic_callback, lr_monitor
        ],
        accelerator="gpu",
        strategy="ddp",
        fast_dev_run=opt.debug,
    )

    # tune learning rate
    if opt.lr_find is True:
        trainer.tune(CorrModel, train_dataloader, val_dataloader)

    # fit model
    trainer.fit(CorrModel, train_dataloader, val_dataloader)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm",
                        type=str,
                        default="l1",
                        help="choice of the norm for the loss")
    parser.add_argument("--loss",
                        type=str,
                        default="corr",
                        help="choice between corr and image loss")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="size of the batches")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=50,
                        help="number of training epochs")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0001,
                        help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="size of each image dimension")  # Autocorrelation is 2*img_size-1
    parser.add_argument("--n_gpus",
                        type=int,
                        default=0,
                        help="number of used gpus")
    parser.add_argument("--n_nodes",
                        type=int,
                        default=1,
                        help="number of used nodes")
    parser.add_argument("--n_workers",
                        type=int,
                        default=4,
                        help="number of used workers for data loading")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="loaded checkpoint path")
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="mask radius for the partial autocorrelation version")
    parser.add_argument("--logdir",
                        type=str,
                        default="./tb_logs",
                        help="folder for the tensorboard logs")
    parser.add_argument("--imagedir",
                        type=str,
                        default="./Progress/",
                        help="folder for the image logs")
    parser.add_argument("--checkdir",
                        type=str,
                        default="./checkpoints",
                        help="folder for the saved checkpoints")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        help="choose the optimizer type")
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="outer loss function weight",
    )
    parser.add_argument("--partial", dest="partial", action="store_true")
    parser.add_argument("--no-partial", dest="partial", action="store_false")
    parser.set_defaults(partial=False)
    parser.add_argument("--zoom", dest="zoom", action="store_true")
    parser.add_argument("--no-zoom", dest="zoom", action="store_false")
    parser.set_defaults(zoom=False)
    parser.add_argument("--lr_find", dest="lr_find", action="store_true")
    parser.add_argument("--no-lr_find", dest="lr_find", action="store_false")
    parser.set_defaults(lr_find=False)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)
    # shrinking mask parameters
    parser.add_argument("--shrink", dest="shrink", action="store_true")
    parser.add_argument("--no-shrink", dest="shrink", action="store_false")
    parser.set_defaults(shrink=False)
    parser.add_argument("--shrink_epoch",
                        type=int,
                        default=500,
                        help="epoch where the radius shrinking starts")
    parser.add_argument("--shrink_step",
                        type=int,
                        default=10,
                        help="epochs to do one radius shrinking")
    parser.add_argument("--radius_step",
                        type=int,
                        default=1,
                        help="how much am I shrinking the radius each step?")
    # scheduler parameters
    parser.add_argument("--scheduler_milestone",
                        type=int,
                        default=100000,
                        help="epoch where we reduce the learning rate")
    parser.add_argument("--scheduler_gamma",
                        type=float,
                        default=0.1,
                        help="learning rate step")
    parser.add_argument("--one_cycle_lr",
                        type=float,
                        default=None,
                        help="max one cycle policy learning rate")

    opt = parser.parse_args()
    print(opt)

    # train
    main(opt)
