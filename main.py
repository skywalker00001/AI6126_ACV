#import wandb
import random
from readline import parse_and_bind
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import cv2
import time
import wandb

import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch import cuda
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from parameters import Parameters
from utils import (
    total_time,
    set_random_seeds,
)
from trainer import Trainer
from data_loader import Data_Loader
from log import logger



if not os.path.exists(Parameters["MODEL_SAVE_PATH"]):
        os.mkdir(Parameters["MODEL_SAVE_PATH"])
        logger.info("New models file in \'{}\'!".format(Parameters["MODEL_SAVE_PATH"]))

if not os.path.exists(Parameters["RESULTS_PATH"]):
        os.mkdir(Parameters["RESULTS_PATH"])
        logger.info("New models file in \'{}\'!".format(Parameters["RESULTS_PATH"]))

set_random_seeds(Parameters["SEED"])

    
wandb.login()

if __name__ == '__main__':
    # log
    logger.info("\n")
    logger.info("-----------------------------------------------------------------")
    logger.info("main.py")
    logger.info("This is version: {}".format(Parameters["MODEL_VERSION"]))

    logger.info("DEVICE is: {}".format(Parameters["DEVICE"]))
    # set_random_seeds(Parameters["SEED"])
    
    # wandb.login()

    # load Data_loader

    train_img_path = os.path.join(Parameters["TRAIN_PATH"], 'train_image')
    train_label_path = os.path.join(Parameters["TRAIN_PATH"], 'train_mask')
    val_img_path = os.path.join(Parameters["VAL_PATH"], 'val_image')
    val_label_path = os.path.join(Parameters["VAL_PATH"], 'val_mask')
    # test_img_path = os.path.join(Parameters["TEST_PATH"], 'test_image')

    # train_loader: img([8, 3, 512, 512]), label([8, 1, 512, 512])
    train_loader = Data_Loader(train_img_path, train_label_path, \
                Parameters["IMSIZE"], Parameters["TRAIN_BATCH_SIZE"], "train").loader()
    val_loader = Data_Loader(val_img_path, val_label_path, \
                Parameters["IMSIZE"], Parameters["VAL_BATCH_SIZE"], "val").loader()
    # test_loader = Data_Loader(test_img_path, None, \
    #             Parameters["IMSIZE"], Parameters["TEST_BATCH_SIZE"], "test").loader()

    with wandb.init(
        project="Face_Parsing"+ Parameters["MODEL_VERSION"],
        ):
        # Define model and optimizer
        model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=Parameters["NUM_CLASSES"],
            ).to(Parameters["DEVICE"])

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                    Parameters["LEARNING_RATE"], [Parameters["BETA1"], Parameters["BETA2"]])
        start_time = time.time()

        # if need, load the paramters of the model
        if (Parameters["MODEL_IF_LOAD"]):      
            # update stored best_mious and best_epochs
            num = Parameters["EXPECTED_MODEL_NUMBER"]
            b_mious, b_epochs = [], []
            for i in range(num):
                PATH = os.path.join(Parameters["MODEL_LOAD_PATH"], \
                                                    "{}_MODEL{}.pth".format(i, Parameters["MODEL_LOAD_VERSION"]))
                checkpoint = torch.load(PATH)
                b_epochs.append(checkpoint['epoch'])
                b_mious.append(checkpoint['miou'])
            Parameters["BEST_MIOUS"] = b_mious
            Parameters["BEST_EPOCHS"] = b_epochs
            logger.info("\n")
            logger.info("loaded_best_mious: {}".format(str(Parameters["BEST_MIOUS"])))
            logger.info("loaded_best_epochs: {}".format(str(Parameters["BEST_EPOCHS"])))
            # really load the model and optimizer
            LOAD_PATH = os.path.join(Parameters["MODEL_LOAD_PATH"], \
                                                "FUNDAMODEL{}.pth".format(Parameters["MODEL_LOAD_VERSION"]))
            load_checkpoint = torch.load(LOAD_PATH)
            model.load_state_dict(load_checkpoint['model_state_dict'])
            optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
            Parameters["START_EPOCH"] = load_checkpoint['epoch']
            logger.info("\n")
            logger.info('Now the model is at epoch {} with miou {}.'.format(load_checkpoint['epoch'], load_checkpoint['miou']))
        
        trainer = Trainer(model, optimizer, train_loader, val_loader, Parameters)
        result_best_mious, result_best_epochs = trainer.train()
        logger.info("\n")
        logger.info("best_mious: {}".format(str(result_best_mious)))
        logger.info("best_epochs: {}".format(str(result_best_epochs)))
        
        end_time = time.time()
        epoch_mins, epoch_secs = total_time(start_time, end_time)
        logger.info("\n")
        logger.info(f'Total Time: {epoch_mins}m {epoch_secs}s')     