import time
from tqdm import tqdm
import numpy as np
import wandb
import torch
import os

from log import logger
from utils import (
    cross_entropy2d,
    get_intersect_union,
    total_time,
    save_model,
    wandb_log_image_table,
)

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        self.model_version = config["MODEL_VERSION"]
        self.device = config["DEVICE"]
        self.pale = config["PALETTE"]
        # Data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # exact model and optimizer
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = config["NUM_EPOCH"]
        self.start_epoch = config["START_EPOCH"]
        self.num_classes = config["NUM_CLASSES"]
        # Save model
        self.model_save_step = config["MODEL_SAVE_STEP"]
        self.model_save_path = config["MODEL_SAVE_PATH"]
        self.best_mious = config["BEST_MIOUS"]
        self.best_epochs = config["BEST_EPOCHS"]
        self.need_log_images = config["LOG_IMAGES"]
        self.smoothing = config["SMOOTHING"]
        

    def train(self):

        for epoch in range(self.start_epoch, self.start_epoch+ self.num_epoch):
            total_epoch_start = time.time()
            self.model.train()
            train_loss = 0
            train_miou = 0
            train_num = 0
            # train_pred = np.array([])
            # train_label = np.array([])
            train_intersect_all = np.zeros(self.num_classes)
            train_union_all = np.zeros(self.num_classes)
            val_loss = 0
            val_miou = 0
            train_epoch_start = time.time()
            # train
            with tqdm(total=len(self.train_loader), desc="training progress bar") as progress_bar:
                progress_bar.set_description('Epoch: {}/{} training'.format(epoch+1, self.start_epoch+ self.num_epoch ))
                for batch, (imgs, labels) in enumerate(self.train_loader):
                    # imgs: [batch, 3, imsize, imsize]
                    # labels: [batch, 1, imsize, imsize]
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    # Forward
                    # outputs: [batch, num_class, imsize, imsize]
                    outputs = self.model(imgs)
                    # labels_real_plain: [batch, imsize, imsize]
                    labels_real_plain = labels[:, 0, :, :] * 255.0
                    # compute loss
                    loss = cross_entropy2d(outputs, labels_real_plain.long())
                    train_loss += loss.item() * imgs.shape[0]
                    # Backprop the gradient and update parameters
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_num += imgs.shape[0]  
                    # compute miou
                    # pred_mask: [batch, imsize, imsize]
                    pred_mask = torch.argmax(outputs, dim=1)
                    train_intersect_batch, train_union_batch = get_intersect_union(pred_mask.long().cpu().numpy(), \
                                      labels_real_plain.long().cpu().numpy(), num_classes=self.num_classes)
                    train_intersect_all += train_intersect_batch
                    train_union_all += train_union_batch
                    # # Cat the pred and label
                    # train_pred = np.concatenate((train_pred, outputs.long().cpu().numpy()), axis=0)
                    # # logger.info("train_pred: ".format(str(train_pred.shape)))
                    # train_label = np.concatenate((train_label, labels_real_plain.long().cpu().numpy()), axis=0)

                    if(batch) % 10 == 9:
                        progress_bar.set_postfix(batch='{}'.format(batch), \
                                                 train_loss='{:.5f}'.format(train_loss / train_num))
                    progress_bar.update(1)

            train_epoch_end = time.time()
            train_epoch_mins, train_epoch_secs = total_time(train_epoch_start, train_epoch_end)
            # Compute miou
            train_iou_all = train_intersect_all / train_union_all * 100.0
            train_miou = train_iou_all.mean()
            # Log validation metrics
            val_epoch_start = time.time()
            val_loss, val_miou = self.valid(epoch, log_images=self.need_log_images, batch_idx=0)
            val_epoch_end = time.time()
            val_epoch_mins, val_epoch_secs = total_time(val_epoch_start, val_epoch_end)
            # update wandb
            wandb.log({"train_loss": (train_loss / train_num), "train_miou": train_miou , 
                          "val_loss": val_loss, "val_miou": val_miou, "epoch": epoch+1}, step=epoch - self.start_epoch)        
            # Save model based on miou
            min_best_mious = min(self.best_mious)
            indexof_min_best_mious = self.best_mious.index(min(self.best_mious))
            if val_miou > min_best_mious:   # replace the model which has the min miou with current model.
                self.best_mious[indexof_min_best_mious] = val_miou
                self.best_epochs[indexof_min_best_mious] = epoch
                save_model(self.model, self.optimizer, epoch+1, val_miou, \
                           os.path.join(self.model_save_path, '{}_MODEL{}.pth').format(indexof_min_best_mious, self.model_version))
                logger.info("Saving best model at epoch {} with miou {}".format(epoch+1, val_miou))
            # Save model based on epoch
            if epoch % self.model_save_step == (self.model_save_step - 1):   # each 10 epochs save the model once.
                save_model(self.model, self.optimizer, epoch+1, val_miou, \
                           os.path.join(self.model_save_path, 'FUNDAMODEL{}.pth').format(self.model_version))
                logger.info("Saving fundamodel at epoch {} with miou {}".format(epoch+1, val_miou))

            total_epoch_end = time.time()
            total_epoch_mins, total_epoch_secs = total_time(total_epoch_start, total_epoch_end)           
            logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {total_epoch_mins}m {total_epoch_secs}s')
            logger.info(f'Train Time: {train_epoch_mins}m {train_epoch_secs}s | Val Time: {val_epoch_mins}m {val_epoch_secs}s')
            logger.info(f'\tTrain Loss: {(train_loss / train_num):.5f} | Train Miou: {train_miou:.2f}%')
            logger.info(f'\tValid Loss: {val_loss:.5f} | Valid Miou: {val_miou:.2f}%')
            logger.info("\n")
        return self.best_mious ,self.best_epochs

    def valid(self, epoch, log_images=False, batch_idx=0):
        "Compute performance of the model on the validation dataset and log a wandb.Table"
        self.model.eval()
        val_loss = 0.
        val_miou = 0.
        # val_pred = torch.tensor([]).to(self.device)
        # val_label = torch.tensor([]).to(self.device)
        val_intersect_all = np.zeros(self.num_classes)
        val_union_all = np.zeros(self.num_classes)
        val_num = 0
        with torch.inference_mode():
            with tqdm(total=len(self.val_loader), desc="validating progress bar") as progress_bar:
                progress_bar.set_description('epoch: {}/{} validating'.format(epoch+1, self.start_epoch+ self.num_epoch))
                for batch, (imgs, labels) in enumerate(self.val_loader):
                    # imgs: [batch, 3, imsize, imsize]
                    # labels: [batch, 1, imsize, imsize]
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    # outputs: [batch, num_class, imsize, imsize]
                    outputs = self.model(imgs)
                    # *255 to restore the real pixel value (transform.ToTensor has / 255)
                    # labels_real_plain: [batch, imsize, imsize]
                    labels_real_plain = labels[:, 0, :, :] * 255.0
                    # compute loss (average loss over one batch)
                    loss = cross_entropy2d(outputs, labels_real_plain.long())
                    val_loss += loss.item() * imgs.shape[0]
                    val_num += imgs.shape[0]  
                    # compute miou
                    # pred_mask: [batch, imsize, imsize]
                    pred_mask = torch.argmax(outputs, dim=1)
                    val_intersect_batch, val_union_batch = get_intersect_union(pred_mask.long().cpu().numpy(), \
                                      labels_real_plain.long().cpu().numpy(), num_classes=self.num_classes)
                    val_intersect_all += val_intersect_batch
                    val_union_all += val_union_batch
                    # Log one batch of images to the dashboard, always same batch_idx.
                    if (epoch % self.model_save_step == (self.model_save_step - 1)) and batch==batch_idx and log_images:
                        wandb_log_image_table(imgs*255, pred_mask, labels_real_plain, pale=self.pale)
                    # update progress_bar
                    progress_bar.set_postfix(loss='{:.5f}'.format(val_loss / val_num))
                    progress_bar.update(1)

        # val_pred = np.argmax(val_pred, axis=1)
        # val_miou = get_all_miou(val_pred, val_label)
        # Compute miou
        val_iou_all = val_intersect_all / val_union_all * 100.0
        val_miou = val_iou_all.mean()
        return val_loss / val_num, val_miou