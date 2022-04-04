import numpy as np
from PIL import Image
import wandb
import torch
import torch.nn as nn


# Set random seeds and deterministic pytorch for reproducibility
def set_random_seeds(SEED=42):
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = nn.functional.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = nn.functional.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def get_intersect_union(predict, groundtruth, num_classes=19):
    area_intersect_batch = np.zeros(num_classes)
    area_union_batch = np.zeros(num_classes)
    total_size = predict.shape[0]
    for idx in range(total_size):
        pred_mask = predict[idx]
        gt_mask = groundtruth[idx]
        for cls_idx in range(num_classes):
            area_intersect = np.sum(
                (pred_mask == gt_mask) * (pred_mask == cls_idx))

            area_pred_label = np.sum(pred_mask == cls_idx)
            area_gt_label = np.sum(gt_mask == cls_idx)
            area_union = area_pred_label + area_gt_label - area_intersect

            area_intersect_batch[cls_idx] += area_intersect
            area_union_batch[cls_idx] += area_union
    return area_intersect_batch, area_union_batch
    # iou_all = area_intersect_all / area_union_all * 100.0
    # miou = iou_all.mean()
    # return miou

def get_my_palette(impath):
    img = Image.open(impath) 
    palette = img.getpalette()
    return palette

def put_my_palette(img, pale):
    img = img.putpalette(pale)
    return img

# ts: [512, 512] after * 255
def tensor2uint18(ts):
    ts = ts.long().cpu().numpy()
    ts = ts.astype(np.uint8)
    return ts

# images: [batch, 3, 512, 512], tensor
# labels: [batch, 512, 512], tensor
def wandb_log_image_table(images, predicted, labels, pale):
    "Log a wandb.Table with (img, pred, target, scores)"
    table = wandb.Table(columns=["image", "pred", "target"])
    images, predicted, labels = images.cpu(), predicted.cpu(), labels.cpu()
    for img, pred, targ in zip(images, predicted, labels):
        # img
        img = img.long().numpy()
        img = np.transpose(img, (1, 2, 0))
        # pred
        pred = tensor2uint18(pred)
        pred = Image.fromarray(pred)
        pred.putpalette(pale)
        # targ
        targ = tensor2uint18(targ)
        targ = Image.fromarray(targ)
        targ.putpalette(pale)
        # add_data
        table.add_data(wandb.Image(img), wandb.Image(pred), wandb.Image(targ))
    wandb.log({"predictions_table":table}, commit=False)

def save_model(model, optimizer, epoch, miou, PATH):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'miou': miou
            }, PATH)
    
# Helper function to logger.info time 
def total_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs