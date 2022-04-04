import torch
import os
import cv2
from PIL import Image

from utils import tensor2uint18


from log import logger

class Tester(object):
    def __init__(self, model, test_loader, config, suffix=""):
        self.device = config["DEVICE"]
        self.pale = config["PALETTE"]
        # exact model and optimizer
        self.model = model
        # Data loader
        self.test_loader = test_loader
        self.suffix = suffix
        self.test_batch_size = config["TEST_BATCH_SIZE"]
        self.gray_label_path = os.path.join(config["RESULTS_PATH"], 'gray{}'.format(self.suffix))
        self.color_label_path = os.path.join(config["RESULTS_PATH"], 'color{}'.format(self.suffix))
        self.making_files()

    def making_files(self):
        if not os.path.exists(self.gray_label_path):
              os.mkdir(self.gray_label_path)
              logger.info("New gray{} file!".format(self.suffixs))
        if not os.path.exists(self.color_label_path):
              os.mkdir(self.color_label_path)
              logger.info("New color{} file!".format(self.suffix))

    def test(self):
        "Compute performance of the model on the test dataset."
        self.model.eval()
        test_num = 0
        result_num = []
        with torch.inference_mode():
            for batch, imgs in enumerate(self.test_loader):
                # imgs: [batch, 3, imsize, imsize]
                # labels: [batch, 1, imsize, imsize]
                imgs = imgs.to(self.device)
                # outputs: [batch, num_class, imsize, imsize]
                outputs = self.model(imgs)
                pred_mask_tensor = torch.argmax(outputs, dim=1)
                pred_mask_numpy = pred_mask_tensor.cpu().numpy()
                for k in range(imgs.size(0)):
                    # gray_piciture
                    cv2.imwrite(os.path.join(self.gray_label_path, str(test_num + k) +'.png'), pred_mask_numpy[k])
                    # color_picture
                    color_label = tensor2uint18(pred_mask_tensor[k])
                    color_label = Image.fromarray(color_label)
                    color_label.putpalette(self.pale)
                    color_label.save(os.path.join(self.color_label_path, str(test_num + k) +'.png'))

                    result_num.append(test_num + k)
                test_num += imgs.size(0)

        logger.info("Testing {} results has completed!".format(test_num))
        return result_num