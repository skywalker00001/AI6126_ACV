import torch
import os
import segmentation_models_pytorch as smp

from parameters import Parameters
from tester import Tester
from data_loader import Data_Loader
from utils import (
    total_time,
    set_random_seeds,
)
from log import logger

# log
logger.info(("\n"))
logger.info("-----------------------------------------------------------------")
logger.info("testing.py")


if __name__ == '__main__':
    set_random_seeds(Parameters["SEED"])


    test_img_path = os.path.join(Parameters["TEST_PATH"], 'test_image')
    test_loader = Data_Loader(test_img_path, None, \
            Parameters["IMSIZE"], Parameters["TEST_BATCH_SIZE"], "test").loader()

    # Define model and optimizer
    loaded_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=Parameters["NUM_CLASSES"],
        ).to(Parameters["DEVICE"])


    epochs, mious = [], []
    num = Parameters["EXPECTED_MODEL_NUMBER"]
    for i in range(num):
        PATH = os.path.join(Parameters["MODEL_LOAD_PATH"], \
                                            "{}_MODEL{}.pth".format(i, Parameters["MODEL_LOAD_VERSION"]))
        checkpoint = torch.load(PATH)
        # loaded_model.load_state_dict(checkpoint['model_state_dict'])
        epochs.append(checkpoint['epoch'])
        mious.append(checkpoint['miou'])
    logger.info('best_epochs: {}'.format(str(epochs)))
    logger.info('best_mious: {}'.format(str(mious)))

    for i in range(Parameters["EXPECTED_MODEL_NUMBER"]):

        gray_label_path = os.path.join(Parameters["RESULTS_PATH"], 'gray{}'.format(i))
        color_label_path = os.path.join(Parameters["RESULTS_PATH"], 'color{}'.format(i))
        if not os.path.exists(gray_label_path):
            os.mkdir(gray_label_path)
            logger.info("New gray{} file!".format(i))
        if not os.path.exists(color_label_path):
            os.mkdir(color_label_path)
            logger.info("New color{} file!".format(i))

        LOAD_PATH = os.path.join(Parameters["MODEL_LOAD_PATH"], \
                                            "{}_MODEL{}.pth".format(i, Parameters["MODEL_LOAD_VERSION"]))
        load_checkpoint = torch.load(LOAD_PATH)
        loaded_model.load_state_dict(load_checkpoint['model_state_dict'])
        loaded_epoch = load_checkpoint['epoch']
        loaded_miou = load_checkpoint['miou']
        logger.info('loaded_epoch: {}'.format((loaded_epoch)))
        logger.info('loaded_miou: {}'.format(str(loaded_miou)))


        tester = Tester(loaded_model, test_loader, Parameters, i)
        result_list = tester.test()

        result_file = open(os.path.join(Parameters["RESULTS_PATH"], 'result_number_{}.txt'.format(i)),'w+')
        result_file.write('total_number: ' + str(len(result_list)))
        result_file.write('\n')
        result_file.write(str(result_list))
        result_file.close()

    gray_label_path = os.path.join(Parameters["RESULTS_PATH"], 'gray{}'.format("_New"))
    color_label_path = os.path.join(Parameters["RESULTS_PATH"], 'color{}'.format("_New"))
    if not os.path.exists(gray_label_path):
        os.mkdir(gray_label_path)
        logger.info("New gray{} file!".format("_New"))
    if not os.path.exists(color_label_path):
        os.mkdir(color_label_path)
        logger.info("New color{} file!".format("_New"))

    LOAD_PATH = os.path.join(Parameters["MODEL_LOAD_PATH"], \
                                        "FUNDAMODEL{}.pth".format(Parameters["MODEL_LOAD_VERSION"]))
    load_checkpoint = torch.load(LOAD_PATH)
    loaded_model.load_state_dict(load_checkpoint['model_state_dict'])
    loaded_epoch = load_checkpoint['epoch']
    loaded_miou = load_checkpoint['miou']
    logger.info('loaded_epoch: {}'.format(str(loaded_epoch)))
    logger.info('loaded_miou: {}'.format(str(loaded_miou)))


    tester = Tester(loaded_model, test_loader, Parameters, '_New')
    result_list = tester.test()

    result_file = open(os.path.join(Parameters["RESULTS_PATH"], 'result_number_{}.txt'.format("New")),'w+')
    result_file.write('total_number: ' + str(len(result_list)))
    result_file.write('\n')
    result_file.write(str(result_list))
    result_file.close()

    pt1 = Parameters["RESULTS_PATH"]
    logger.info(pt1)
    files = os.listdir(pt1)
    logger.info(files)
    