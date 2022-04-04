from torch import cuda
import torch
import numpy as np
import os

from utils import get_my_palette


# ROOT = 'drive/MyDrive/ACV/Project1'
ROOT = "/Users/yihou/Documents/projects/PYCodes/acv/project1"

# # Setting up the device for GPU usage
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
# logger.info("DEVICE is: ".format(DEVICE))

# Set random seeds and deterministic pytorch for reproducibility
SEED = 42
# torch.manual_seed(SEED) # pytorch random seed
# np.random.seed(SEED) # numpy random seed
# torch.backends.cudnn.deterministic = True


Parameters = {
    "ROOT": ROOT,
    "DEVICE": DEVICE,
    "SEED": SEED,

    "MODEL_VERSION": '5.0',
    "MODEL_LOAD_VERSION": '5.0',
    "NUM_EPOCH": 10,
    "START_EPOCH": 0,
    "NUM_CLASSES": 19,
    "IMSIZE": 512,
    "TRAIN_BATCH_SIZE": 16,    # input batch size for training (default: 64)
    "VAL_BATCH_SIZE": 64,    # input batch size for testing (default: 1000)
    "TEST_BATCH_SIZE": 64, 
    # Model Para
    "LEARNING_RATE": 2e-4 ,   # learning rate (default: 0.01)
    "LR_DECAY": 0.95,
    "BETA1": 0.5,
    "BETA2": 0.999,
    # Path
    "TRAIN_PATH": os.path.join(ROOT, 'train'),
    "VAL_PATH": os.path.join(ROOT, 'val'),
    "TEST_PATH": os.path.join(ROOT, 'test'),
    "RESULTS_PATH": os.path.join(ROOT, 'results'),
    "LOG_PATH": os.path.join(ROOT, 'codes/myLog.log'),
    # Save
    "MODEL_SAVE_STEP": 5,
    "MODEL_SAVE_PATH": os.path.join(ROOT, 'models'),
    # Load
    "MODEL_IF_LOAD": False,
    "MODEL_LOAD_PATH": os.path.join(ROOT, 'models'),
    # Best model
    "EXPECTED_MODEL_NUMBER": 2,
    "LOG_IMAGES": False,
    "SMOOTHING": 0,
}

Parameters["BEST_MIOUS"] = Parameters["EXPECTED_MODEL_NUMBER"] * [0]
Parameters["BEST_EPOCHS"] = Parameters["EXPECTED_MODEL_NUMBER"] * [0]
# get palette
Parameters["PALETTE"] = get_my_palette(os.path.join(Parameters["TRAIN_PATH"], 'train_mask/1.png'))