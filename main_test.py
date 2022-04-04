from parameters import Parameters
import os

from log import logger

# testing1 = Parameters["HH"]
# logger.info(testing1)

# if not os.path.exists(os.path.join(Parameters["ROOT"], "models")):
#         os.mkdir(os.path.join(Parameters["ROOT"], "models"))
#         logger.info("New {} file!".format("\'models\'"))

# if not os.path.exists(os.path.join(Parameters["ROOT"], "results")):
#         os.mkdir(os.path.join(Parameters["ROOT"], "results"))
#         logger.info("New {} file!".format("\'results\'"))

# if not os.path.exists(Parameters["MODEL_SAVE_PATH"]):
#         os.mkdir(Parameters["MODEL_SAVE_PATH"])
#         logger.info("New models file in \'{}\'!".format(Parameters["MODEL_SAVE_PATH"]))

# if not os.path.exists(Parameters["RESULTS_PATH"]):
#         os.mkdir(Parameters["RESULTS_PATH"])
#         logger.info("New models file in \'{}\'!".format(Parameters["RESULTS_PATH"]))

if __name__ == '__main__':
    # testing1 = Parameters["HH"]
    # logger.info(testing1)
    # logger.info("1")
    val_loss = 1.2223333
    val_miou = 2.33333444
    logger.info(f'\tValid Loss: {val_loss:.5f} | Valid Miou: {val_miou:.2f}%')