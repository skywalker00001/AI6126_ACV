# Import logging, get the logger, and set the processing level:

import logging
from parameters import Parameters
import os

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # process everything, even if everything isn't logger.infoed

# If you want to logger.info to stdout:
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) # or any other level
logger.addHandler(ch)

# If you want to also write to a file
fh = logging.FileHandler(Parameters["LOG_PATH"])
# fh.setLevel(logging.DEBUG) # or any level you want
fh.setLevel(logging.INFO)
# define the format to the file
# formatter1 = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
formatter1 = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s')
fh.setFormatter(formatter1)
logger.addHandler(fh)

# ERROR 40, WARNING 30, INFO 20, DEBUG 10.
# if logger.level < setlevel, will not write.

# Then, wherever you would use logger.info use one of the logger methods:

# logger.info(foo)

# foo = "xixi"
# logger.debug(foo)

# # logger.info('finishing processing')
# logger.info('finishing processing')

# # logger.info('Something may be wrong')
# logger.warning('Something may be wrong')

# # logger.info('Something is going really bad')
# logger.error('Something is going really bad')