import logging
import sys

def start_log(logfile=None, loglevel=logging.INFO, log_name=None):
    # Set up logging
    if log_name is None:
        log_name = __name__
    logger = logging.getLogger(log_name)
    logger.setLevel(loglevel)

    # Define the logging format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if logfile is not None:
        # Set up a file handler for logging to a file
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Set up a stream handler for logging to the console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger