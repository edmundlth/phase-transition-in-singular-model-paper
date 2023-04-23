import logging
import sys
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

MCMCConfig = namedtuple(
    "MCMCConfig", ["num_posterior_samples", "num_warmup", "num_chains", "thinning"]
)


def start_log(logfile=None, loglevel=logging.INFO, log_name=None, log_to_stdout=True):
    """
    Set up a logger object for logging messages to a file and/or to the console.

    Parameters:
    -----------
    logfile : str or None, optional
        The name of the log file to write messages to. If None, messages will
        not be written to a file. Default is None.
    loglevel : int or str, optional
        The logging level to use for the logger object. Can be specified as an
        integer or as a string such as 'DEBUG', 'INFO', 'WARNING', etc. The
        default level is 'INFO'.
    log_name : str or None, optional
        The name to use for the logger object. If None, the name of the calling
        module (__name__) will be used. Default is None.
    log_to_stdout : bool, optional
        Whether to log messages to the console in addition to writing them to
        the log file. If True, messages will be logged to the console. If False,
        messages will be logged only to the log file. Default is True.

    Returns:
    --------
    logger : logging.Logger object
        A logger object that can be used to log messages to the file and/or
        the console.
    """
    if log_name is None:
        log_name = __name__
    print(f"LOGGER NAME: {log_name}")
    logger = logging.getLogger(log_name)
    logger.setLevel(loglevel)

    # Define the logging format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if logfile is not None:
        # Set up a file handler for logging to a file
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if log_to_stdout:
        # Set up a stream handler for logging to the console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(loglevel)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def linspaced_itemps_by_n(n, num_itemps):
    """
    Returns a 1D numpy array of length `num_itemps` that contains `num_itemps`
    evenly spaced inverse temperatures between the values calculated from the
    formula 1/log(n) * (1 - 1/sqrt(2log(n))) and 1/log(n) * (1 + 1/sqrt(2ln(n))).
    The formula is used in the context of simulating the behavior of a physical
    system at different temperatures using the Metropolis-Hastings algorithm.

    Parameters:
    -----------
    n : int
        The size of the system.
    num_itemps : int
        The number of inverse temperatures to generate.

    Returns:
    --------
    itemps : numpy.ndarray
        A 1D numpy array of length `num_itemps` containing the evenly spaced
        inverse temperatures.
    """
    return np.linspace(
        1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))),
        num_itemps,
    )


def compute_bayesian_loss(loglike_fn, X_test, Y_test, param_list):
    rec_array = jnp.hstack([loglike_fn(param, X_test, Y_test) for param in param_list])
    result = jnp.mean(
        jnp.exp(rec_array), axis=1
    )  # posterior predictive probability averaged over mcmc samples
    result = jnp.mean(
        -jnp.log(result)
    )  # negative log posterior, and averged over test samples
    return result


def compute_gibbs_loss(loglike_fn, X_test, Y_test, param_list):
    gerrs = []
    for i in range(len(param_list)):
        param = param_list[i]
        gibbs_err = np.mean(loglike_fn(param, X_test, Y_test))
        gerrs.append(gibbs_err)
    gg = np.mean(gerrs)
    return -gg
