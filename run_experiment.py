import argparse
import os
import time
import pickle
import functools
import json

import jax
import jax.tree_util as jtree
import numpy as np
import haiku as hk
import numpyro

from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import start_log, MCMCConfig
from src.haiku_numpyro_mlp import (
    build_forward_fn,
    build_log_likelihood_fn,
    build_model,
    generate_input_data,
    generate_output_data,
    run_mcmc,
    chain_wise_enlls,
    expected_nll,
)

import logging

logger = logging.getLogger(__name__)


def program_initialisation(args):
    outdirpath = args.output_dir
    if os.path.exists(outdirpath):
        logging.warn(f"WARNING: Output directory path already exist: {outdirpath}")
    os.makedirs(outdirpath, exist_ok=True)

    def make_filepath_fn(filename):
        return os.path.join(outdirpath, filename)

    logfilepath = make_filepath_fn(args.logfilename)
    logger = start_log(logfilepath, loglevel=logging.DEBUG, log_name=__name__)
    logger.info("Program starting...")
    logger.info(f"Commandline inputs: {args}")
    start_time = time.time()
    return logger, make_filepath_fn, start_time


def main(expt_config, args):
    logger, make_filepath_fn, start_time = program_initialisation(args)
    logger.info(f"Available devices (n={jax.device_count()}): {jax.devices()}")
    logger.info(f"Experiment configuration: {json.dumps(expt_config, indent=2)}")

    rngseed = expt_config["rng_seed"]
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))

    # Construct true `forward_fn` and generate data X, Y
    truth_config = expt_config["truth"]["model_args"]
    true_layer_sizes = truth_config["layer_sizes"]
    input_dim = expt_config["input_dim"]

    num_training_data = expt_config["num_training_data"]
    X = generate_input_data(
        num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    forward_true = hk.transform(
        build_forward_fn(
            layer_sizes=true_layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH[
                truth_config["activation_fn_name"].lower()
            ],
            with_bias=truth_config["with_bias"],
        )
    )
    init_true_param = forward_true.init(jax.random.PRNGKey(0), X)

    true_param_filepath = truth_config["param_filepath"]
    if true_param_filepath is None:
        logger.info(
            "True parameter not specified. Randomly generating a new one based on provided model architecture."
        )
        true_param = init_true_param
    else:
        with open(true_param_filepath, "rb") as infile:
            true_param = pickle.load(infile)
    logger.info(f"True parameter:{true_param}")

    Y = generate_output_data(
        forward_true, true_param, X, next(rngkeyseq), sigma=truth_config["sigma_obs"]
    )
    if args.save_training_data:
        filepath = make_filepath_fn("training_data.npz")
        np.savez_compressed(filepath, X=X, Y=Y)
        logger.info(f"Training data saved at: {filepath}")

    # Construct `forward` for model, numpyro `model` and log_likelihood_fn
    model_config = expt_config["model"]["model_args"]
    sigma_obs = model_config["sigma_obs"]
    prior_name = expt_config["prior"]["distribution_name"]
    if prior_name.lower() != "normal":  # TODO: only implementing normal for now.
        raise NotImplementedError("Only normal prior implemented.")
    prior_config = expt_config["prior"]["distribution_args"]
    prior_mean = prior_config["loc"]
    prior_std = prior_config["scale"]
    layer_sizes = model_config["layer_sizes"]
    forward = hk.transform(
        build_forward_fn(
            layer_sizes=layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH[
                model_config["activation_fn_name"].lower()
            ],
            initialisation_mean=prior_mean,
            initialisation_std=prior_std,
            with_bias=model_config["with_bias"],
        )
    )
    init_param = forward.init(next(rngkeyseq), X)
    _, treedef = jtree.tree_flatten(init_param)
    param_center = init_param
    log_likelihood_fn = functools.partial(
        build_log_likelihood_fn, forward.apply, sigma=sigma_obs
    )
    model = functools.partial(
        build_model,
        forward.apply,
        param_center=param_center,
        prior_mean=prior_mean,
        prior_std=prior_std,
        sigma=sigma_obs,
    )

    # Run MCMC sampling at the specified `itemp`
    itemp = expt_config["itemp"]
    mcmc_config = MCMCConfig(**expt_config["mcmc_config"])
    mcmc = run_mcmc(
        model,
        [X, Y],
        next(rngkeyseq),
        mcmc_config,
        itemp=itemp,
        progress_bar=(not args.quiet),
    )

    # Compute Expected negative log likelihood
    chain_enlls, chain_sizes = chain_wise_enlls(mcmc, treedef, log_likelihood_fn, X, Y)
    enll = np.sum(np.array(chain_enlls) * np.array(chain_sizes)) / np.sum(chain_sizes)
    chain_enlls_std = np.std(chain_enlls)
    logger.info(f"Expected NLL at temp={1/itemp:.3f} is {enll:.3f}.")
    logger.info(f"Across chain enll std: {chain_enlls_std}.")
    time_taken = time.time() - start_time
    # Save experiment output and diagnostics.
    expt_config["output"].update(
        {
            "output_directory": args.output_dir,
            "status": 0,
            "enll": float(enll),  # json doesn't know how to serialise float32
            "commandline_args": vars(args),
            "chain_enlls": list(
                map(float, chain_enlls)
            ),  # json doesn't know how to serialise float32
            "chain_sizes": chain_sizes,
            "chain_enlls_std": float(chain_enlls_std),
            "wall_time_taken": time_taken,
        }
    )

    outfilename = make_filepath_fn("result.json")
    with open(outfilename, "w") as outfile:
        json.dump(expt_config, outfile, indent=4)
    logger.info(f"Result JSON saved at: {outfilename}")

    if args.save_posterior_samples:
        posterior_samples = mcmc.get_samples(group_by_chain=True)
        filepath = make_filepath_fn("posterior_samples.npz")
        np.savez_compressed(filepath, **posterior_samples)
        logger.info(f"Posterior samples saved at: {filepath}")

    logger.info(f"Program successfully finished. Time taken: {time_taken:.2f} seconds")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single experiment to obtain MCMC estimate of the expected NLL of a specified truth-data-model-prior configuration at a speicified inverse temperature."
    )
    parser.add_argument(
        "--config_filepath",
        default=None,
        type=str,
        help="Path to experiment configuration JSON-file",
    )
    parser.add_argument(
        "--config_index",
        default=None,
        type=int,
        help="If not specified, JSON object in --config_filepath is the experiment configuration itself. If specified, treat the JSON object in --config_filepath as a JSON list. The experiment config is the object at --config_index. ",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="a directory for storing output files.",
    )
    parser.add_argument(
        "--save_posterior_samples",
        action="store_true",
        default=False,
        help="If specified, posterior samples will be saved in the output directory.",
    )
    parser.add_argument(
        "--save_training_data",
        action="store_true",
        default=False,
        help="If specified, the training data will be saved in the output directory.",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help='use "cpu" or "gpu". If not specified, use the platform detected by JAX',
    )
    parser.add_argument(
        "--host_device_count",
        default=None,
        type=int,
        help="Number of host device for parallel run of MCMC chains.",
    )
    parser.add_argument(
        "--logfilename",
        default="program.log",
        type=str,
        help="The name of the file that store program logs. This file will be stored in the output directory.",
    )

    args = parser.parse_args()
    # read in experiment config.
    with open(args.config_filepath) as config_file:
        expt_config = json.load(config_file)

    if args.config_index is not None:
        expt_config = expt_config[args.config_index]

    if args.host_device_count is not None:
        numpyro.set_host_device_count(args.host_device_count)
    else:
        numpyro.set_host_device_count(expt_config["mcmc_config"]["num_chains"])

    jax_platform = jax.lib.xla_bridge.get_backend().platform
    if args.device is not None:
        numpyro.set_platform(args.device)
    else:
        numpyro.set_platform(jax_platform)
    print("jax backend:", jax_platform)
    print(f"JAX devices (num={jax.device_count()}): {jax.devices()}")
    main(expt_config, args)
