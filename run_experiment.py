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

from const import ACTIVATION_FUNC_SWITCH
from haiku_mlp_rlct_estimate import (
    build_forward_fn,
    build_log_likelihood_fn,
    build_model,
    generate_input_data,
    generate_output_data,
    run_mcmc, 
    expected_nll
)


def main(expt_config, args):
    start_time = time.time()

    rngseed = expt_config["rng_seed"]
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))

    # Construct true `forward_fn` and generate data X, Y
    truth_config = expt_config["truth"]["model_args"]
    true_layer_sizes = truth_config["layer_sizes"]
    input_dim = true_layer_sizes[0]
    output_dim = true_layer_sizes[-1]
    with open(truth_config["param_filepath"], "rb") as infile:
        true_param = pickle.load(infile)
    print(true_param)

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
            with_bias=truth_config["with_bias"]
        )
    )
    true_init = forward_true.init(jax.random.PRNGKey(0), X)
    Y = generate_output_data(
        forward_true, true_param, X, next(rngkeyseq), sigma=truth_config["sigma_obs"]
    )

    # Construct `forward` for model, numpyro `model` and log_likelihood_fn
    model_config = expt_config["model"]["model_args"]
    sigma_obs = model_config["sigma_obs"]
    prior_name = expt_config["prior"]["distribution_name"] 
    if prior_name.lower() != "normal": # TODO: only implementing normal for now.
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
            with_bias=model_config["with_bias"]
        )
    )
    init_param = forward.init(next(rngkeyseq), X)
    init_param_flat, treedef = jtree.tree_flatten(init_param)
    param_center = init_param
    log_likelihood_fn = functools.partial(
        build_log_likelihood_fn, forward.apply, sigma=sigma_obs
    )
    model = functools.partial(build_model, forward.apply)
    itemp = expt_config["itemp"]
    mcmc_config = expt_config["mcmc_config"]
    mcmc = run_mcmc(
        model,
        X,
        Y,
        next(rngkeyseq),
        param_center,
        prior_mean,
        prior_std,
        sigma_obs,
        num_posterior_samples=mcmc_config["num_posterior_samples"],
        num_warmup=mcmc_config["num_warmup"],
        num_chains=mcmc_config["num_chains"],
        thinning=mcmc_config["thinning"],
        itemp=itemp,
        progress_bar=(not args.quiet),
    )
    posterior_samples = mcmc.get_samples()
    num_mcmc_samples = len(
        posterior_samples[list(posterior_samples.keys())[0]]
    )
    param_list = [
        [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
        for i in range(num_mcmc_samples)
    ]
    enll = expected_nll(log_likelihood_fn, map(treedef.unflatten, param_list), X, Y)
    print(f"Finished temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
    
    # save to output directory and record directory full path.
    outdirpath = args.output_dir
    if os.path.exists(outdirpath):
        print(f"WARNING: Output directory path already exist: {outdirpath}")
    if not os.path.isdir(outdirpath):
        os.mkdir(outdirpath)
    expt_config["output"]["output_directory"] = outdirpath
    # update experiment status.
    expt_config["output"]["status"] = 0
    expt_config["output"]["enll"] = float(enll) # json doesn't know how to serialise float32 
    expt_config["output"]["commandline_args"] = vars(args)
    expt_config["output"]["wall_time_taken"] = time.time() - start_time

    outfilename = os.path.join(outdirpath, "result.json")
    with open(outfilename, 'w') as outfile:
        json.dump(expt_config, outfile, indent=4)

    if args.save_posterior_samples:
        filepath = os.path.join(outdirpath, "posterior_samples.npz")
        np.savez(filepath, **posterior_samples)
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
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--host_device_count",
        default=None,
        type=int,
        help="Number of host device for parallel run of MCMC chains.",
    )

    args = parser.parse_args()
    # read in experiment config.
    with open(args.config_filepath) as config_file:
         expt_config = json.load(config_file)

    numpyro.set_platform(args.device)    
    if args.host_device_count is None:
        numpyro.set_host_device_count(args.host_device_count)
    else:
        numpyro.set_host_device_count(expt_config["mcmc_config"]["num_chains"])

    print(expt_config)
    print(args)
    main(expt_config, args)
