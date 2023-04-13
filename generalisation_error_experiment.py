import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np

import haiku as hk
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import os
import json

import functools
import sys
import argparse
from src.haiku_numpyro_mlp import (
    build_forward_fn,
    build_log_likelihood_fn,
    build_model,
    generate_input_data,
    generate_output_data,
    run_mcmc,
    expected_nll,
)
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import start_log
import logging


def log_likelihood(forward_fn, param, x, y, sigma=1.0):
    y_hat = forward_fn(param, None, x)
    ydist = dist.Normal(y_hat, sigma)
    return ydist.log_prob(y)


def compute_bayesian_loss(loglike_fn, X_test, Y_test, param_list):
    bgs = []
    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]
        lls = []
        for j in range(len(param_list)):
            param = param_list[j]
            lls.append(loglike_fn(param, x, y))
        bgs.append(-np.log(np.mean(np.exp(lls))))
    bg = np.mean(bgs)
    return bg


def main(args):
    logger = start_log(None, loglevel=logging.DEBUG)
    logger.info("Program starting...")
    logger.info(f"Commandline inputs: {args}")
    
    rngseed = args.rng_seed
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))

    # Construct true `forward_fn` and generate data X, Y
    input_dim = 1
    X = generate_input_data(
        args.num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    forward_true = hk.transform(
        build_forward_fn(
            layer_sizes=args.true_layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH["tanh"],
            initialisation_mean=args.prior_mean,
            initialisation_std=args.prior_std,
            with_bias=False,
        )
    )
    init_true_param = forward_true.init(next(rngkeyseq), X)
    true_flat, true_treedef = jtree.tree_flatten(init_true_param)
    true_param = init_true_param

    Y = generate_output_data(
        forward_true, true_param, X, next(rngkeyseq), sigma=args.sigma_obs
    )

    # Construct `forward` for model, numpyro `model` and log_likelihood_fn

    forward = hk.transform(
        build_forward_fn(
            layer_sizes=args.layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH["tanh"],
            initialisation_mean=args.prior_mean,
            initialisation_std=args.prior_std,
            with_bias=False,
        )
    )
    init_param = forward.init(next(rngkeyseq), X)
    _, treedef = jtree.tree_flatten(init_param)
    param_center = init_param
    model = functools.partial(
        build_model,
        forward.apply,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        sigma=args.sigma_obs,
    )

    loglike_fn = jax.jit(
        functools.partial(log_likelihood, forward.apply, sigma=args.sigma_obs)
    )
    mcmc = run_mcmc(
        model,
        X,
        Y,
        next(rngkeyseq),
        param_center,
        num_posterior_samples=args.num_posterior_samples,
        num_warmup=args.num_warmup,
        num_chains=args.num_chains,
        thinning=args.thinning,
        itemp=1.0,
        progress_bar=True,
    )
    posterior_samples = mcmc.get_samples()
    num_mcmc_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
    print(f"Num mcmc samples={num_mcmc_samples}")

    param_list = [
        treedef.unflatten(
            [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
        )
        for i in range(num_mcmc_samples)
    ]

    X_test = generate_input_data(
        args.num_test_samples, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    Y_test = generate_output_data(
        forward_true, true_param, X_test, next(rngkeyseq), sigma=args.sigma_obs
    )
    b_loss = compute_bayesian_loss(loglike_fn, X_test, Y_test, param_list)
    bg = b_loss + np.mean(
        log_likelihood(
            forward_true.apply, true_param, X_test, Y_test, sigma=args.sigma_obs
        )
    )

    result = {
        "n": args.num_training_data, 
        "BL": float(b_loss), 
        "Bg": float(bg), 
    }
    print(json.dumps(result, indent=2))
    outfilename = args.outfileprefix + ".json"
    with open(outfilename, "w") as outfile:
        json.dump(result, outfile, indent=4)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT estimation of MLP models.")
    parser.add_argument("--num-test-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-posterior-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num-chains", nargs="?", default=4, type=int)
    parser.add_argument("--num-training-data", nargs="?", default=1032, type=int)
    parser.add_argument(
        "--input_dim",
        nargs="?",
        default=1,
        type=int,
        help="Dimension of the input data X.",
    )

    parser.add_argument(
        "--layer_sizes",
        nargs="+",
        type=int,
        default=[1, 1],
        help="A list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. ",
    )
    parser.add_argument(
        "--true_layer_sizes",
        nargs="+",
        type=int,
        default=[1, 1],
        help="Same as --layer_sizes for for the true model. If not specified, values for --layer_sizes are used. ",
    )
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation-fn-name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default=None, type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--host_device_count",
        default=None,
        type=int,
        help="Number of host device for parallel run of MCMC chains.",
    )
    parser.add_argument(
        "--outfileprefix",
        default="result",
        type=str,
        help="output filename prefix",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--rng-seed", nargs="?", default=42, type=int)

    args = parser.parse_args()

    
    if args.host_device_count is not None:
        numpyro.set_host_device_count(args.host_device_count)
    else:
        numpyro.set_host_device_count(args.num_chains)

    jax_platform = jax.lib.xla_bridge.get_backend().platform
    if args.device is not None:
        numpyro.set_platform(args.device)
    else:
        numpyro.set_platform(jax_platform)
    print("jax backend:", jax_platform)
    print(f"JAX devices (num={jax.device_count()}): {jax.devices()}")
    main(args)
