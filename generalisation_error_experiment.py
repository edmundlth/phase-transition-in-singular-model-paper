import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import scipy

import haiku as hk
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import json
import pickle

import functools
import sys
import argparse
from src.haiku_numpyro_mlp import (
    build_forward_fn,
    build_model,
    generate_input_data,
    generate_output_data,
    run_mcmc,
    build_log_likelihood_fn,
    chain_wise_enlls,
    plot_rlct_regression,
)
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import start_log, linspaced_itemps_by_n
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
    logger = start_log(None, loglevel=logging.DEBUG, log_name=__name__)
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
            with_bias=args.with_true_bias,
        )
    )
    init_true_param = forward_true.init(next(rngkeyseq), X)
    if args.true_param_filepath is None:
        logger.info(
            "True parameter not specified. Randomly generating a new one based on provided model architecture."
        )
        true_param = init_true_param
    else:
        with open(args.true_param_filepath, "rb") as infile:
            true_param = pickle.load(infile)

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
            with_bias=args.with_bias,
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
    logger.info(f"Num mcmc samples={num_mcmc_samples}")

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
        "rng_seed": args.rng_seed, 
        "n": args.num_training_data, 
        "BL": float(b_loss), 
        "Bg": float(bg), 
    }

    if args.num_itemps is not None: 
        n = args.num_training_data
        assert n == len(X)
        assert n == len(Y)
        itemps = linspaced_itemps_by_n(args.num_training_data, num_itemps=args.num_itemps)
        logger.info(f"Sequence of itemps: {itemps}")
        log_likelihood_fn = functools.partial(
            build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
        )
        enlls = []
        stds = []
        for i_itemp, itemp in enumerate(itemps):
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
                itemp=itemp,
                progress_bar=(not args.quiet),
            )
            chain_enlls, chain_sizes = chain_wise_enlls(mcmc, treedef, log_likelihood_fn, X, Y)
            enll = np.sum(np.array(chain_enlls) * np.array(chain_sizes)) / np.sum(chain_sizes)
            chain_enlls_std = np.std(chain_enlls)
            logger.info(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
            logger.info(f"Across chain enll std: {chain_enlls_std}.")
            enlls.append(enll)
            stds.append(chain_enlls_std)
            if len(enlls) > 1:
                slope, intercept, r_val, _, _ = scipy.stats.linregress(
                    1 / itemps[: len(enlls)], enlls
                )
                logger.info(
                    f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}"
                )
        if args.plot_rlct_regression:
            fig, ax = plot_rlct_regression(itemps, enlls, n)
            filename = args.outfileprefix + "_rlct_regression.png"
            fig.savefig(filename)
            logger.info(f"RLCT regression figure saved at: {filename}")

        slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
        _map_float = lambda lst: list(map(float, lst)) # just to make sure things are json serialisable.
        result["rlct_estimation"] = {
            "n": n,
            "itemps": _map_float(itemps), 
            "enlls": _map_float(enlls),
            "chain_stds": _map_float(stds),
            "slope": float(slope), 
            "intercept": float(intercept),
            "r^2": float(r_val**2),
        }
    
    logger.info(json.dumps(result, indent=2))
    outfilename = args.outfileprefix + ".json"
    logger.info(f"Saving result JSON at: {outfilename}")
    with open(outfilename, "w") as outfile:
        json.dump(result, outfile) # no need for `indent` for such a small output. 
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT estimation of MLP models.")
    parser.add_argument("--true_param_filepath", nargs="?", default=None, type=str)
    parser.add_argument("--num-test-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-posterior-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num-chains", nargs="?", default=4, type=int)
    parser.add_argument("--num-training-data", nargs="?", default=1032, type=int)
    parser.add_argument("--num_itemps", nargs="?", default=None, type=int, help="If this is specified, the RLCT estimation procedure will be run.")
    parser.add_argument(
        "--plot_rlct_regression", action="store_true", default=False, help="If set, plot the RLCT regression figure."
    )
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
    parser.add_argument(
        "--with_bias", action="store_true", default=False, help="If set, the model network will use bias parameters."
    )
    parser.add_argument(
        "--with_true_bias", action="store_true", default=False, help="If set, the true network will use bias parameters."
    )
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=3.0, type=float)
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
