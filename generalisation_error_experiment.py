import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import scipy

import haiku as hk
import numpyro
import numpyro.distributions as dist
import json
import pickle

import functools
import os
import argparse
from src.haiku_numpyro_mlp import (
    build_forward_fn,
    build_model,
    generate_input_data,
    generate_output_data,
    run_mcmc,
    build_log_likelihood_fn,
    plot_rlct_regression,
    rlct_estimate_regression,
)
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import (
    start_log,
    linspaced_itemps_by_n,
    compute_bayesian_loss,
    compute_gibbs_loss,
    compute_functional_variance, 
    compute_waic, 
    compute_wbic,
    MCMCConfig,
)
import logging

logger = logging.getLogger("__main__")


def log_likelihood(forward_fn, param, x, y, sigma=1.0):
    y_hat = forward_fn(param, None, x)
    ydist = dist.Normal(y_hat, sigma)
    return ydist.log_prob(y)


def load_synthetic_data(args, rngkey, generate_test_data=False):
    rngkeyseq = hk.PRNGSequence(rngkey)
    # Construct true `forward_fn` and generate data X, Y
    input_dim = args.input_dim
    X = generate_input_data(
        args.num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq), xmin=-1.0, xmax=1.0
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

    if generate_test_data:
        X_test = generate_input_data(
            args.num_test_samples, input_dim=args.input_dim, rng_key=next(rngkeyseq)
        )
        Y_test = generate_output_data(
            forward_true, true_param, X_test, next(rngkeyseq), sigma=args.sigma_obs
        )
        return forward_true, true_param, X, Y, X_test, Y_test
    else:
        return forward_true, true_param, X, Y


def load_data(file_path, train_size, test_size, random_seed=None):
    # Load data from the .npz file
    data = np.load(file_path)
    X = data["X"]
    Y = data["Y"]
    
    n = len(X)
    assert n == len(Y), "Number of input samples is not the same as output samples."
    assert n >= train_size + test_size, "Training dataset and test dataset is overlaping"

    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the data using a random permutation
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    # Split the data into training and test sets
    X_train, X_test = X[:train_size], X[-test_size:]
    Y_train, Y_test = Y[:train_size], Y[-test_size:]

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {Y_train.shape}")
    logger.info(f"y_test shape: {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test


def main(args, rngseed):
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

    loglike_fn = jax.jit(
        functools.partial(log_likelihood, forward.apply, sigma=args.sigma_obs)
    )
    mcmc_config = MCMCConfig(
        args.num_posterior_samples, args.num_warmup, args.num_chains, args.thinning
    )
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))
    if args.datafilepath is not None:
        X, Y, X_test, Y_test = load_data(
            args.datafilepath,
            train_size=args.num_training_data,
            test_size=args.num_test_samples,
            random_seed=rngseed,
        )
        forward_true, true_param = None, None
    else:
        forward_true, true_param, X, Y, X_test, Y_test = load_synthetic_data(
            args, next(rngkeyseq), generate_test_data=True
        )
    
    init_param = forward.init(next(rngkeyseq), X)
    _, treedef = jtree.tree_flatten(init_param)
    param_center = init_param
    model = functools.partial(
        build_model,
        forward.apply,
        param_center=param_center,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        sigma=args.sigma_obs,
    )
    mcmc = run_mcmc(
        model,
        [X, Y],
        next(rngkeyseq),
        mcmc_config,
        itemp=1.0,
        progress_bar=(not args.quiet),
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

    
    loglike_array = np.hstack(# dimension = (num test samples, num mcmc samples)
        [loglike_fn(param, X_test, Y_test) for param in param_list]
    )
    gibbs_loss = compute_gibbs_loss(loglike_array)
    logger.info(f"Gibbs loss: {gibbs_loss}")

    bayes_loss = compute_bayesian_loss(loglike_array)
    logger.info(f"Bayes loss: {bayes_loss}")
    
    # This is loglikelihood array for training data now. 
    # It overwrites the previous loglike_array to free up memory.
    loglike_array = np.hstack( 
        [loglike_fn(param, X, Y) for param in param_list]
    )
    
    gibbs_train_loss = compute_gibbs_loss(loglike_array)
    logger.info(f"Gibbs train loss: {gibbs_train_loss}")

    bayes_train_loss = compute_bayesian_loss(loglike_array)
    logger.info(f"Bayes train loss: {bayes_train_loss}")

    func_var = compute_functional_variance(loglike_array)
    logger.info(f"Functional Variance: {func_var}")

    waic = compute_waic(loglike_array)
    logger.info(f"WAIC: {waic}")
    result = {
        "commandline_args": vars(args), 
        "rng_seed": rngseed,
        "n": args.num_training_data,
        "BL": float(bayes_loss),
        "BLt": float(bayes_train_loss),
        "GL": float(gibbs_loss),
        "GLt": float(gibbs_train_loss),
        "V_n": float(func_var), 
        "WAIC": float(waic),
    }

    if not args.no_wbic:
        itemp = 1 / np.log(args.num_training_data)
        tempered_mcmc = run_mcmc(
            model,
            [X, Y],
            next(rngkeyseq),
            mcmc_config,
            itemp=itemp,
            progress_bar=(not args.quiet),
        )
        posterior_samples = tempered_mcmc.get_samples()
        num_mcmc_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
        tempered_posterior_param_list = [
            treedef.unflatten(
                [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
            )
            for i in range(num_mcmc_samples)
        ]
        # this is now a tempered loglikelihood array on training data. 
        # Again, this overwrites the previous variable to save memory. 
        loglike_array = np.hstack( 
            [loglike_fn(param, X, Y) for param in tempered_posterior_param_list]
        )
        wbic = compute_wbic(loglike_array)
        result["WBIC"] = float(wbic)
        logger.info(f"WBIC: {wbic}")

    if forward_true is not None and true_param is not None:
        truth_entropy = -np.mean(
            log_likelihood(
                forward_true.apply, true_param, X_test, Y_test, sigma=args.sigma_obs
            )
        )
        truth_entropy_train = -np.mean(
            log_likelihood(
                forward_true.apply, true_param, X, Y, sigma=args.sigma_obs
            )
        )
        bayes_error = bayes_loss - truth_entropy
        gibbs_error = gibbs_loss - truth_entropy
        result["truth_entropy"] = float(truth_entropy)
        result["truth_entropy_train"] = float(truth_entropy_train)
        result["Gg"] = float(gibbs_error)
        result["Bg"] = float(bayes_error)

        if not args.no_true_param_output:
            # store the true parameter as list of lists. 
            true_param_list, _ = jtree.tree_flatten(true_param)
            result["true_param"] = [np.array(p).tolist() for p in true_param_list]

    logger.info(f"Finished generalisation error calculation: {json.dumps(result)}")
    # Saving the result first. 
    # If the RLCT estimation parts run, it will overwrite the file with more data. 
    outfilepath = os.path.join(args.outdirpath, f"result_{args.num_training_data}_{rngseed}.json")
    logger.info(f"Saving result JSON at: {outfilepath}")
    with open(outfilepath, "w") as outfile:
        json.dump(result, outfile, indent=2)

    if args.num_itemps is not None:
        # Run RLCT estimation regression. 
        n = args.num_training_data
        assert n == len(X)
        assert n == len(Y)
        itemps = linspaced_itemps_by_n(
            args.num_training_data, num_itemps=args.num_itemps
        )
        log_likelihood_fn = functools.partial(
            build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
        )
        enlls, stds = rlct_estimate_regression(
            itemps,
            next(rngkeyseq),
            model,
            log_likelihood_fn,
            X,
            Y,
            treedef,
            param_center,
            mcmc_config,
            progress_bar=(not args.quiet),
        )
        if args.plot_rlct_regression:
            fig, _ = plot_rlct_regression(itemps, enlls, n)
            filepath = os.path.join(args.outdirpath, f"rlct_regression_{args.num_training_data}_{rngseed}.png")
            fig.savefig(filepath)
            logger.info(f"RLCT regression figure saved at: {filepath}")

        slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
        _map_float = lambda lst: list(
            map(float, lst)
        )  # just to make sure things are json serialisable.
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
    outfilepath = os.path.join(args.outdirpath, f"result_{args.num_training_data}_{rngseed}.json")
    logger.info(f"Saving result JSON at: {outfilepath}")
    with open(outfilepath, "w") as outfile:
        json.dump(result, outfile, indent=2)
    return

def commandline_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--datafilepath", nargs="?", default=None, type=str, help="Path to a .npz file storing data. If specified, --true_param_filepath, --true_layer_sizes, --with_true_bias etc are ignored.")
    parser.add_argument("--true_param_filepath", nargs="?", default=None, type=str)
    parser.add_argument("--num_test_samples", nargs="?", default=300, type=int)
    parser.add_argument("--num_posterior_samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=4, type=int)
    parser.add_argument("--num_training_data", nargs="?", default=1032, type=int)
    parser.add_argument(
        "--num_itemps",
        nargs="?",
        default=None,
        type=int,
        help="If this is specified, the RLCT estimation procedure will be run.",
    )
    parser.add_argument(
        "--plot_rlct_regression",
        action="store_true",
        default=False,
        help="If set, plot the RLCT regression figure.",
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
        "--with_bias",
        action="store_true",
        default=False,
        help="If set, the model network will use bias parameters.",
    )
    parser.add_argument(
        "--with_true_bias",
        action="store_true",
        default=False,
        help="If set, the true network will use bias parameters.",
    )
    parser.add_argument("--sigma_obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior_std", nargs="?", default=3.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation_fn_name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default=None, type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--host_device_count",
        default=None,
        type=int,
        help="Number of host device for parallel run of MCMC chains.",
    )
    parser.add_argument(
        "--outdirpath",
        default="result",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument(
        "--no_wbic", action="store_true", default=False, help="If specified, the WBIC will NOT be calculated."
    )
    parser.add_argument(
        "--no_true_param_output", action="store_true", default=False, help="If specified, the true parameters will NOT be in the result JSON file."
    )
    parser.add_argument("--rng_seeds", nargs="+", default=[42], type=int)
    return parser

if __name__ == "__main__":
    parser = commandline_parser()
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


    logger = start_log(None, loglevel=logging.DEBUG, log_name=__name__)
    logger.info("Program starting...")
    logger.info(f"Commandline inputs: {args}")

    logger.info(f"Result to be saved at directory: {os.path.abspath(args.outdirpath)}")
    os.makedirs(args.outdirpath, exist_ok=True)
    
    for i, rngseed in enumerate(args.rng_seeds):
        logger.info(f"Starting experiment {i} of {len(args.rng_seeds)} with rngseed: {rngseed}")
        main(args, rngseed)
