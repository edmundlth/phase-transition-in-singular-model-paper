import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
from scipy.special import logsumexp
import functools
import matplotlib.pyplot as plt
import argparse
import os
import logging
import json

from src.utils import (
    start_log,
    MCMCConfig,
)
from src.haiku_numpyro_mlp import (
    run_mcmc,
    rlct_estimate_regression,
)


def build_gaussian_mixture_model(component_dim, num_component=2, sigma=1.0, prior_sigma=10.0, alpha=0.3):
    def gaussian_mixture_model(data, itemp=1.0):
        weights = numpyro.sample("weights", dist.Dirichlet(concentration=np.ones(num_component) * alpha))
        mus = [
            numpyro.sample(
                f"mu{i + 1}",
                dist.MultivariateNormal(
                    loc=jnp.zeros((component_dim,)), 
                    covariance_matrix=jnp.eye(component_dim) * prior_sigma
                )
            ) for i in range(num_component)
            
        ]
        with numpyro.plate("data", data.shape[0]), numpyro.handlers.scale(scale=itemp):
            numpyro.sample(
                "obs", 
                dist.MixtureGeneral(
                    mixing_distribution=dist.Categorical(probs=weights), 
                    component_distributions=[
                        dist.MultivariateNormal(loc=mu, covariance_matrix=jnp.eye(component_dim) * sigma)
                        for mu in mus
                    ]
                ), 
                obs=data
            )
    return gaussian_mixture_model

def _compute_loglike_array(loglike_fn, data, posterior_param_list):
    loglike_array = np.hstack(
        [loglike_fn(param, data) for param in posterior_param_list]
    )
    return loglike_array

def compute_bayesian_loss(loglike_array):
    # dimension = (num test samples, num mcmc samples)
    num_mcmc_samples = loglike_array.shape[1]
    result = -np.mean(logsumexp(loglike_array, b=1 / num_mcmc_samples, axis=1))
    return result

def compute_gibbs_loss(loglike_array):
    # gerrs = []
    # for param in posterior_param_list:
    #     gibbs_err = np.mean(loglike_fn(param, data))
    #     gerrs.append(gibbs_err)
    gerrs = np.mean(loglike_array, axis=0)
    gg = np.mean(gerrs)
    return -gg


def compute_functional_variance(loglike_array):
    # variance over posterior samples and averaged over dataset.
    # V = 1/n \sum_{i=1}^n Var_w(\log p(X_i | w))
    result = np.mean(np.var(loglike_array, axis=0))
    return result


def compute_waic(loglike_array):
    func_var = compute_functional_variance(loglike_array)
    bayes_train_loss = compute_bayesian_loss(loglike_array)
    return bayes_train_loss + func_var


def log_likelihood_func(params, data, component_dim, sigma=1.0):
    mukeys = sorted([key for key in params.keys() if key.startswith("mu")])
    mus = [params[key] for key in mukeys]
    weights = params["weights"]
    cov_matrix = jnp.eye(component_dim) * sigma
    component_densities = jnp.array(
        [
            dist.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix).log_prob(data) 
            for mu in mus
        ]
    ).T
    log_probs = jnp.log(weights) + component_densities
    loglikes = logsumexp(log_probs, axis=-1, keepdims=True)
    return loglikes


def main(args, rngseed):
    np.random.seed(rngseed)
    rngkey = jax.random.PRNGKey(rngseed)
    num_component = 2
    a0 = np.array(args.mixture_weights)
    a0 /= np.sum(a0)
    assert len(a0) == num_component

    delta = args.delta
    b0 = np.ones(args.component_dim) * delta
    c0 = -np.ones(args.component_dim) * delta

    n1, n2 = list(map(int, args.num_training_data * a0))
    data = np.concatenate([
        np.random.randn(n1, args.component_dim) * args.sigma + b0, 
        np.random.randn(n2, args.component_dim) * args.sigma + c0
    ])

    n1_test, n2_test = list(map(int, args.num_test_samples * a0))
    test_data = np.concatenate([
        np.random.randn(n1_test, args.component_dim) * args.sigma + b0, 
        np.random.randn(n2_test, args.component_dim) * args.sigma + c0
    ])

    mcmc_config = MCMCConfig(
        args.num_posterior_samples, args.num_warmup, args.num_chains, args.thinning
    )
    
    loglike_fn = functools.partial(log_likelihood_func, component_dim=args.component_dim, sigma=args.sigma)
    model = build_gaussian_mixture_model(
        args.component_dim, 
        num_component, 
        sigma=args.sigma,
        prior_sigma=args.prior_std, 
        alpha=args.alpha
    )
    
    mcmc = run_mcmc(
        model, 
        data, 
        rngkey, 
        mcmc_config, 
        itemp=1.0, 
        progress_bar=(not args.quiet)
    )
    posterior_samples = mcmc.get_samples()
    num_mcmc_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
    logger.info(f"Num mcmc samples={num_mcmc_samples}")

    posterior_param_list = [
        {key: posterior_samples[key][i] for key in posterior_samples.keys()} for i in range(num_mcmc_samples)
    ]

    loglike_array = _compute_loglike_array(loglike_fn, data, posterior_param_list)
    loglike_array_test = _compute_loglike_array(loglike_fn, test_data, posterior_param_list)
    
    gibbs_loss = compute_gibbs_loss(loglike_array_test)
    logger.info(f"Gibbs loss: {gibbs_loss}")
    
    gibbs_train_loss = compute_gibbs_loss(loglike_array)
    logger.info(f"Gibbs train loss: {gibbs_train_loss}")

    bayes_loss = compute_bayesian_loss(loglike_array_test)
    logger.info(f"Bayes loss: {bayes_loss}")
    
    bayes_train_loss = compute_bayesian_loss(loglike_array)
    logger.info(f"Bayes train loss: {bayes_train_loss}")

    func_var = compute_functional_variance(loglike_array)
    logger.info(f"Functional Variance: {func_var}")

    waic = compute_waic(loglike_array)
    logger.info(f"WAIC: {waic}")

    true_param = {
        "weights": a0, 
        "mu1": b0, 
        "mu2": c0, 
    }
    truth_entropy = -np.mean(loglike_fn(true_param, test_data))
    truth_entropy_train = -np.mean(loglike_fn(true_param, data))
    bayes_error = bayes_loss - truth_entropy
    gibbs_error = gibbs_loss - truth_entropy

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
        "truth_entropy": float(truth_entropy),
        "truth_entropy_train": float(truth_entropy_train),
        "Gg": float(gibbs_error),
        "Bg": float(bayes_error),
    }
    logger.info(f"Finished generalisation error calculation: {json.dumps(result)}")
    if args.outdirpath:
        outfilepath = os.path.join(args.outdirpath, f"result_{args.num_training_data}_{rngseed}.json")
        logger.info(f"Saving result JSON at: {outfilepath}")
        with open(outfilepath, "w") as outfile:
            json.dump(result, outfile, indent=2)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal mixture model experiment.")
    parser.add_argument(
        "--outdirpath",
        default=None,
        type=str,
        help="Path to output directory",
    )    
    parser.add_argument("--num_test_samples", nargs="?", default=3183, type=int)
    parser.add_argument("--num_posterior_samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=6, type=int)
    parser.add_argument("--num_training_data", nargs="?", default=1032, type=int)

    # parser.add_argument("--num_component", nargs="?", default=2, type=int, help="Number of component distributions in the mixture model.")
    parser.add_argument("--component_dim", nargs="?", default=2, type=int, help="Dimension of the component Gaussian model.")
    parser.add_argument("--mixture_weights", nargs="+", default=[0.5, 0.5], type=float, help="Mixture weights. Will be normalised before use so that they sum to unity.")
    parser.add_argument("--delta", nargs="?", default=0.5, type=float)
    parser.add_argument("--sigma", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_std", nargs="?", default=10.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--alpha", nargs="?", default=0.3, type=float, help="Parameter for Dirichlet distribution as prior for mixture weights.")
    
    parser.add_argument("--device", default=None, type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--host_device_count",
        default=None,
        type=int,
        help="Number of host device for parallel run of MCMC chains.",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--rng_seeds", nargs="+", default=[42], type=int)

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

    if args.outdirpath:
        logger.info(f"Result to be saved at directory: {os.path.abspath(args.outdirpath)}")
        os.makedirs(args.outdirpath, exist_ok=True)
    

    for i, rngseed in enumerate(args.rng_seeds):
        logger.info(f"Starting experiment {i} of {len(args.rng_seeds)} with rngseed: {rngseed}")
        main(args, rngseed)
        logger.info(f"Finished experiment {i} of {len(args.rng_seeds)} with rngseed: {rngseed}")



    
    



    