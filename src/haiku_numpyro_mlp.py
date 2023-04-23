import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import optax
import haiku as hk
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import scipy

import os
import time
import logging
from src.utils import MCMCConfig

# logger = logging.getLogger(__name__)
logger = logging.getLogger("__main__")  # TODO: Figure out these differences...


def const_factorised_normal_prior(param_example, prior_mean=0.0, prior_std=1.0):
    """
    Return a param PyTree with the same structure as `param_example` but with every
    element replaced with a random sample from normal distribution with `prior_mean` and `prior_std`.
    """
    param_flat, treedef = jtree.tree_flatten(param_example)
    result = []
    for i, param in enumerate(param_flat):
        result.append(
            numpyro.sample(
                str(i),
                dist.Normal(loc=prior_mean, scale=prior_std),
                sample_shape=param.shape,
            )
        )
    return treedef.unflatten(result)


def localised_normal_prior(param_center, std=1.0):
    """
    Return a param PyTree with the same structure as `param_center` but with every
    element replaced with a random sample from normal distribution centered around values of `param_center` with standard deviation `std`.
    """
    result = []
    param_flat, treedef = jtree.tree_flatten(param_center)
    for i, p in enumerate(param_flat):
        result.append(numpyro.sample(str(i), dist.Normal(loc=p, scale=std)))
    return treedef.unflatten(result)


def build_forward_fn(
    layer_sizes,
    activation_fn,
    initialisation_mean=0.0,
    initialisation_std=1.0,
    with_bias=False,
):
    """
    Construct a Haiku transformed forward function for an MLP network
    based on specified architectural parameters.
    """
    w_initialiser = hk.initializers.RandomNormal(
        stddev=initialisation_std, mean=initialisation_mean
    )
    if with_bias:
        b_initialiser = hk.initializers.RandomNormal(
            stddev=initialisation_std, mean=initialisation_mean
        )
    else:
        b_initialiser = None

    def forward(x):
        mlp = hk.nets.MLP(
            layer_sizes,
            activation=activation_fn,
            w_init=w_initialiser,
            b_init=b_initialiser,
            with_bias=with_bias,
        )
        return mlp(x)

    return forward


def build_loss_fn(forward_fn, param, x, y):
    y_pred = forward_fn(param, None, x)
    return jnp.mean(optax.l2_loss(y_pred, y))


def build_model(
    forward_fn, X, Y, param_center, prior_mean, prior_std, itemp=1.0, sigma=1.0
):
    param_dict = const_factorised_normal_prior(param_center, prior_mean, prior_std)
    # param_dict = localised_normal_prior(param_center, prior_std)
    y_hat = forward_fn(param_dict, None, X)
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample(
            "Y", dist.Normal(y_hat, sigma / jnp.sqrt(itemp)).to_event(1), obs=Y
        )
    return


def build_log_likelihood_fn(forward_fn, param, x, y, sigma=1.0):
    y_hat = forward_fn(param, None, x)
    ydist = dist.Normal(y_hat, sigma)
    return ydist.log_prob(y).sum()


def expected_nll(log_likelihood_fn, param_list, X, Y):
    nlls = []
    for param in param_list:
        nlls.append(-log_likelihood_fn(param, X, Y))
    return np.mean(nlls)


def generate_input_data(num_training_data, input_dim, rng_key, xmin=-2, xmax=2):
    X = jax.random.uniform(
        key=rng_key,
        shape=(num_training_data, input_dim),
        minval=xmin,
        maxval=xmax,
    )
    return X


def generate_output_data(foward_fn, param, X, rng_key, sigma=0.1):
    y_true = foward_fn.apply(param, None, X)
    Y = y_true + jax.random.normal(rng_key, y_true.shape) * sigma
    return Y


def rlct_estimate_regression(
    itemps,
    rng_key,
    model,
    log_likelihood_fn,
    X,
    Y,
    treedef,
    param_center,
    mcmc_config: MCMCConfig,
    progress_bar=True,
):
    n = len(X)
    enlls = []
    stds = []
    rngkeys = jax.random.split(rng_key, num=len(itemps))
    for i_itemp, itemp in enumerate(itemps):
        mcmc = run_mcmc(
            model,
            X,
            Y,
            rngkeys[i_itemp],
            param_center,
            mcmc_config,
            itemp=itemp,
            progress_bar=progress_bar,
        )
        chain_enlls, chain_sizes = chain_wise_enlls(
            mcmc, treedef, log_likelihood_fn, X, Y
        )
        enll = np.sum(np.array(chain_enlls) * np.array(chain_sizes)) / np.sum(
            chain_sizes
        )
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
    return enlls, stds


def run_mcmc(
    model,
    X,
    Y,
    rng_key,
    param_center,
    mcmc_config: MCMCConfig,
    itemp=1.0,
    progress_bar=True,
):
    kernel = numpyro.infer.NUTS(model)
    logger.info("Running MCMC...")
    start_time = time.time()
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_samples=mcmc_config.num_posterior_samples,
        num_warmup=mcmc_config.num_warmup,
        num_chains=mcmc_config.num_chains,
        thinning=mcmc_config.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
    )
    mcmc.run(rng_key, X, Y, param_center, itemp=itemp)
    logger.info(
        f"Finished running MCMC. Time taken: {time.time() - start_time:.3f} seconds"
    )
    return mcmc


def chain_wise_enlls(mcmc, treedef, log_likelihood_fn, X, Y):
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    num_chains, num_mcmc_samples_per_chain = posterior_samples[
        list(posterior_samples.keys())[0]
    ].shape[:2]
    num_mcmc_samples = num_chains * num_mcmc_samples_per_chain
    logger.info(f"Total number of MCMC samples: {num_mcmc_samples}")
    logger.info(f"Number of MCMC chain: {num_chains}")
    logger.info(f"Number of MCMC samples per chain: {num_mcmc_samples_per_chain}")
    chain_enlls = []
    chain_sizes = []
    for chain_index in range(num_chains):
        # TODO: is there a better way to do this? vectorisation possible?
        param_list = [
            [
                posterior_samples[name][chain_index, i]
                for name in sorted(posterior_samples.keys())
            ]
            for i in range(num_mcmc_samples_per_chain)
        ]
        chain_enll = expected_nll(
            log_likelihood_fn, map(treedef.unflatten, param_list), X, Y
        )
        chain_size = len(param_list)
        chain_enlls.append(chain_enll)
        chain_sizes.append(chain_size)
        logger.info(
            f"Chain {chain_index} with size {chain_size} has enll {chain_enll}."
        )
    return chain_enlls, chain_sizes


def plot_rlct_regression(itemps, enlls, n):
    fig, ax = plt.subplots(figsize=(5, 5))
    slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
    ax.plot(1 / itemps, enlls, "kx")
    ax.plot(
        1 / itemps,
        1 / itemps * slope + intercept,
        label=f"$\lambda$={slope:.3f}, $nL_n(w_0)$={intercept:.3f}, $R^2$={r_val**2:.2f}",
    )
    ax.legend()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Expected NLL")
    ax.set_title(f"n={n}, L_n={intercept / n:.3f}")
    return fig, ax
