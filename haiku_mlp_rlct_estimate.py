import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import scipy

import optax
import functools

import haiku as hk
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import os

from const import ACTIVATION_FUNC_SWITCH
from utils import start_log
import logging
logger = logging.getLogger(__name__)

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
    layer_sizes, activation_fn, initialisation_mean=0.0, initialisation_std=1.0, with_bias=False
):
    """
    Construct a Haiku transformed forward function for an MLP network
    based on specified architectural parameters.
    """
    w_initialiser = hk.initializers.RandomNormal(
        stddev=initialisation_mean, mean=initialisation_std
    )

    def forward(x):
        mlp = hk.nets.MLP(
            layer_sizes, activation=activation_fn, w_init=w_initialiser, with_bias=with_bias
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


def run_mcmc(
    model,
    X,
    Y,
    rng_key,
    param_center,
    prior_mean,
    prior_std,
    sigma,
    num_posterior_samples=2000,
    num_warmup=1000,
    num_chains=1,
    thinning=1,
    itemp=1.0,
    progress_bar=True,
):
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_samples=num_posterior_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
    )
    mcmc.run(
        rng_key, X, Y, param_center, prior_mean, prior_std, itemp=itemp, sigma=sigma
    )
    return mcmc


def main(args):
    logger = start_log()

    rngseed = args.rng_seed
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))
    if args.layer_sizes is None:
        true_param_array = [jnp.array([list(args.b0)]), jnp.array([list(args.a0)])]
        layer_sizes = [len(args.b0), len(args.a0)]
    else:
        true_param_array = None
        layer_sizes = args.layer_sizes

    input_dim = layer_sizes[0]
    output_dim = layer_sizes[-1]

    activation_fn = ACTIVATION_FUNC_SWITCH[args.activation_fn_name.lower()]
    forward = build_forward_fn(
        layer_sizes=layer_sizes,
        activation_fn=activation_fn,
        initialisation_mean=args.prior_mean,
        initialisation_std=args.prior_std,
    )
    forward = hk.transform(forward)
    log_likelihood_fn = functools.partial(
        build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
    )

    X = generate_input_data(
        args.num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    init_param = forward.init(next(rngkeyseq), X)
    init_param_flat, treedef = jtree.tree_flatten(init_param)
    # param_shapes = [p.shape for p in init_param_flat]
    if true_param_array is not None:
        true_param = treedef.unflatten(true_param_array)
    else:
        true_param = init_param
    Y = generate_output_data(
        forward, true_param, X, next(rngkeyseq), sigma=args.sigma_obs
    )
    param_center = true_param

    # param_prior_sampler = functools.partial(
    #     const_factorised_normal_prior, param_shapes, treedef, args.prior_mean, args.prior_std
    # )
    model = functools.partial(build_model, forward.apply)

    n = X.shape[0]
    itemps = jnp.linspace(
        1 / jnp.log(n) * (1 - 1 / jnp.sqrt(2 * jnp.log(n))),
        1 / jnp.log(n) * (1 + 1 / jnp.sqrt(2 * jnp.log(n))),
        args.num_itemps,
    )
    logger.info(f"itemps={itemps}")

    enlls = []
    for i_itemp, itemp in enumerate(itemps):
        mcmc = run_mcmc(
            model,
            X,
            Y,
            next(rngkeyseq),
            param_center,
            args.prior_mean,
            args.prior_std,
            args.sigma_obs,
            num_posterior_samples=args.num_posterior_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            thinning=args.thinning,
            itemp=itemp,
            progress_bar=(not args.quiet),
        )
        posterior_samples = mcmc.get_samples()
        num_mcmc_samples = num_mcmc_samples = len(
            posterior_samples[list(posterior_samples.keys())[0]]
        )
        param_list = [
            [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
            for i in range(num_mcmc_samples)
        ]
        enll = expected_nll(log_likelihood_fn, map(treedef.unflatten, param_list), X, Y)
        enlls.append(enll)
        logger.info(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
        if len(enlls) > 1:
            slope, intercept, r_val, _, _ = scipy.stats.linregress(
                1 / itemps[: len(enlls)], enlls
            )
            logger.info(
                f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}"
            )

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
    n = args.num_training_data
    ax.set_title(f"n={n}, L_n={intercept / n:.3f}")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT estimation of MLP models.")
    parser.add_argument("--num-itemps", nargs="?", default=6, type=int)
    parser.add_argument("--num-posterior-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=1, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-training-data", nargs="?", default=1032, type=int)
    parser.add_argument("--a0", nargs="+", default=[0.5], type=float)
    parser.add_argument("--b0", nargs="+", default=[0.9], type=float)
    parser.add_argument(
        "--layer_sizes",
        nargs="+",
        default=None,
        type=int,
        help="A optional list of positive integers specifying MLP layers sizes including the input and output dimensions. If specified, --a0 and --b0 are ignored. ",
    )
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation-fn-name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="a directory for storing output files.",
    )
    parser.add_argument("--plot_posterior_samples", action="store_true", default=False)
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--rng-seed", nargs="?", default=42, type=int)

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
