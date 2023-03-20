# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network
================================

We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.

.. image:: ../_static/img/examples/bnn.png
    :align: center
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy import stats

# check setting from https://github.com/suswei/RLCT/blob/e9e04ca5e64250dfbb94134ec5283286dcdc4358/notebooks/demo.ipynb

numpyro.enable_x64()


def nonlin(x):
    return jnp.tanh(x)
    # return jax.nn.relu(x)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y, itemp, a_prior, b_prior, sigma_obs):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, itemp, a_prior, b_prior, sigma_obs)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def get_data(args):

    X = np.random.uniform(-2, 2, (args.num_data, 1))
    a = args.true_param_dict["a"]
    b = args.true_param_dict["b"]
    Y = jnp.matmul(nonlin(jnp.matmul(X, b)), a)
    Y += args.sigma_obs * np.random.randn(*Y.shape)
    return X, Y


def model(X, Y, itemp, a_prior, b_prior, sigma_obs):
    b = numpyro.sample(
        "b",
        b_prior
    )

    a = numpyro.sample(
        "a",
        a_prior
    )

    y_hat = jnp.matmul(nonlin(jnp.matmul(X, b)), a)

    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("Y", dist.Normal(y_hat, sigma_obs/np.sqrt(itemp)).to_event(1), obs=Y) # omitting .to_event(1) prompts a warning and gives bad results but I don't understand why


def expected_nll(X, Y, param_dict, sigma_obs):
    b = param_dict["b"]
    a = param_dict["a"]
    y_hat = jnp.matmul(nonlin(jnp.matmul(X, b)), a)
    ydist = dist.Normal(y_hat, sigma_obs)
    nll = -ydist.log_prob(Y).sum()
    return nll


def main(args):

    args.true_param_dict = {"a": jnp.array([[args.a0]]), "b": jnp.array([[args.b0]])}
    print(args)
    X, Y = get_data(args)

    num_itemps = 10
    itemps = np.linspace(
        1 / np.log(args.num_data) * (1 - 1 / np.sqrt(2 * np.log(args.num_data))),
        1 / np.log(args.num_data) * (1 + 1 / np.sqrt(2 * np.log(args.num_data))),
        num_itemps
    )

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    num_rows = 1
    num_cols = len(itemps) // num_rows + (len(itemps) % num_rows != 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = np.ravel(axes)

    b_prior = dist.Normal(args.prior_mean + jnp.zeros((1, 1)), args.prior_std * jnp.ones((1, 1)))
    a_prior = dist.Normal(args.prior_mean + jnp.zeros((1, 1)), args.prior_std * jnp.ones((1, 1)))

    enlls = []
    for i, itemp in enumerate(itemps):
        ax = axes[i]

        posterior_samples = run_inference(model, args, rng_key, X, Y, itemp, a_prior, b_prior, args.sigma_obs)
        nlls = []
        for i in range(args.num_samples):
            param_dict = {name: param[i] for name, param in posterior_samples.items()}
            nlls.append(expected_nll(X, Y, param_dict, args.sigma_obs))

        enll = np.mean(nlls)
        enlls.append(enll)
        print(f"Finished temp={1/itemp}. Expected NLL={enll}")

        i1, i2 = 0, 0
        j1, j2 = 0, 0
        ax.plot(posterior_samples["a"][:, i1, i2], posterior_samples["b"][:, j1, j2], "kx", markersize=1, alpha=0.5)
        ax.plot([args.true_param_dict["a"][i1, i2]], [args.true_param_dict["b"][j1, j2]], "rX", markersize=10)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines([0], xmin=xmin, xmax=xmax, linestyles="dotted", alpha=0.5, color='r')
        ax.vlines([0], ymin=ymin, ymax=ymax, linestyles="dotted", alpha=0.5, color='r')
        ax.set_title(f"Posterior samples. NLL={enll:.2f}, itemp={itemp:.2f}", fontdict={"fontsize": 5});

    fig, ax = plt.subplots(figsize=(5, 5))

    slope, intercept, r_val, _, _ = stats.linregress(1 / itemps, enlls)
    ax.plot(1 / itemps, enlls, "kx")
    ax.plot(1 / itemps, 1 / itemps * slope + intercept, label=f"$\lambda$={slope:.3f}, fitted $nL_n(w_0)$={intercept:.3f}, R^2={r_val ** 2}")
    print('slope {} intercept {} r2 {}'.format(slope, intercept, r_val))

    ax.legend()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Expected NLL")
    plt.show()

if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.11.0")
    parser = argparse.ArgumentParser(description="2dim tanh RLCT a*tanh(bx)")
    parser.add_argument("--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=1, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=1032, type=int)
    parser.add_argument("--a0", nargs="?", default=0.5, type=float)
    parser.add_argument("--b0", nargs="?", default=0.9, type=float)
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
