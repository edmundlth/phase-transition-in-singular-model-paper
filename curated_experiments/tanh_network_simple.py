import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


ACTIVATION_FUNC_SWITCH = {
    "tanh": jnp.tanh,
    "id": lambda x: x,
    "relu": lambda x: jnp.maximum(0, x),
}

XMIN, XMAX = -2, 2

def nonlin(x):
    return jnp.tanh(x)

def regression_func(X, param_dict):
    W1 = param_dict["W1"]
    W2 = param_dict["W2"]
    return jnp.matmul(jnp.tanh(jnp.matmul(X, W2)), W1)


def generate_data(param_dict, sigma, num_training_samples, input_dim=1):
    X = np.random.uniform(XMIN, XMAX, (num_training_samples, input_dim))
    Y = regression_func(X, param_dict)
    Y += sigma * np.random.randn(*Y.shape)
    return X, Y


def model(X, Y, w1_prior, w2_prior, itemp=1.0, sigma=0.1):
    param_dict = {
        "W1": numpyro.sample("W1", w1_prior),
        "W2": numpyro.sample("W2", w2_prior),
    }
    y_hat = regression_func(X, param_dict)
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample(
            "Y", dist.Normal(y_hat, sigma / np.sqrt(itemp)).to_event(1), obs=Y
        )
    return


def expected_nll(param_dict, X, Y, sigma=0.1):
    y_hat = regression_func(X, param_dict)
    ydist = dist.Normal(y_hat, sigma)
    return -ydist.log_prob(Y).sum()
    # return 1 / (2 * self.sigma**2) * jnp.sum((self.Y - y_hat)**2)


def run_inference(
    model,
    X,
    Y,
    w1_prior, 
    w2_prior,
    rng_key,
    num_warmup,
    num_posterior_samples,
    num_chains,
    thinning=1,
    print_summary=False,
    progress_bar=False,
    itemp=1.0,
    sigma=0.1,
):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_posterior_samples,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
    )
    mcmc.run(rng_key, X, Y, w1_prior, w2_prior, itemp=itemp, sigma=sigma)
    if print_summary:
        mcmc.print_summary()
        print("\nMCMC elapsed time:", time.time() - start)
    return mcmc


def true_rlct(num_hidden_nodes):
    h = int(num_hidden_nodes**0.5)
    H = num_hidden_nodes
    return (H + h * h + h) / (4 * h + 2)


def main(args):
    true_param_dict = {
        "W1": jnp.array([list(args.a0)]),
        "W2": jnp.array([list(args.b0)]).T,
    }
    input_dim, num_hidden_nodes = true_param_dict["W1"].shape
    # output_dim = true_param_dict["W2"].shape[1]
    num_training_samples = args.num_data
    one1 = jnp.ones_like(true_param_dict["W1"])
    w1_prior = dist.Normal(args.prior_mean * one1, args.prior_std * one1)
    one2 = jnp.ones_like(true_param_dict["W2"])
    w2_prior = dist.Normal(args.prior_mean * one2, args.prior_std * one2)

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    X, Y = generate_data(true_param_dict, args.sigma_obs, args.num_data, input_dim)
    
    num_itemps = args.num_itemps
    n = num_training_samples
    itemps = np.linspace(
        1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))),
        num_itemps,
    )
    print(f"itemps={itemps}")


    num_rows = 2
    num_cols = len(itemps) // num_rows + (len(itemps) % num_rows != 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = np.ravel(axes)
    enlls = []
    for i, itemp in enumerate(itemps):
        ax = axes[i]

        mcmc = run_inference(
            model,
            X,
            Y,
            w1_prior, 
            w2_prior,
            rng_key,
            num_warmup=args.num_warmup,
            num_posterior_samples=args.num_samples,
            num_chains=args.num_chains,
            thinning=args.thinning,
            print_summary=True,
            progress_bar=True,
            itemp=itemp,
            sigma=args.sigma_obs
        )
        posterior_samples = mcmc.get_samples()
        nlls = []
        num_mcmc_samples = len(posterior_samples[list(posterior_samples.keys())[0]])
        for i in range(num_mcmc_samples):
            param_dict = {name: param[i] for name, param in posterior_samples.items()}
            nlls.append(expected_nll(param_dict, X, Y, sigma=args.sigma_obs))

        enll = np.mean(nlls)
        enlls.append(enll)
        print(f"Finished temp={1/itemp}. Expected NLL={enll}")

        i1, i2 = 0, 0
        j1, j2 = 0, 0
        ax.plot(
            posterior_samples["W1"][:, i1, i2],
            posterior_samples["W2"][:, j1, j2],
            "kx",
            markersize=1,
            alpha=0.5,
        )
        ax.plot(
            [true_param_dict["W1"][i1, i2]],
            [true_param_dict["W2"][j1, j2]],
            "rX",
            markersize=10,
        )

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines([0], xmin=xmin, xmax=xmax, linestyles="dotted", alpha=0.5, color="r")
        ax.vlines([0], ymin=ymin, ymax=ymax, linestyles="dotted", alpha=0.5, color="r")
        ax.set_title(
            f"Posterior samples. NLL={enll:.2f}, itemp={itemp:.2f}",
            fontdict={"fontsize": 5},
        )
    if args.output_name:
        fig.savefig(f"{args.output_name}_posterior_samples.png")

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
    if args.output_name:
        fig.savefig(f"{args.output_name}_rlct_regression.png")

    print(f"slope={slope} intercept={intercept} r2={r_val**2}")
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One hidden layer tanh model.")
    parser.add_argument("--num-itemps", nargs="?", default=6, type=int)
    parser.add_argument("--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=1, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=1032, type=int)
    parser.add_argument("--a0", nargs="+", default=[0.5], type=float)
    parser.add_argument("--b0", nargs="+", default=[0.9], type=float)
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--output_name", default=None, type=str, help='a prefixed used to name output files.')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
