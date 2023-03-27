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

class MLP_Experiment(object):

    def __init__(self, layer_sizes, num_training_data, true_param=None, true_param_array=None, activation_fn_name="tanh", sigma=0.1, prior_mean=0.0, prior_std=1.0, X=None, Y=None, rngseed=42):
        self.layer_sizes = layer_sizes 
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]

        self.num_training_data = num_training_data
        self.sigma = sigma
        self.activation_fn = ACTIVATION_FUNC_SWITCH[activation_fn_name.lower()]
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.rngseed = rngseed
        self.rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(self.rngseed))
        
        self.forward = build_forward_fn(
            layer_sizes, 
            self.activation_fn, 
            initialisation_mean=prior_mean,
            initialisation_std=prior_std
        )
        self.X = self.generate_input_data(self.num_training_data)
        if true_param is not None:
            self.true_param = true_param
        elif true_param_array is not None:
            self.true_param = self._param_array_to_param_tree(true_param_array)
        else:
            self.true_param = self.forward.init(next(self.rngkeyseq), self.X)
        self.true_param_flat, self.treedef = jtree.tree_flatten(self.true_param)
        self.param_shapes = [p.shape for p in self.true_param_flat]
        self.Y = self.generate_output_data(self.true_param, self.X)

        
    def _param_array_to_param_tree(self, param_array=None):
        sample_param = self.forward.init(jax.random.PRNGKey(0), self.X)
        treedef = jtree.tree_structure(sample_param)
        return treedef.unflatten(param_array)

    def loss_fn(self, param, x, y):
        y_pred = self.forward.apply(param, None, x)
        return jnp.mean(optax.l2_loss(y_pred, y))

    def log_prior(self, param):
        param_array, _ = jtree.tree_flatten(param)
        result = 0.0
        for param in param_array:
            result += dist.Normal(loc=self.prior_mean, scale=self.prior_std).log_prob(param).sum()
        return result

    def log_likelihood(self, param, x, y, sigma=1.0):
        y_hat = self.forward.apply(param, None, x)
        ydist = dist.Normal(y_hat, sigma)
        return ydist.log_prob(y).sum()

    def log_posterior(self, param, x, y, itemp=1.0):
        return itemp * self.log_likelihood(param, x, y) + self.log_prior(param)


    def model(self, X, Y, param_prior_sampler, itemp=1.0):
        param_dict = param_prior_sampler()
        # param_dict = normal_prior(shapes, treedef, prior_mean=prior_mean, prior_std=prior_std)
        y_hat = self.forward.apply(param_dict, None, X)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("Y", dist.Normal(y_hat, self.sigma / jnp.sqrt(itemp)).to_event(1), obs=Y)
        return

    def expected_nll(self, param_list, X, Y, sigma):
        nlls = []
        for param in param_list:
            nlls.append(-self.log_likelihood(param, X, Y, sigma=sigma))
        return np.mean(nlls)
    
    def generate_input_data(self, num_training_data, rng_key=None, xmin=-2, xmax=2):
        rng_key = rng_key if rng_key is not None else next(self.rngkeyseq)
        X = jax.random.uniform(
            key=rng_key, 
            shape=(num_training_data, self.input_dim), 
            minval=xmin, 
            maxval=xmax
        )
        return X
    
    def generate_output_data(self, param, X, rng_key=None):
        rng_key = rng_key if rng_key is not None else next(self.rngkeyseq)
        y_true = self.forward.apply(param, None, X)
        Y = y_true + jax.random.normal(rng_key, y_true.shape) * self.sigma
        return X, Y
    
    def run_mcmc(
            self, 
            num_posterior_samples=2000,
            num_warmup=1000,
            num_chains=1, 
            thinning=1, 
            param_prior_sampler=None,
            itemp=1.0,
            progress_bar=True, 
            rng_key=None
        ):
        rng_key = rng_key if rng_key is not None else next(self.rngkeyseq)
        kernel = numpyro.infer.NUTS(self.model)
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_posterior_samples,
            num_chains=num_chains,
            thinning=thinning, 
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
        )

        if param_prior_sampler is None:
            param_prior_sampler = lambda : const_factorised_normal_prior(
                shapes=self.param_shapes, 
                treedef=self.treedef, 
                prior_mean=self.prior_mean, 
                prior_std=self.prior_std
            )
        mcmc.run(
            rng_key, 
            self.X, self.Y, 
            param_prior_sampler,
            itemp=itemp, 
        )
        return mcmc

    def run_rlct_estimation(
            self, 
            num_itemps,
            num_posterior_samples=2000,
            num_warmup=1000,
            num_chains=1, 
            thinning=1, 
            param_prior_sampler=None,
            progress_bar=True
        ):
        n = self.X.shape[0]
        itemps = jnp.linspace(
            1 / jnp.log(n) * (1 - 1 / jnp.sqrt(2 * jnp.log(n))),
            1 / jnp.log(n) * (1 + 1 / jnp.sqrt(2 * jnp.log(n))), 
            num_itemps
        )
        print(f"itemps={itemps}")
        
        enlls = []
        for i_itemp, itemp in enumerate(itemps):
            mcmc = self.run_mcmc(
                num_posterior_samples=num_posterior_samples,
                num_warmup=num_warmup,
                num_chains=num_chains, 
                thinning=thinning, 
                itemp=itemp,
                param_prior_sampler=param_prior_sampler,
                progress_bar=progress_bar
            )
            posterior_samples = mcmc.get_samples()
            num_mcmc_samples = num_mcmc_samples = len(posterior_samples[list(posterior_samples.keys())[0]])
            param_list = [
                [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
                for i in range(num_mcmc_samples)
            ]
            enll = self.expected_nll(map(self.treedef.unflatten, param_list), self.X, self.Y, self.sigma)
            enlls.append(enll)
            print(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
            if len(enlls) > 1:
                slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps[:len(enlls)], enlls)
                print(f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}")
        return itemps, enlls


def const_factorised_normal_prior(shapes, treedef, prior_mean=0.0, prior_std=1.0):
    result = []
    for i, shape in enumerate(shapes):
        result.append(numpyro.sample(str(i), dist.Normal(loc=prior_mean, scale=prior_std), sample_shape=shape))
    return treedef.unflatten(result)


def localised_normal_prior(param_center, std):
    result = []
    param_flat, treedef = jtree.tree_flatten(param_center)
    for i, p in enumerate(param_flat):
        result.append(numpyro.sample(str(i), dist.Normal(loc=p, scale=std)))
    return treedef.unflatten(result)


def build_forward_fn(layer_sizes, activation_fn, initialisation_mean=0.0, initialisation_std=1.0):
    """
    Construct a Haiku transformed forward function for an MLP network 
    based on specified architectural parameters. 
    """
    w_initialiser = hk.initializers.RandomNormal(
        stddev=initialisation_mean, 
        mean=initialisation_std
    )
    @hk.transform
    def forward(x):
        mlp = hk.nets.MLP(
            layer_sizes, 
            activation=activation_fn, 
            w_init=w_initialiser,
            with_bias=False
        )
        return mlp(x)
    return forward



def main(args):

    if args.layer_sizes is None:
        true_param_array = [
            jnp.array([list(args.b0)]), 
            jnp.array([list(args.a0)])
        ]
        layer_sizes = [len(args.b0), len(args.a0)]
    else:
        true_param_array = None
        layer_sizes = args.layer_sizes
    
    expt = MLP_Experiment(
        layer_sizes=layer_sizes, 
        num_training_data=args.num_training_data,
        true_param_array=true_param_array,
        activation_fn_name=args.activation_fn_name, 
        sigma=args.sigma_obs, 
        prior_mean=args.prior_mean, 
        prior_std=args.prior_std,
    )

    itemps, enlls = expt.run_rlct_estimation(
        num_itemps=args.num_itemps,
        num_posterior_samples=args.num_posterior_samples, 
        num_chains=args.num_chains, 
        num_warmup=args.num_warmup, 
        thinning=args.thinning, 
        progress_bar=(not args.quiet), 
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
    ax.plot(1/itemps, enlls, "kx")
    ax.plot(1/itemps, 1/itemps * slope + intercept, 
            label=f"$\lambda$={slope:.3f}, $nL_n(w_0)$={intercept:.3f}, $R^2$={r_val**2:.2f}")
    ax.legend()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Expected NLL")
    n = args.num_training_data
    ax.set_title(f"n={n}, L_n={intercept / n:.3f}")
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
    parser.add_argument("--layer_sizes", nargs="+", default=None, type=int, help="A optional list of positive integers specifying MLP layers sizes including the input and output dimensions. If specified, --a0 and --b0 are ignored. ")
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation-fn-name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--output_dir", default=None, type=str, help='a directory for storing output files.')
    parser.add_argument("--plot_posterior_samples", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False, help="Lower verbosity level.")
    parser.add_argument("--rng-seed", nargs="?", default=42, type=int)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
