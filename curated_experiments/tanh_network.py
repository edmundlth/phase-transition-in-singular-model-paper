import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, BarkerMH


ACTIVATION_FUNC_SWITCH = {
    "tanh": jnp.tanh, 
    "id": lambda x: x, 
    "relu": lambda x: jnp.maximum(0, x), 
}

XMIN, XMAX = -2, 2

class OneLayerTanhModel(object):
    
    def __init__(
        self, 
        true_param_dict, 
        num_training_samples=10, 
        input_dim=1, 
        output_dim=1,
        num_hidden_nodes=1, 
        itemp=1.0,
        sigma=1.0, 
        prior_name="uniform", 
        activation_func_name="tanh",
        prior_sigma=1.0
    ):
        self.true_param_dict = true_param_dict
        self.num_hidden_nodes = num_hidden_nodes
        self.num_training_samples = num_training_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.itemp=1.0
        self.prior_name = prior_name
        
        if self.prior_name.lower() == "uniform":
            self.w1_prior = dist.Uniform(
                low=-jnp.ones((self.input_dim, self.num_hidden_nodes)), 
                high=jnp.ones((self.input_dim, self.num_hidden_nodes))
            )
            self.w2_prior = dist.Uniform(
                low=-jnp.ones((self.num_hidden_nodes, self.output_dim)), 
                high=jnp.ones((self.num_hidden_nodes, self.output_dim))
            )
        elif self.prior_name.lower() == "normal":
            self.w1_prior = dist.Normal(
                jnp.zeros((self.input_dim, self.num_hidden_nodes)), 
                jnp.ones((self.input_dim, self.num_hidden_nodes)) * prior_sigma
            )
            self.w2_prior = dist.Normal(
                jnp.zeros((self.num_hidden_nodes, self.output_dim)), 
                jnp.ones((self.num_hidden_nodes, self.output_dim)) * prior_sigma
            )
        
        self.activation_func = ACTIVATION_FUNC_SWITCH[activation_func_name]
        

        
        self.X, self.Y = self.generate_data(num_training_samples)        
        self.rng_key, self.rng_key_predict = random.split(random.PRNGKey(0))
        
        
    def regression_func(self, X, param_dict):
        W1 = param_dict["W1"]
        W2 = param_dict["W2"]
        return jnp.matmul(self.activation_func(jnp.matmul(X, W1)), W2)
    
    def expected_nll(self, param_dict):
        y_hat = self.regression_func(self.X, param_dict)
        ydist = dist.Normal(y_hat, self.sigma)
        nll = -ydist.log_prob(self.Y).sum()
        return nll
#         nll = 1 / (2 * self.sigma**2) * jnp.sum((self.Y - y_hat)**2)
#         return nll
    
    def model(self, X, Y, itemp=None):
        param_dict = {
            "W1": numpyro.sample("W1", self.w1_prior),
            "W2": numpyro.sample("W2", self.w2_prior)
        }
        y_hat = self.regression_func(X, param_dict)
        if itemp is None:
            itemp = self.itemp
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("Y", dist.Normal(y_hat, self.sigma / np.sqrt(itemp)).to_event(1), obs=Y)
        return
    
    def generate_data(self, num_training_samples):
        X = np.random.uniform(XMIN, XMAX, (num_training_samples, self.input_dim))
        Y = self.regression_func(X, self.true_param_dict)
        Y += self.sigma * np.random.randn(*Y.shape)
        return X, Y

    def run_inference(
        self, 
        num_warmup, 
        num_posterior_samples, 
        num_chains, 
        thinning=1, 
        print_summary=False, 
        progress_bar=False, 
        itemp=None
    ):
        start = time.time()
        kernel = NUTS(self.model)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_posterior_samples,
            num_chains=num_chains,
            thinning=thinning, 
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
        )
        self.mcmc.run(
            self.rng_key, 
            self.X, 
            self.Y, 
            itemp=itemp,
        )
        if print_summary:
            self.mcmc.print_summary()
            print("\nMCMC elapsed time:", time.time() - start)
        return self.mcmc.get_samples()
    
    def predict(self, posterior_samples, X):
        def _helper(rng_key, posterior_samples, X):
            model = handlers.substitute(handlers.seed(self.model, rng_key), posterior_samples)
            # note that Y will be sampled in the model because we pass Y=None here
            model_trace = handlers.trace(model).get_trace(X=X, Y=None)
            return model_trace["Y"]["value"]
        num_posterior_samples = self.mcmc.get_samples()["W1"].shape[0]
        vmap_args = (
            posterior_samples,
            random.split(self.rng_key_predict, num_posterior_samples),
        )
        predictions = vmap(
            lambda samples, rng_key: _helper(rng_key, samples, X)
        )(*vmap_args)
        return predictions
    
    
    def true_rlct(self):
        h = int(self.num_hidden_nodes ** 0.5)
        H = self.num_hidden_nodes
        return (H + h * h + h) / (4 * h + 2)
    


def main(args):
    true_param_dict = {
        "W1": jnp.array([list(args.b0)]),
        "W2": jnp.array([list(args.a0)]).T
    }
    input_dim, num_hidden_nodes = true_param_dict["W1"].shape
    output_dim = true_param_dict["W2"].shape[1]
    num_training_samples = args.num_data
    one_layer_tanh = OneLayerTanhModel(
        true_param_dict, 
        num_training_samples=num_training_samples,
        input_dim=input_dim, 
        output_dim=output_dim,
        num_hidden_nodes=num_hidden_nodes,
        sigma=args.sigma_obs, 
        itemp=1.0,
        prior_name="normal", 
        activation_func_name="tanh", 
        prior_sigma=args.prior_std
    )

    num_itemps = args.num_itemps
    n = one_layer_tanh.num_training_samples
    itemps = np.linspace(
        1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))), 
        num_itemps
    )
    print(f"itemps={itemps}")

    num_rows = 2
    num_cols = len(itemps) // num_rows + (len(itemps) % num_rows != 0)

    if args.plot_posterior_samples: 
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
        axes = np.ravel(axes)
    enlls = []
    for i_itemp, itemp in enumerate(itemps):
        
        posterior_samples = one_layer_tanh.run_inference(
            num_warmup=args.num_warmup, 
            num_posterior_samples=args.num_samples,
            num_chains=args.num_chains,
            thinning=args.thinning,
            progress_bar=True, 
            itemp=itemp, 
            print_summary=True
        )
        nlls = []
        num_mcmc_samples = len(posterior_samples[list(posterior_samples.keys())[0]])
        for i in range(num_mcmc_samples):
            param_dict = {name: param[i] for name, param in posterior_samples.items()}
            nlls.append(one_layer_tanh.expected_nll(param_dict))

        enll = np.mean(nlls)
        enlls.append(enll)
        print(f"Finished {i_itemp} temp={1/itemp}. Expected NLL={enll}")
        if len(enlls) > 1:
            slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps[:len(enlls)], enlls)
            print(f"est. RLCT={slope}, energy={intercept / n}, r2={r_val**2}")
        
        if args.plot_posterior_samples:
            ax = axes[i_itemp]
            ax.plot(posterior_samples["W1"][:, 0, 0], posterior_samples["W2"][:, 0, 0], "kx", markersize=1, alpha=0.5)
            ax.plot([true_param_dict["W1"][0, 0]], [true_param_dict["W2"][0, 0]], "rX", markersize=10)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.hlines([0], xmin=xmin, xmax=xmax, linestyles="dotted", alpha=0.5, color='r')
            ax.vlines([0], ymin=ymin, ymax=ymax, linestyles="dotted", alpha=0.5, color='r')
            ax.set_title(f"Posterior samples. NLL={enll:.2f}, itemp={itemp:.2f}", fontdict={"fontsize": 5});
    if args.plot_posterior_samples and args.output_name:
        fig.savefig(f"{args.output_name}_posterior_samples.png")


    fig, ax = plt.subplots(figsize=(5,5))
    slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
    ax.plot(1/itemps, enlls, "kx")
    ax.plot(1/itemps, 1/itemps * slope + intercept, 
            label=f"$\lambda$={slope:.3f}, $nL_n(w_0)$={intercept:.3f}, $R^2$={r_val**2:.2f}")
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
    parser.add_argument("--plot_posterior_samples", action="store_true", default=False)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
