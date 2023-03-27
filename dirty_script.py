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

numpyro.set_host_device_count(4)
rngseed = 42
rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))


def loss_fn(params, x, y):
    y_pred = forward.apply(params, None, x)
    return jnp.mean(optax.l2_loss(y_pred, y))

def log_prior(params):
    param_array, _ = jtree.tree_flatten(params)
    result = 0.0
    for param in param_array:
        result += dist.Normal(loc=prior_mean, scale=prior_std).log_prob(param).sum()
    return result

def log_likelihood(params, x, y, sigma=1.0):
    y_hat = forward.apply(params, None, x)
    ydist = dist.Normal(y_hat, sigma)
    return ydist.log_prob(y).sum()

def log_posterior(params, x, y, itemp=1.0):
    return itemp * log_likelihood(params, x, y) + log_prior(params)


def numpyro_model(X, Y, shapes, treedef, itemp=1.0, sigma=0.1, prior_mean=0.0, prior_std=1.0):
    param_dict = normal_prior(shapes, treedef, prior_mean=prior_mean, prior_std=prior_std)
    y_hat = forward.apply(param_dict, None, X)
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("Y", dist.Normal(y_hat, sigma / jnp.sqrt(itemp)).to_event(1), obs=Y)
    return

def normal_prior(shapes, treedef, prior_mean=0.0, prior_std=1.0):
    result = []
    for i, shape in enumerate(shapes):
        result.append(numpyro.sample(str(i), dist.Normal(loc=prior_mean, scale=prior_std), sample_shape=shape))
    return treedef.unflatten(result)


def normal_localising_prior(params_center, std):
    result = []
    param_flat, treedef = jtree.tree_flatten(params_center)
    for i, p in enumerate(param_flat):
        result.append(numpyro.sample(str(i), dist.Normal(loc=p, scale=std)))
    return treedef.unflatten(result)


def expected_nll(param_list, X, Y, sigma):
    nlls = []
    for param in param_list:
        nlls.append(-log_likelihood(param, X, Y, sigma=sigma))
    return np.mean(nlls)


w_0 = [
    jnp.array([[0.1, 0.2, 0.01]]), 
    jnp.array([[0.02, -0.1, 0.5]]).T
]
input_dim, num_hidden_nodes = w_0[0].shape
output_dim = w_0[1].shape[1]
activation_fn = jax.nn.tanh

@hk.transform
def forward(x):
    mlp = hk.nets.MLP(
        [num_hidden_nodes, output_dim], # WARNING: there are global variables 
        activation=activation_fn, 
        w_init=w_initialiser,
        with_bias=False
    )
    return mlp(x)


sigma = 0.1
prior_std = 1.0
prior_mean = 0.0
w_initialiser = hk.initializers.RandomNormal(stddev=prior_std, mean=prior_mean)
num_itemps = 6

X = jax.random.uniform(
    key=next(rngkeyseq), 
    shape=(3, input_dim), 
    minval=-2, 
    maxval=2
)
true_param = forward.init(next(rngkeyseq), X)
treedef = jtree.tree_structure(true_param)
true_param = treedef.unflatten(w_0)
true_param_flat, treedef = jtree.tree_flatten(true_param)
shapes = [p.shape for p in true_param_flat]
print(true_param)

num_warmup = 500
num_posterior_samples = 2000
num_chains = 4
thinning = 4
kernel = numpyro.infer.NUTS(numpyro_model)
mcmc = numpyro.infer.MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_posterior_samples,
    num_chains=num_chains,
    thinning=thinning, 
    progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
)

ns = [100, 500, 1000, 2000, 5000, 7000]
run_rec = {}
for num_training_data in ns:
    X = jax.random.uniform(
        key=next(rngkeyseq), 
        shape=(num_training_data, input_dim), 
        minval=-2, 
        maxval=2
    )
    y_true = forward.apply(true_param, next(rngkeyseq), X)
    Y = y_true + jax.random.normal(next(rngkeyseq), y_true.shape) * sigma
    n = num_training_data
    itemps = jnp.linspace(
        1 / jnp.log(n) * (1 - 1 / jnp.sqrt(2 * jnp.log(n))),
        1 / jnp.log(n) * (1 + 1 / jnp.sqrt(2 * jnp.log(n))), 
        num_itemps
    )
    print(f"itemps={itemps}")

    enlls = []
    for i_itemp, itemp in enumerate(itemps): 
        mcmc.run(
            next(rngkeyseq), 
            X, Y, 
            shapes, treedef,
            itemp=itemp, 
            sigma=sigma, 
            prior_mean=prior_mean,
            prior_std=prior_std, 
        )
        posterior_samples = mcmc.get_samples()
        num_mcmc_samples = num_mcmc_samples = len(posterior_samples[list(posterior_samples.keys())[0]])

        param_list = [
            [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
            for i in range(num_mcmc_samples)
        ]
        enll = expected_nll(map(treedef.unflatten, param_list), X, Y, sigma)
        enlls.append(enll)
        print(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
        if len(enlls) > 1:
            slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps[:len(enlls)], enlls)
            print(f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}")
    
    run_rec[num_training_data] = {
        "itemps": itemps, "enlls": enlls, "slope": slope, "intercept": intercept, "rval": r_val
    }
    print(f"Finished (n={num_training_data}): RLCT est={slope:.3f}, energy={intercept / n:.3f}")


nums = sorted(run_rec.keys())
lambdas = [run_rec[n]["slope"] for n in nums]
energies = [run_rec[n]["intercept"] / n for n in nums]

fig, axes = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
ax = axes[0]
ax.plot(nums, lambdas, "kx--")
ax.set_xlabel("n")
ax.set_ylabel("$\hat{\lambda}$")
ax.set_xscale("log")

ax = axes[1]
ax.plot(nums, energies, "kx--")
ax.set_xlabel("n")
ax.set_ylabel("$\hat{L_n}(w_0)$")

plt.show()