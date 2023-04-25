import jax.numpy as jnp
import numpy as np
import itertools
import copy
import jax
import haiku as hk
import jax.tree_util as jtree
import pickle
from src.const import ACTIVATION_FUNC_SWITCH
from src.haiku_numpyro_mlp import build_forward_fn
# from generalisation_error_experiment import commandline_parser, main
import os
from datetime import datetime
import sys


NSTART = 50
NEND = 20000
NUMRNGSEEDS = 20

CONFIG_RANGES = dict(
    a = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    sigma_obs = [0.1, 0.5, 1.0],
    prior_std = [0.1, 0.5, 1.0, 5.0],
    num_training_data = [int(n) for n in np.logspace(np.log10(NSTART), np.log10(NEND), base=10, num=20)]
)

BASE_CONFIG = {
    # "true_param_filepath": None, 
    # "num_training_data": None,
    # "outdirpath": None,
    # "sigma_obs": None,
    # "prior_std": None,
    
    "rng_seeds": list(range(NUMRNGSEEDS)),
    "input_dim": 1, 
    "num_itemps": 4, 
    "true_layer_sizes": [1, 1], 
    "layer_sizes": [1, 1], 
    "num_posterior_samples": 5000, 
    "thinning": 4, 
    "num_chains": 6, 
    "num_warmup": 800, 
    "num_test_samples": 10123, 
    "host_device_count": 6, 
    "plot_rlct_regression": True,
    "activation_fn_name": "tanh", 
    "prior_mean": 0.0, 
}


def get_config_vals(index):
    keys, values = zip(*CONFIG_RANGES.items())
    all_vals = list(itertools.product(*values))
    print(f"Total number of configs: {len(all_vals)}")
    if index >= len(all_vals):
        raise ValueError(f"Index {index} exceed length of all available configurations ({len(all_vals)})")
    config = dict(zip(keys, all_vals[index]))
    return config


def make_true_param(a, outputdir):    
    input_dim = BASE_CONFIG["input_dim"]
    true_layer_sizes = BASE_CONFIG["true_layer_sizes"]
    activation_fn_name = BASE_CONFIG["activation_fn_name"]
    prior_mean = BASE_CONFIG["prior_mean"]
    
    forward_true = hk.transform(
        build_forward_fn(
            layer_sizes=true_layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH[activation_fn_name],
            initialisation_mean=prior_mean,
            initialisation_std=1.0, 
            with_bias=False
        )
    )
    rngkey = jax.random.PRNGKey(0)
    dummy_X = jax.random.uniform(rngkey, shape=(5, input_dim))
    _, true_treedef = jtree.tree_flatten(forward_true.init(rngkey, dummy_X))
    true_param = true_treedef.unflatten(
        [jnp.array([[a]]), 
        jnp.array([[a]]).T]
    )
    true_param_filepath = os.path.join(outputdir, "true_param.pkl")
    print(f"Saving true parameter pickle file at: {true_param_filepath}")
    with open(true_param_filepath, 'wb') as f:
        pickle.dump(true_param, f)
    return true_param_filepath


if __name__ == "__main__":
    outputdir, index = sys.argv[1:3]
    config = get_config_vals(index)
    os.makedirs(outputdir, exist_ok=True)
    input_dim = BASE_CONFIG["input_dim"]
    true_layer_sizes = BASE_CONFIG["true_layer_sizes"]
    activation_fn_name = BASE_CONFIG["activation_fn_name"]
    prior_mean = BASE_CONFIG["prior_mean"]

    a = config.pop('a')
    true_param_filepath = make_true_param(a, outputdir)
    
    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d%H%M%S")

    cmd_list = ["python generalisation_error_experiment.py"]
    cmd_list += [f"--true_param_filepath {true_param_filepath}"]
    cmd_list += [f"--{key} {val}" for key, val in config.items()]
    cmd_list += [f"--{key} {val}" for key, val in BASE_CONFIG.items()]
    command = " ".join(cmd_list)

    print("Constructed command: ")
    print(command)
    os.system(command)

