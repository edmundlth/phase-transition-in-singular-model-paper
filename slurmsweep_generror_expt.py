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
import os
from datetime import datetime
import sys


# expt on 20230425
# NSTART = 50
# NEND = 20000
# NUMRNGSEEDS = 20
# CONFIG_RANGES = dict(
#     a=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
#     num_training_data=[
#         int(n) for n in np.logspace(np.log10(NSTART), np.log10(NEND), base=10, num=20)
#     ],
#     sigma_obs=[0.1, 0.5, 1.0],
#     prior_std=[0.1, 0.5, 1.0, 5.0],
# )

# expt on 20230426
# NSTART = 100
# NEND = 20000
# NUMRNGSEEDS = 100
# CONFIG_RANGES = dict(
#     a=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
#     num_training_data=[10, 30, 50, 70, 90] + [
#         int(n) for n in np.logspace(np.log10(NSTART), np.log10(NEND), base=10, num=20)
#     ],
#     sigma_obs=[0.1, 1.0],
#     prior_std=[0.1, 5.0],
#     prior_mean=[0.0, 0.6, 1.0], 
# )

# expt on 20230427
NSTART = 100
NEND = 5000
NUMRNGSEEDS = 50
XMIN, XMAX = -3, 3
CONFIG_RANGES = dict(
    a = [
    [[[3.0, -.7]], [[1.0], [1.0]]],
    [[[2.0, -.7]], [[0.3], [.4]]], 
    [[[5.0, -.5]], [[0.3], [.2]]]
    ],
    num_training_data=[10, 30, 50, 70, 90] + list(map(int, np.linspace(NSTART, NEND, num=20))),
    sigma_obs=[0.1, 0.3, 0.5],
    prior_std=[1.0, 10.0],
    prior_mean=[0.0], 
)


BASE_CONFIG = {
    # "true_param_filepath": None,
    # "num_training_data": None,
    # "outdirpath": None,
    # "sigma_obs": None,
    # "prior_std": None,
    "rng_seeds": list(range(NUMRNGSEEDS)),
    "input_dim": 1,
    # "num_itemps": 4, # commenting out will disable rlct estimation.
    "true_layer_sizes": [2, 1],
    "layer_sizes": [2, 1],
    "num_posterior_samples": 3000,
    "thinning": 3,
    "num_chains": 6,
    "num_warmup": 500,
    "num_test_samples": 10123,
    "host_device_count": 6,
    "plot_rlct_regression": True,
    "activation_fn_name": "tanh",
    # "prior_mean": 0.0,
}


def get_config_vals(index):
    keys, values = zip(*CONFIG_RANGES.items())
    all_vals = list(itertools.product(*values))
    print(f"Total number of configs: {len(all_vals)}")
    if index >= len(all_vals):
        raise ValueError(
            f"Index {index} exceed length of all available configurations ({len(all_vals)})"
        )
    config = dict(zip(keys, all_vals[index]))
    return config


def make_true_param(a, outputdir):
    input_dim = BASE_CONFIG["input_dim"]
    true_layer_sizes = BASE_CONFIG["true_layer_sizes"]
    activation_fn_name = BASE_CONFIG["activation_fn_name"]

    forward_true = hk.transform(
        build_forward_fn(
            layer_sizes=true_layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH[activation_fn_name],
            initialisation_mean=0.0,
            initialisation_std=1.0,
            with_bias=False,
        )
    )
    rngkey = jax.random.PRNGKey(0)
    dummy_X = jax.random.uniform(rngkey, shape=(5, input_dim))
    _, true_treedef = jtree.tree_flatten(forward_true.init(rngkey, dummy_X))
    if isinstance(a, float):
        true_param = true_treedef.unflatten([jnp.array([[a]]), jnp.array([[a]]).T])
    elif isinstance(a, list):
        true_param = true_treedef.unflatten([jnp.array(elem) for elem in a])
    true_param_filepath = os.path.join(outputdir, "true_param.pkl")
    print(f"Saving true parameter pickle file at: {true_param_filepath}")
    with open(true_param_filepath, "wb") as f:
        pickle.dump(true_param, f)
    return true_param_filepath


def _make_cmd_arg(key, val):
    if isinstance(val, list):
        val = " ".join(map(str, val))
        return f"--{key} {val}"
    elif isinstance(val, bool):
        if val:
            return f"--{key}"
        else:
            return ""
    else:
        return f"--{key} {val}"


if __name__ == "__main__":
    outputdir, index = sys.argv[1:3]
    index = int(index)
    config = get_config_vals(index)

    output_subdir = os.path.join(outputdir, f"expt{index}")
    os.makedirs(output_subdir, exist_ok=True)

    a = config.pop("a")
    print(f"a={a}")
    true_param_filepath = make_true_param(a, output_subdir)

    cmd_list = ["python generalisation_error_experiment.py"]
    cmd_list += [
        f"--true_param_filepath {true_param_filepath}",
        f"--outdirpath {output_subdir}",
    ]
    cmd_list += [_make_cmd_arg(key, val) for key, val in BASE_CONFIG.items() if key not in config]
    cmd_list += [_make_cmd_arg(key, val) for key, val in config.items()]
    command = " ".join(cmd_list)

    print("Constructed command: ")
    print(command)
    os.system(command)
