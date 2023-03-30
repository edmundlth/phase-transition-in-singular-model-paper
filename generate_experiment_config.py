import json
import copy
from typing import List
import datetime
import jax
import haiku as hk
from src.haiku_numpyro_mlp import build_forward_fn, generate_input_data
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import linspaced_itemps_by_n


def generate_configs_ns(partial_config, ns):
    configs = []
    for n in ns:
        new_config = copy.deepcopy(partial_config)
        new_config["num_training_data"] = n
        configs.append(new_config)
    return configs

def generate_configs_itemps(partial_config, num_itemps):
    configs = []
    n = partial_config["num_training_data"]
    itemps = linspaced_itemps_by_n(n, num_itemps)
    for itemp in itemps:
        new_config = copy.deepcopy(partial_config)
        new_config["itemp"] = itemp
        configs.append(new_config)
    return configs

def generate_config_realisable_1hltanh_expt(
    expt_name: str,
    rng_seed: int,
    itemp: float,
    num_training_data: int,
    true_param_filepath: str,
    sigma_obs: float,
    input_dim: int, 
    layer_sizes: List[int],
    prior_mean: float,
    prior_std: float,
    num_posterior_samples: int,
    num_warmup: int,
    num_chains: int,
    thinning: int,
    use_datetimesuffix: bool = True,
):
    if use_datetimesuffix:
        now = datetime.datetime.now()
        suffix = now.strftime("%Y%m%d%H%M")
        expt_name += "_" + suffix

    config = {
        "expt_name": expt_name,
        "rng_seed": rng_seed,
        "itemp": itemp,
        "num_training_data": num_training_data,
        "input_dim": input_dim,
        "truth": {
            "model_type": "mlp",
            "model_args": {
                "layer_sizes": layer_sizes,
                "with_bias": False,
                "activation_fn_name": "tanh",
                "sigma_obs": sigma_obs,
                "param_filepath": true_param_filepath,
            },
        },
        "model": {
            "model_type": "mlp",
            "model_args": {
                "layer_sizes": layer_sizes,
                "with_bias": False,
                "activation_fn_name": "tanh",
                "sigma_obs": sigma_obs,
            },
        },
        "prior": {
            "distribution_name": "normal",
            "distribution_args": {"loc": prior_mean, "scale": prior_std},
        },
        "mcmc_config": {
            "num_posterior_samples": num_posterior_samples,
            "num_warmup": num_warmup,
            "thinning": thinning,
            "num_chains": num_chains,
        },
        "output": {"status": -1, "enll": None, "output_directory": None},
    }

    return config


def save_config(config, outfilepath):
    with open(outfilepath, "w") as outfile:
        json.dump(config, outfile)
    return


def generate_random_param(
    rng_seed, layer_sizes, input_dim, prior_mean, prior_std, activation_fn_name="tanh"
):
    forward = hk.transform(
        build_forward_fn(
            layer_sizes=layer_sizes,
            activation_fn=ACTIVATION_FUNC_SWITCH[activation_fn_name],
            initialisation_mean=prior_mean,
            initialisation_std=prior_std,
        )
    )
    dummy_X = generate_input_data(5, input_dim, jax.random.PRNGKey(0))
    init_param = forward.init(jax.random.PRNGKey(rng_seed), dummy_X)
    return init_param


if __name__ == "__main__":
    import sys
    import pickle
    import os
    rng_seed = sys.argv[1]
    layer_sizes = [4, 1]
    input_dim = 1
    prior_std = 2.0
    rand_param = generate_random_param(4, [3, 1], 1, 0.0, prior_std)

    print(rand_param)
    # Save the parameter tree to a file
    now = datetime.datetime.now()
    suffix = now.strftime("%Y%m%d%H%M%S")
    str_layer_size = '_'.join([str(s) for s in layer_sizes])

    outdirpath = "./curated_experiments/energy_entropy_trend_in_n/experiment_configs/"

    filepath = os.path.join(
        outdirpath,
        f"params_{input_dim}_{str_layer_size}_{suffix}.pkl"
    )

    with open(filepath, 'wb') as f:
        pickle.dump(rand_param, f)

    partial_config = generate_config_realisable_1hltanh_expt(
        expt_name="one_hidden_layer_tanh_h=4", 
        rng_seed=rng_seed, 
        itemp=1.0, 
        num_training_data=1234, 
        true_param_filepath=filepath, 
        sigma_obs=0.1, 
        input_dim=1, 
        layer_sizes=layer_sizes, 
        prior_mean=0.0, 
        prior_std=prior_std, 
        num_posterior_samples=2000, 
        num_warmup=1000, 
        num_chains=10, 
        thinning=4, 
        use_datetimesuffix=True
    )
    config_list = generate_configs_itemps(partial_config, 10)

    filepath = os.path.join(outdirpath, "config_list.json")
    with open(filepath, "w") as outfile:
        json.dump(config_list, outfile, indent=4)
    




    