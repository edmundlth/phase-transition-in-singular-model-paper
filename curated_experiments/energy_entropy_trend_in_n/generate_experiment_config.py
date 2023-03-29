import json
from typing import List
import jax


def generate_realisable_1hltanh_expt_config(
    expt_name: str,
    rng_seed: int,
    itemp: float,
    num_training_data: int,
    true_param_filepath: str,
    sigma_obs: float,
    layer_sizes: List[int],
    prior_mean: float,
    prior_std: float,
    num_posterior_samples: int,
    num_warmup: int,
    num_chains: int,
    thinning: int, 
):
    config = {
        "expt_name": expt_name,
        "rng_seed": rng_seed,
        "itemp": itemp,
        "num_training_data": num_training_data,
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


