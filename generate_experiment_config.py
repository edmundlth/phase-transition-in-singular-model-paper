import json
from typing import List
import datetime
import jax
import haiku as hk
from src.haiku_numpyro_mlp import build_forward_fn, generate_input_data
from src.const import ACTIVATION_FUNC_SWITCH

def generate_realisable_1hltanh_expt_config(
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
    print(dummy_X)
    print(forward.apply(init_param, None, dummy_X))
    return init_param


if __name__ == "__main__":
    rand_param = generate_random_param(5, [2, 1], 2, 0.0, 1.0)
    print(rand_param)