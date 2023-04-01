import json
import copy
from typing import List
import datetime
import jax
import haiku as hk
from src.haiku_numpyro_mlp import build_forward_fn, generate_input_data
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import linspaced_itemps_by_n


def vary_configs_ns(partial_config, ns):
    configs = []
    for n in ns:
        new_config = copy.deepcopy(partial_config)
        new_config["num_training_data"] = n
        configs.append(new_config)
    return configs

def vary_configs_itemps(partial_config, num_itemps):
    configs = []
    n = partial_config["num_training_data"]
    itemps = linspaced_itemps_by_n(n, num_itemps)
    for itemp in itemps:
        new_config = copy.deepcopy(partial_config)
        new_config["itemp"] = itemp
        configs.append(new_config)
    return configs

def vary_configs_rngseed(partial_config, rng_seed_list):
    configs = []
    for seed in rng_seed_list:
        new_config = copy.deepcopy(partial_config)
        new_config["rng_seed"] = seed
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
    import pickle
    import os
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Generate a JSON list of experiment configurations according to commandline specification."
    )
    parser.add_argument(
        "--true_param_filepath",
        default=None,
        type=str,
        help="Path to a .pkl (pickle) file containing the parameter dictionary of the true parameter. If specified, the true parameter won't be generated anew.",
    )
    parser.add_argument(
        "--expt_name",
        default=None,
        type=str,
        help="A string to specify a name to give the experiment configurations. If not provided a new on is auto generated.",
    )
    parser.add_argument("--num_training_data_list", nargs="+", required=True, type=int)
    parser.add_argument(
        "--rng_seed_list", 
        nargs="+",
        required=True, 
        type=int, 
        help="A list of PRNG seeds for each repetitions of the experiments."
    )
    parser.add_argument("--input_dim", nargs="?", default=1, type=int, help="Dimension of the input data X.")
    parser.add_argument(
        "--layer_sizes",
        nargs="+",
        type=int,
        help="A list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. ",
    )
    parser.add_argument("--sigma_obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior_std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation_fn_name", nargs="?", default="tanh", type=str)
    parser.add_argument("--num_itemps", nargs="?", default=6, type=int)
    parser.add_argument("--num_posterior_samples", nargs="?", default=2000, type=int)
    parser.add_argument("--thinning", nargs="?", default=1, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=1, type=int)
    parser.add_argument(
        "--output_dir",
        required=True, 
        type=str,
        help="a directory for storing generated configurations and true parameter files.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.true_param_filepath is None:
        
        true_param = generate_random_param(
            args.rng_seed_list[-1], args.layer_sizes, args.input_dim, args.prior_mean, args.prior_std
        )
        print(f"Generated new true parameter: {true_param}")
    
        # Save the new true parameter tree to a file
        str_layer_size = '_'.join([str(s) for s in args.layer_sizes])
        true_param_filepath = os.path.join(
            args.output_dir, 
            f"params_{args.input_dim}_{str_layer_size}.pkl"
        )
        with open(true_param_filepath, 'wb') as f:
             pickle.dump(true_param, f)
    else:
        # Copy the specified file to the output directory. 
        true_param_filepath = os.path.join(
            args.output_dir, 
            os.path.basename(args.true_param_filepath)
        )
        shutil.copy(args.true_param_filepath, true_param_filepath)

    expt_name = args.expt_name if args.expt_name is not None else "expt"

    partial_config = generate_config_realisable_1hltanh_expt(
        expt_name=expt_name, 
        rng_seed=args.rng_seed_list[0], 
        itemp=1.0, # dummy value for now.
        num_training_data=args.num_training_data_list[0], 
        true_param_filepath=true_param_filepath, 
        sigma_obs=args.sigma_obs, 
        input_dim=args.input_dim, 
        layer_sizes=args.layer_sizes, 
        prior_mean=args.prior_mean, 
        prior_std=args.prior_std, 
        num_posterior_samples=args.num_posterior_samples, 
        num_warmup=args.num_warmup, 
        num_chains=args.num_chains, 
        thinning=args.thinning
    )
    config_list = vary_configs_ns(partial_config, args.num_training_data_list)
    config_list = sum(
        [vary_configs_rngseed(partial_config, args.rng_seed_list) 
         for partial_config in config_list], []
    )
    config_list = sum(
        [vary_configs_itemps(partial_config, args.num_itemps) for partial_config in config_list], []
    )
    print(f"Number of experiment configs: {len(config_list)}")

    filepath = os.path.join(args.output_dir, f"config_list.json")
    with open(filepath, "w") as outfile:
        json.dump(config_list, outfile, indent=4)
    