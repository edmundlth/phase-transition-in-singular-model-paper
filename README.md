# Instructions for running Jupyter notebook
In this repository, `pipenv` is used as the package manager. To install `pipenv` on MacOS, run 
```
$ brew install pipenv
```

To install require package, 
```
$ cd /path/to/repository/root/    # make sure Pipfile.lock is in the current directory. 

$ pipenv install . 
```

To start a Jupyter server, either run `jupyter notebook` in a `pipenv shell`
```
$ pipenv shell
$ jupyter notebook
```

or run 
```
$ pipenv run jupyter notebook
```


# Running the `haiku_mlp_rlct_estimate.py` script for estimating MLP learning coefficients
Script commandline inputs. 
```
$ python haiku_mlp_rlct_estimate.py --help
usage: haiku_mlp_rlct_estimate.py [-h] [--num-itemps [NUM_ITEMPS]] [--num-posterior-samples [NUM_POSTERIOR_SAMPLES]] [--thinning [THINNING]] [--num-warmup [NUM_WARMUP]] [--num-chains [NUM_CHAINS]]
                                  [--num-training-data [NUM_TRAINING_DATA]] [--a0 A0 [A0 ...]] [--b0 B0 [B0 ...]] [--input_dim [INPUT_DIM]] [--layer_sizes LAYER_SIZES [LAYER_SIZES ...]] [--sigma-obs [SIGMA_OBS]]
                                  [--prior-std [PRIOR_STD]] [--prior-mean [PRIOR_MEAN]] [--activation-fn-name [ACTIVATION_FN_NAME]] [--device DEVICE] [--output_dir OUTPUT_DIR] [--show_plot] [--quiet] [--rng-seed [RNG_SEED]]

RLCT estimation of MLP models.

options:
  -h, --help            show this help message and exit
  --num-itemps [NUM_ITEMPS]
                        Number of inverse temperature values (centered aroudn 1/log(n)) to be used in the regression for RLCT estimation.
  --num-posterior-samples [NUM_POSTERIOR_SAMPLES]
                        Number of posterior samples drawn in each MCMC chain.
  --thinning [THINNING]
                        Thinning factor for MCMC sampling.
  --num-warmup [NUM_WARMUP]
                        Number of samples discarded at the begining of each MCMC chain, a.k.a. burn in.
  --num-chains [NUM_CHAINS]
                        Number of independent MCMC chains to run. This will set the number of compute devices used as well.
  --num-training-data [NUM_TRAINING_DATA]
                        Size of the randomly generated training data set.
  --a0 A0 [A0 ...]
  --b0 B0 [B0 ...]
  --input_dim [INPUT_DIM]
                        Dimension of the input data X.
  --layer_sizes LAYER_SIZES [LAYER_SIZES ...]
                        A optional list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. If specified, --a0 and --b0 are ignored.
  --sigma-obs [SIGMA_OBS]
  --prior-std [PRIOR_STD]
  --prior-mean [PRIOR_MEAN]
  --activation-fn-name [ACTIVATION_FN_NAME]
  --device DEVICE       use "cpu" or "gpu".
  --output_dir OUTPUT_DIR
                        a directory for storing output files.
  --show_plot           Do plt.show()
  --quiet               Lower verbosity level.
  --rng-seed [RNG_SEED]
```

Example: 
1. Running the script with default parameters 
```
$ python haiku_mlp_rlct_estimate.py
```
should produce the following output
![Sample output](./sample_rlct_estimation_regression.png)

2. Commandline arguments to change the MCMC configuration
```
$ python haiku_mlp_rlct_estimate.py --num-posterior-samples 4000 --thinning 4 --num-chains 4 --num-warmup 1500
```

3. Changing number of training samples 
```
$ python haiku_mlp_rlct_estimate.py --num-training-data 3123
```

3. To change the true parameters $a_0$ and $b_0$ in $y = \sum_i a_{0i} \tanh(b_{0i} x)$
```
$ python haiku_mlp_rlct_estimate.py --a0 -0.4 --b0 0.0 --num-training-data 5123
```
![Sample output](./sample_rlct_estimation_regression2.png)

4. For MLP with different architecture e.g. input dimension of 2, output dimension 1 and 2 hidden layers with size `[2, 3]` with ReLU activation
```
$ python haiku_mlp_rlct_estimate.py --input_dim 2 --layer_sizes 2 3 1 --activation-fn-name relu
```

# `run_experiment.py`
This is a script that runs an experiment to produce an MCMC estimate of the expected NLL of a given truth-data-model-prior configuration at a specified inverse temperature. 

See `./sample_experiment_config.json` for a sample experiment configuration file. Sample commandline arguments for running the script: 

```
python run_experiment.py --config_filepath ./sample_experiment_config.json --output_dir ./outputs/test_20230328/ --host_device_count
```

To run the script on a SLURM managed HPC, submit the job described in `job.slurm` with 
```
CONFIGPATH=/path/to/config_list.json CONFIGINDEX=${index} JOBNAME="name" OUTPUTDIR="./outputs" sbatch job.slurm
```

# `generate_experiment_config.py`
Example
```
python generate_experiment_config.py --expt_name "1htanh_h4" --num_itemps 8 --rng_seed_list 0 1 2 3 4 --num_training_data_list 5000 6000 7000 8000 9000 10000 11000 12000 --layer_sizes 4 1 --input_dim 1 --num_posterior_samples 5000 --num_chains 4 --num_warmup 500 --thinning 4 --output_dir ./curated_experiments/energy_entropy_trend_in_n/1htanh_h4_20230330/
```