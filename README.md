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


# `run_experiment.py`
This is a script that runs an experiment to produce an MCMC estimate of the expected NLL of a given truth-data-model-prior configuration at a specified inverse temperature. 

See `./sample_experiment_config.json` for a sample experiment configuration file. Sample commandline arguments for running the script: 

```
python run_experiment.py --config_filepath ./sample_experiment_config.json --output_dir ./outputs/test_20230328/ --host_device_count
```