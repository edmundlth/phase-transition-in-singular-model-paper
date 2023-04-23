#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name=genexpt

# The project ID which this job should run under:
#SBATCH --account="punim0715"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per node:
#SBATCH --mem=4096

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-05:00:00

#SBATCH --output=genexpt/20230420_large_tanh_bias/slurm_%A_%a.out
#SBATCH --error=genexpt/20230420_large_tanh_bias/slurm_%A_%a.err
#SBATCH --array=100-2000:50


# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load python/3.9.5


# Activate existing virtual environment
source /home/elau1/venv/bin/activate


# The job command(s):
NUMRNGSEEDS=5
for seed in $(seq 1 $NUMRNGSEEDS); do python generalisation_error_experiment.py --outdirpath genexpt/20230420_large_tanh_bias/ --true_param_filepath test_1_25_25_15_1_tanh_bias_reg.pkl --true_layer_sizes 25 25 15 1 --layer_sizes 5 1 --with_true_bias --with_bias --num-training-data ${SLURM_ARRAY_TASK_ID} --num-test-samples 8034 --num-posterior-samples 6000 --thinning 3 --num-chains 8 --num-warmup 800 --rng-seed $(( ${SLURM_ARRAY_TASK_ID} + ${seed} )) --sigma-obs 0.3 --prior-std 5.0 --activation-fn-name tanh --num_itemps 4 --plot_rlct_regression ; done

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s