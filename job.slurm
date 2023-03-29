#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="test_job_name"

# The project ID which this job should run under:
#SBATCH --account="punim0715"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The amount of memory in megabytes per node:
#SBATCH --mem=4096

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-0:10:00

#SBATCH --output=test_job_name_output.txt
#SBATCH --error=test_job_name_error.txt


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
python run_experiment.py --config_filepath ./sample_experiment_config.json --output_dir ./outputs/test_$(date +"%Y%m%d_%H%M")/ --host_device_count 4 --save_posterior_samples

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s