#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name=smpt

# The project ID which this job should run under:
#SBATCH --account="punim0715"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# The amount of memory in megabytes per node:
#SBATCH --mem=8192

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-00:17:00

#SBATCH --output=slurm_%a_%A.out
#SBATCH --error=slurm_%a_%A.err
#SBATCH --array=0-9


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


JOBNAME="${JOBNAME:-"test_job"}"
DATETIMESTRING=$(date +"%Y%m%d_%H%M")
CONFIGPATH="${CONFIGPATH}"
CONFIGINDEX="${CONFIGINDEX}"
OUTPUTDIR="${OUTPUTDIR:-"./outputs"}"

# The job command(s):
python run_experiment.py --config_filepath ${CONFIGPATH} --config_index ${SLURM_ARRAY_TASK_ID} --output_dir ${OUTPUTDIR}/${JOBNAME}_${SLURM_ARRAY_TASK_ID}_${DATETIMESTRING}/ --host_device_count 4 --save_posterior_samples --save_training_data --quiet

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s