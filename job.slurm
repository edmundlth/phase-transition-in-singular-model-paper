#!/bin/bash
JOBNAME="${ENV_JOBNAME:-"test_job"}"
NODES="${ENV_NODES:-1}"
NTASKS="${ENV_NTASKS:-1}"
CPUPERTASK="${ENV_CPUPERTASK:-4}"

CONFIGPATH="${ENV_CONFIGPATH}"
CONFIGINDEX="${ENV_CONFIGINDEX}"

DATETIMESTRING=$(date +"%Y%m%d_%H%M")


# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=$NODES

# The name of the job:
#SBATCH --job-name=$JOBNAME

# The project ID which this job should run under:
#SBATCH --account="punim0715"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=$CPUPERTASK

# The amount of memory in megabytes per node:
#SBATCH --mem=$((1024 * $NTASK * $CPUPERTASK))

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-1:30:00

#SBATCH --output=${JOBNAME}_stdout.txt
#SBATCH --error=${JOBNAME}_stderr.txt


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
python run_experiment.py --config_filepath ${CONFIGPATH} --config_index ${CONFIGINDEX} --output_dir ./outputs/${JOBNAME}_${DATETIMESTRING}/ --host_device_count ${CPUPERTASK} --save_posterior_samples --save_training_data --quiet

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s