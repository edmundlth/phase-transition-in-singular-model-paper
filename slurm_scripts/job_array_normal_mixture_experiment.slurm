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
#SBATCH --cpus-per-task=6

# The amount of memory in megabytes per node:
#SBATCH --mem=4096

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-01:00:00

#SBATCH --output=nmm_expt/slurm_%A_%a.out
#SBATCH --error=nmm_expt/slurm_%A_%a.err
#SBATCH --array=10-1300:10


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
NUMRNGSEEDS=50
RNGSEEDS=$( seq 1 ${NUMRNGSEEDS} )
OUTPUTDIRPATH=nmm_expt/20230509/
HOSTDEVICECOUNT=6
DELTA=0.5
python normal_mixture_experiment.py --outdirpath ${OUTPUTDIRPATH} --num_training_data ${SLURM_ARRAY_TASK_ID} --delta ${DELTA} --num_chains ${HOSTDEVICECOUNT} --host_device_count ${HOSTDEVICECOUNT} --rng_seeds ${RNGSEEDS}

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s