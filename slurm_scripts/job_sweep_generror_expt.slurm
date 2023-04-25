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
#SBATCH --mem=2048

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-01:30:00

#SBATCH --output=genexpt/1htanhsweep/slurm_%A_%a.out
#SBATCH --error=genexpt/1htanhsweep/slurm_%A_%a.err
#SBATCH --array=0-1399


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
OUTPUTDIR=./genexpt/1htanhsweep/20230425/
python slurmsweep_generror_expt.py ${OUTPUTDIR} ${SLURM_ARRAY_TASK_ID}

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s