NAME="1htanh_h8_unreal"
export CONFIGDIR=curated_experiments/energy_entropy_trend_in_n/$NAME
export CONFIGPATH=$CONFIGDIR/config_list.json
export SLURMLOGDIR=$CONFIGDIR/slurm_logs
export OUTPUTDIR=$CONFIGDIR
export JOBNAME=$NAME
export CPUPERTASK=8


NUMEXPT=$(cat $CONFIGPATH | jq 'length')
ENDINDEX=$(($NUMEXPT - 1))
echo "Job array last index: $ENDINDEX"
echo "Slurm logs will be stored at: $SLURMLOGDIR"
mkdir -p $SLURMLOGDIR

sed "s|ENDINDEX|$ENDINDEX|g; s|CPUPERTASK|$CPUPERTASK|g; s|SLURMLOGDIR|$SLURMLOGDIR|g" job_array.slurm | sbatch