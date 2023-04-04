export CONFIGDIR=curated_experiments/energy_entropy_trend_in_n/1htanh_h4_20230401
export CONFIGPATH=$CONFIGDIR/config_list.json
export SLURMLOGDIR=$CONFIGDIR/slurm_logs
export OUTPUTDIR=$CONFIGDIR
export JOBNAME="1htanh_h4_20230401"
export CPUPERTASK=6

NUMEXPT=$(cat $CONFIGPATH | jq 'length')
ENDINDEX=$(($NUMEXPT - 1))
echo "Job array last index: $ENDINDEX"
echo "Slurm logs will be stored at: $SLURMLOGDIR"
mkdir -p $SLURMLOGDIR

sed "s|ENDINDEX|$ENDINDEX|g; s|CPUPERTASK|$CPUPERTASK|g; s|SLURMLOGDIR|$SLURMLOGDIR|g" job_array.slurm