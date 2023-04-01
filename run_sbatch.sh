export CONFIGDIR=./curated_experiments/energy_entropy_trend_in_n/1htanh_h4_20230401/
export CONFIGPATH=$CONFIGDIR/config_list.json
export OUTPUTDIR=$CONFIGDIR
export JOBNAME="1htanh_h4_20230401"
export CPUPERTASK=8

NUMEXPT=$(cat $CONFIGPATH | jq 'length')
ENDINDEX=$(($NUMEXPT - 1))
echo "Job array last index: $ENDINDEX"
sed "s/ENDINDEX/$ENDINDEX/" job_array.slurm | sbatch