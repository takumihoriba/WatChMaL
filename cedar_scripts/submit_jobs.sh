#!/bin/bash
# Example script to set up and submit batch jobs

# exit when any command fails
set -e

# initial environment setup
export DATATOOLS=/project/rpp-blairt2k/machine_learning/production_software/DataTools
cd $DATATOOLS/cedar_scripts
source sourceme.sh

# name and output data directory for this run
name=IWCDmPMT_test
data_dir=/project/rpp-blairt2k/machine_learning/data

# Run setup scripts
source setup_jobs.sh "$name" "$data_dir"

# set directory where log files will be saved
export LOGDIR="/scratch/$USER/log/$name/"
mkdir -p "$LOGDIR"
# cd to this directory so SLURM puts its logs there too
cd "$LOGDIR"

# submit jobs with desired options
JOBTIME=`date` sbatch --time=0:59:00 --array=0-9 --job-name=e "$DATATOOLS/cedar_scripts/run_WCSim_job.sh" "$name" "$data_dir" -n 100 -e 100 -E 1000 -P e- -d 2pi -p fix -x 0 -y 0 -z 0
JOBTIME=`date` sbatch --time=0:59:00 --array=0-9 --job-name=mu "$DATATOOLS/cedar_scripts/run_WCSim_job.sh" "$name" "$data_dir" -n 100 -e 100 -E 1000 -P mu- -d 2pi -p fix -x 0 -y 0 -z 0
JOBTIME=`date` sbatch --time=0:59:00 --array=0-9 --job-name=gamma "$DATATOOLS/cedar_scripts/run_WCSim_job.sh" "$name" "$data_dir" -n 100 -e 100 -E 1000 -P gamma -d 2pi -p fix -x 0 -y 0 -z 0