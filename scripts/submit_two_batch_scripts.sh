#!/bin/bash
# Submit a sequence of batch jobs with dependencies
# Adapted from
# https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#batch-jobs-with-dependencies
# Batch job script:
JOB_SCRIPT=./run_gpu_raven.sh
JOB_SCRIPT2=./run_create_space_time.sh
echo "Submitting job chain of batch script ${JOB_SCRIPT} followed by ${JOB_SCRIPT2}:"

JOBID=$(sbatch ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
echo "${JOB_SCRIPT} submitted with id " ${JOBID}

JOBID2=$(sbatch --dependency=afterany:${JOBID} ${JOB_SCRIPT2} 2>&1 | awk '{print $(NF)}')
echo "${JOB_SCRIPT2} submitted with id " ${JOBID2}