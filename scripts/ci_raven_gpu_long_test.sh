#!/bin/bash

: > ci_job.out
sbatch --wait ./ci_raven_gpu_jobscript_long_test.sh
EXIT_CODE=$?

cat ci_job.out

exit $EXIT_CODE

