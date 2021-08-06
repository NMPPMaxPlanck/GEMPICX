#!/bin/bash
#SBATCH --switches=1@24:00:00
#SBATCH --time=00:40:00
#SBATCH --account=pn72ma
#SBATCH --partition=general
#SBATCH --nodes=64
#SBATCH --ntasks=3072

machines=""
for i in $(scontrol show hostnames=$SLURM_JOB_NODELIST); do
        machines=$machines:$i:$SLURM_NTASKS_PER_NODE
done
echo $machines
machines=${machines:1}
echo $machines

module load slurm_setup
mpiexec -n $SLURM_NTASKS ./vlasov_maxwell vlasov.input
