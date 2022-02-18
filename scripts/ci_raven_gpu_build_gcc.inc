# Common module include file for the ci_raven_gpu_* scripts
#source /etc/profile.d/modules.sh
module purge
module load gcc/10
#module load impi/2019.9
module load openmpi/4
module load cuda/11.2
module load cmake/3.22
module load numdiff/5.9
module list
