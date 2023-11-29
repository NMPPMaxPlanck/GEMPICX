# Common module include file for the ci_raven_gpu_* scripts
#source /etc/profile.d/modules.sh
module purge
module load gcc/11
module load cuda/11.4
# missing module
#module load openmpi_gpu/4
module load openmpi/4
module load cmake/3.22
module load numdiff/5.9
module load git
module load anaconda/3/2021.11
module list
