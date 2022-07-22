# Common module include file for the ci_raven_cpu_* scripts
#source /etc/profile.d/modules.sh
module purge
module load gcc/11
module av
# missing module
module load openmpi/4
module load cmake/3.22
module load numdiff/5.9
module load git
module list
