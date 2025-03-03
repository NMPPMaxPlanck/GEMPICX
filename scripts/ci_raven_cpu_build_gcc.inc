# Common module include file for the ci_raven_cpu_* scripts
#source /etc/profile.d/modules.sh
module purge
module load gcc/12
# module av
# missing module
module load openmpi/4
module load cmake/3.24
module load numdiff/5.9
module load git
module list
