source /etc/profile
module purge
if [ x"$CLUSTER" == x"COBRA"  ]; then
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5
fi
if [ x"$CLUSTER" == x"RAVEN" ]; then
module load intel/21.4.0
module load impi/2021.4
module load mkl/2021.2
fi
module load cmake/3.18
module load gcc
module list
