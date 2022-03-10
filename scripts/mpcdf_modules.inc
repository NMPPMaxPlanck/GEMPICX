source /etc/profile
module purge
if [ x"$CLUSTER" == x"COBRA"  ]; then
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5
elif [ x"$CLUSTER" == x"RAVEN" ]; then
module load intel/21.4.0
module load impi/2021.4
module load mkl/2021.2
else
module load intel/21.3.0
module load impi/2021.3
module load mkl/2021.3
fi
module load cmake/3.18
module load gcc
module load git
module load numdiff
module list
