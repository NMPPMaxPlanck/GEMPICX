#!/bin/bash
#set -x

USAGE="Example usage:
  $0 -p mpcdf-gpu-1D # preset chosen by full name
  $0 -s gr1 # gpu release 1D
  $0 -s cd2 -D GEMPIC_USE_LTO=ON # cpu debug 2D, extra CMAKE flag supplied"

# Default:
PRESET="mpcdf-gpu-3D"

HELP="$0 [-h] [-p preset-name|-s preset-abbreviation] [-D CMAKE_SETTINGS]
configures and builds specific preset, then creates an example jobscript in the build folder

where:
  -h show this help text
  -p set the preset with its full name (default: $PRESET)
  -s set the preset with a three-character abbreviation:
       g|c for choosing between GPU and CPU
       r|d for choosing between release and debug
       1|2|3 for choosing dimension
  -D Additional preset settings
$USAGE"

if (return 0 2>/dev/null); then
  echo "Do not source this script" >&2
  echo "$USAGE" >&2
  return 1
fi

set -e

function error_out() {
  echo "$USAGE" >&2
  exit 1
}

#================================================
#============= Reads script options =============
#============== -p for the preset. ==============
#================================================
FULLNAME_GIVEN=""
SHORT_PRESET=""
ADDITIONAL_OPTIONS=""
# Allows -h flag and -p, -s, -D options expecting arguments. All of these are optional.
while getopts ":hp:s:D:" opt; do
  case $opt in
    h) echo "$HELP"
    exit
    ;;
    p) PRESET="$OPTARG"
    FULLNAME_GIVEN=TRUE
    ;;
    s) SHORT_PRESET="$SHORT_PRESET$OPTARG"
    ;;
    D) ADDITIONAL_OPTIONS="$ADDITIONAL_OPTIONS -$opt$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG." >&2
    error_out
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument."
    exit 1
    ;;
  esac
done
#================================================
#================================================
#================================================

#================================================
#=== Convert abbreviated preset to full name ====
#================================================
if [ $SHORT_PRESET ]; then
  # only allow -p preset-name OR -s short_name, not both
  if [ $FULLNAME_GIVEN ]; then
    echo "A full name preset (-p) and a preset abbreviation (-s) cannot both be given!" >&2
    error_out
  elif [ ${#SHORT_PRESET} -gt 3 ]; then
    echo "Too many abbreviated preset options" >&2
    error_out
  fi
  case "$SHORT_PRESET" in
    *c*) case "$SHORT_PRESET" in
           *d*) PRESET="cpu-debug"
           ;;
           *r*) PRESET="cpu-release"
           ;;
           *) echo "No debug/release option specified in abbreviated preset '$SHORT_PRESET'" >&2
           error_out
         esac
    ;;
    *g*) case "$SHORT_PRESET" in
           *d*) echo "Error! No debug options with GPU" >&2
           exit 1
           ;;
           *r*)
           ;;
           *) if [ ${#SHORT_PRESET} -eq 3 ]; then
                echo "Invalid abbreviated preset '$SHORT_PRESET'" >&2
                error_out
              fi
           ;;
         esac
         PRESET="mpcdf-gpu"
    ;;
    *) echo "No CPU/GPU option specified in abbreviated preset '$SHORT_PRESET'" >&2
    error_out
    ;;
  esac
  case "$SHORT_PRESET" in
    *1*) PRESET="$PRESET-1D"
    ;;
    *2*) PRESET="$PRESET-2D"
    ;;
    *3*) PRESET="$PRESET-3D"
    ;;
    *) echo "No dimension option specified in abbreviated preset '$SHORT_PRESET'" >&2
    error_out
    ;;
  esac
  echo "'$SHORT_PRESET' was converted into '$PRESET'"
fi
#================================================
#================================================
#================================================

# Please note (as of July 19 2024):
# On `viper-i` and `raven-i`, you get the equivalent of 6 CPU cores,
# on `viper` and `raven`, you only get the equivalent of 2 CPU cores,
# https://docs.mpcdf.mpg.de/doc/computing/viper-user-guide.html#resource-limits
# https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#resource-limits
# We adapt the number of parallel build processes accordingly:
N_PARALLEL=$(cat /sys/fs/cgroup/cpu,cpuacct/user.slice/user-${UID}.slice/cpu.cfs_{quota,period}_us | tr '\n' ' ' | awk '{ printf"%d\n", $1/$2 }')
echo "$N_PARALLEL processes"

SOURCE_DIRECTORY=`dirname $0`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`

cd $SOURCE_DIRECTORY

source ./scripts/mpcdf_modules.inc

#================================================
# Verify preset option
#================================================
# list cmake presets    takes 1st column of line # >= 3  check if PRESET fully matches one of the 
#                       (which are the preset names)     listed presets
if cmake --list-presets | tail -n+3 | awk '{print $1}' | grep -F -q -x -- "\"$PRESET\""; then
  BUILD_DIR="$SOURCE_DIRECTORY/build/$PRESET"
else
  echo "Invalid preset: '$PRESET'. `cmake --list-presets`" >&2
  exit 1
fi

echo "PRESET: $PRESET"
echo "BUILD_DIR: $BUILD_DIR"

#================================================
# Build
#================================================
t0=`date +%s`

cmake --preset $PRESET $ADDITIONAL_OPTIONS
cmake --build $BUILD_DIR --parallel $N_PARALLEL


# generate run script in BUILD_DIR
rm -f $BUILD_DIR/run_mpcdf.sh
cat $SOURCE_DIRECTORY/scripts/slurm_mpcdf.inc $SOURCE_DIRECTORY/scripts/mpcdf_modules.inc $SOURCE_DIRECTORY/scripts/srun.inc > $BUILD_DIR/run_mpcdf.sh


t1=`date +%s`
dt=$((t1-t0))
echo "Build Wall Clock Time : ${dt}"