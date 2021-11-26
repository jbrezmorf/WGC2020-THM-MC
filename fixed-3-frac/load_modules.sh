#!/bin/bash
echo "source this script: 'source load_modules.sh'"

x=`declare -F | grep "declare -fx module"`
if [ -n "$x" ]
then
    module -fs purge
    module load metabase
    module load cmake-3.6.1
    module load gcc-6.4.0
    module load boost-1.60-gcc
    module load mpich-3.0.2-gcc
    module rm python-2.7.6-gcc
    module load python-3.6.2-gcc
    module load python36-modules-gcc
    module list
fi
