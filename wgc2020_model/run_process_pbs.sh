#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:mem=2gb:cgroups=cpuacct
#PBS -l walltime=16:00:00
#PBS -q charon
#PBS -N WGC2020_main
#PBS -j oe

cd /auto/liberec3-tul/home/pavel_exner/WGC2020-THM-MC/wgc2020_model
source load_modules.sh
source env/bin/activate
python process.py run /auto/liberec3-tul/home/pavel_exner/WGC2020-THM-MC/wgc2020_model/samples
