#!/bin/bash
set -x

# output directory
output_dir=$1

source venv/bin/activate
python process.py run "$output_dir"

# set running on metacentrum to False
#sed -i '/run_on_metacentrum:/c\run_on_metacentrum: False' config.yaml
#
## number of Markov chains
#n_chains=$1
## output directory
#output_dir=$2
#
## command
#run=false
#visualize=false
#if [ "$3" == "visualize" ]; then
#  visualize=true
#elif [ "$3" == "run" ]; then
#  run=true
#fi
#
## sing == true => use singularity
## sing == false => use docker
#sing=false
#if [ "$4" == "sing" ]; then
#  sing=true
#fi
#
#
#
#
##command='./endorse_fterm exec bash -c \"source ./venv/bin/activate && python3 -m mpi4py run_all.py $mcmc_config $output_dir 4\"'
##echo $command
##eval $command
##./endorse_fterm exec bash -c "\"source ./venv/bin/activate ; python3 -m mpi4py run_all.py $mcmc_config $output_dir 4\""
##./endorse_fterm exec bash -c "\"which python\""
##command="source ./venv/bin/activate && python --version"
##command="which python"
#
##docker run --rm -it -euid=1000 -egid=1000 -etheme=light -ewho=flow -ehome=/mnt//home/paulie -v //home/paulie:/mnt//home/paulie -w //home/paulie/Workspace/Endorse-2Dtest-Bayes -v //home/paulie/Workspace/Endorse-2Dtest-Bayes://home/paulie/Workspace/Endorse-2Dtest-Bayes -v //home/paulie/Workspace://home/paulie/Workspace flow123d/geomop:master_8c1b58980 bash -c "$command"
#
## run sampling
#if [ "$run" == true ]; then
#  command="source ./venv/bin/activate && export OMP_NUM_THREADS=2 && python3 -m mpi4py run_all.py $output_dir $n_chains"
##  command="source ./venv/bin/activate && python3 -m mpi4py run_all.py $output_dir $n_chains"
#fi
#
## visualize
#if [ "$visualize" == true ]; then
#  command="source ./venv/bin/activate && python3 run_all.py $output_dir $n_chains visualize"
#fi
#
#
#if [ "$sing" == true ]; then
#
#  # command for running correct docker image
#  rep_dir=$(pwd)
#  image=$(./endorse_fterm image)
#  sing_command="singularity exec -B $rep_dir:$rep_dir docker://$image"
#
#  # auxiliary command for opening Python environment inside docker image
##  bash_py="bash -c 'source ./venv/bin/activate &&"
#
#  # run setup, prepare PBS script (locally, single proc)
##  command="$sing_command $bash_py python3 -m mpi4py run_all.py $output_dir $n_chains'"
#  command="$sing_command bash -c \"$command\""
#  echo $command
#  eval $command
#
#else
#    ./endorse_fterm exec "bash -c \"$command\""
#fi
