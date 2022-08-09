#!/bin/bash
set -x

# command
run_command=$1

# output directory
output_dir=$2

# sing == true => use singularity
# sing == false => use docker
sing=false
if [ "$3" == "sing" ]; then
  sing=true
fi

#source venv/bin/activate
python process.py "$run_command" "$output_dir"

# set running on metacentrum to False
#sed -i '/run_on_metacentrum:/c\run_on_metacentrum: False' config.yaml


#command="source ./venv/bin/activate && export OMP_NUM_THREADS=2 && python3 -m mpi4py run_all.py $output_dir $n_chains"
command="python3 process.py $run_command $output_dir"


if [ "$sing" == true ]; then

  # command for running correct docker image
  rep_dir=$(pwd)
  image=$(./wgc2020_fterm image)
  sing_command="singularity exec -B $rep_dir:$rep_dir docker://$image"

  command="$sing_command bash -c \"$command\""
  echo $command
  eval $command

else
    ./wgc2020_fterm exec "bash -c \"$command\""
fi
