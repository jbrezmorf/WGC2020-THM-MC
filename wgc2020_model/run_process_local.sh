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

# main command
command="python3 process.py $run_command $output_dir"

if [ "$sing" == true ]; then
  # SINGULARITY
  # command for running correct docker image
  rep_dir=$(pwd)
  image=$(./wgc2020_fterm image)
  sing_command="singularity exec -B $rep_dir:$rep_dir docker://$image"

  command="$sing_command bash -c \"$command\""
  echo $command
  eval $command

else
  # DOCKER
  ./wgc2020_fterm exec "bash -c \"$command\""
fi
