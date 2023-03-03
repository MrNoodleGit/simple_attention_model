#!/bin/bash

project_path=`cat PATHS.txt`
param_dir="$project_path/params/param_vals"

echo $param_dir

param_vals=($(find $param_dir/ -type f))

len=$(expr ${#param_vals[@]} - 1) 

cmd="sbatch --array=0-$len $project_path/run_model.sh $project_path ${param_vals[@]}"

echo $cmd
$cmd