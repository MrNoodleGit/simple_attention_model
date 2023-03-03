#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#purpose: run icatcher for one video


models_path=/icatcher_plus/model_scripts

echo $models_path 

params=("${@:2}")

current_param_values=${params[${SLURM_ARRAY_TASK_ID}]}

module load openmind/cuda/11.3

cmd="python3 model_scripts/main.py $current_param_values"

echo $cmd 

$cmd