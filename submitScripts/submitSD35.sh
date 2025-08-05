#!/bin/bash -l

#####################
# Job-array PA_SD35-1
#####################

#SBATCH --job-name=PA_SD35-1
#SBATCH --time=18:00:00
#SBATCH --qos=gpu1day #1day, 6hours, 30min
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --array=76-150

module purge
module load CUDA/11.8.0

conda activate PA_ML

python_script="/scicore/home/bruder/behleo00/PA/src/main/python/MainImageCreationSD35.py"
python "$python_script" "$SLURM_ARRAY_TASK_ID"

conda deactivate

exit 0