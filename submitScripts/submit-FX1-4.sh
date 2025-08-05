#!/bin/bash -l

#####################
# Job-array PA_FX1-4
#####################

#SBATCH --job-name=PA_FX1-4
#SBATCH --time=06:00:00
#SBATCH --qos=gpu6hours #1day, 6hours, 30min
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --array=


module purge
module load CUDA/11.8.0

conda activate PA_ML

python_script="/scicore/home/bruder/behleo00/PA/src/main/python/MainImageCreationFLUX1.py"
python "$python_script" "$SLURM_ARRAY_TASK_ID"

conda deactivate

exit 0

