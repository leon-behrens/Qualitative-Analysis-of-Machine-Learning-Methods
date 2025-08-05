#!/bin/bash -l



#####################

# job-array Projektarbeit_multitaskLBC

#####################



#SBATCH --job-name=Projektarbeit_multitaskLBC



#SBATCH --time=06:00:00

#SBATCH --qos=gpu6hours #1day, 6hours, 30min




#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=40G

#SBATCH --partition=a100

#SBATCH --gres=gpu:1



module purge

module load
CUDA

conda activate
PA_ML



python_script="/scicore/home/bruder/behleo00/PA/src/main/python/MainImageCreation.py"

python "$python_script"


conda deactivate

exit 0

