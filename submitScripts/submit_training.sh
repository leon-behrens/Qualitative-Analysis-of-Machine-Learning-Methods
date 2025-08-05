#!/bin/bash -l



#####################

# job-array Projektarbeit_multitaskLBC_4

#####################



#SBATCH --job-name=Projektarbeit_multitaskLBC_4



#SBATCH --time=06:00:00

#SBATCH --qos=gpu6hours #1day, 6hours, 30min




#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=40G

#SBATCH --partition=a100

#SBATCH --gres=gpu:1



module purge

module load CUDA/11.8.0


conda activate PA_ML




python_script="/scicore/home/bruder/behleo00/PA/src/main/python/MainAnalysis.py"

python "$python_script"


conda deactivate

exit 0

