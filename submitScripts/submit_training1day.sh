#!/bin/bash -l



#####################

# job-array SD35-t1d

#####################



#SBATCH --job-name=SD35-t1d



#SBATCH --time=18:00:00

#SBATCH --qos=gpu1day #1day, 6hours, 30min




#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=40G

#SBATCH --partition=a100

#SBATCH --gres=gpu:1



module purge

module load CUDA/11.8.0


conda activate PA_ML



python_script2="/scicore/home/bruder/behleo00/PA/src/main/python/MainAnalysisSD35.py"

python "$python_script2"


conda deactivate

exit 0

