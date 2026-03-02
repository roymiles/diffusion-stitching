#!/bin/bash
#SBATCH --job-name=diff_stitch            # Name of the job
#SBATCH --output=job/eval_job_output_%j.txt     # Standard output file
#SBATCH --error=job/eval_job_error_%j.txt       # Standard error file
#SBATCH --ntasks=1                  # Number of tasks (usually 1 for a simple script)
#SBATCH --partition=camera-long     # Partition to submit the job to (e.g., 'general')
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:h200:4           # Number of GPUs (if required)
#SBATCH --mail-type=ALL             # Send email notifications for all events (start, end, fail)

# Set up the PATH for Conda (if Conda is installed in ~/miniconda3)
export PATH="$HOME/miniconda3/bin:$PATH"
echo $PATH

# Initialize Conda
source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate diff_stitching
echo $CONDA_PREFIX

bash eval/run_eval.sh