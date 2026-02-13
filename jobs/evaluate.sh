#!/bin/bash
#SBATCH -J distil_eval
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --account=yli15
#SBATCH --gres=gpu:H200:1
#SBATCH -D /home/jwatson/DistilCoder
#SBATCH -o outputs/%x.%j.out

source /etc/profile.d/modules.sh
module purge
module load cuda12.1/toolkit/12.1.1

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate distil-coder

export PYTHONPATH=.

echo "Starting Evaluation with vLLM..."

# Run the evaluation script
python scripts/evaluate.py \
    --model_path results/final_student_model/merged \
    --output_dir results/benchmarks

echo "Evaluation generation complete."