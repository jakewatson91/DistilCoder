#!/bin/bash
#SBATCH -J distil_coder
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=yli15
#SBATCH --export=ALL
#SBATCH --gres=gpu:H200:1

#SBATCH -D /home/jwatson/DistilCoder
#SBATCH -o outputs/%x.%j.out

exec 2>&1
set -euo pipefail

source /etc/profile.d/modules.sh
module purge
# Load CUDA module (using specific version found on cluster)
module load cuda12.1/toolkit/12.1.1
module list

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate distil-coder

PROJECT_DIR="/home/jwatson/DistilCoder"
cd "$PROJECT_DIR"
export PYTHONPATH=.

# HF token -> env var your YAML expects (huggingface.token_env: HF_TOKEN)
HF_TOKEN_FILE="$HOME/.secrets/hf_token"
if [ -s "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
else
  echo "ERROR: No HF token found at $HF_TOKEN_FILE"
  exit 1
fi
export HF_HOME="$HOME/.cache/huggingface"

# --- Step 1: Generate Synthetic Data (Teacher) ---
echo "Step 1: Generating synthetic data..."
python scripts/generate_data.py

# --- Step 2: Filter Data (Unit Tests) ---
echo "Step 2: Filtering generated data..."
python scripts/filter_data.py --workers 16

# --- Step 3: Train Student Model ---
echo "Step 3: Training student model..."
python scripts/train_student.py
