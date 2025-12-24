#!/bin/bash

########################################################################################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a100|v100"
#SBATCH --job-name=JOB_NAME
#SBATCH --output=hpc_logs/%j.ou
#SBATCH --error=hpc_logs/%j.err

# conda init
# source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# conda activate /scratch/NYU_ID/envs/llms/

########################################################################################################################

python generate_data.py \
  --load_model_name 'run_dpo_online_a_k_seed0' \
  --model_size 360 \
  --temperature 0.95 \
  --batch_size 512 \
  --num_samples 1000000

python generate_data.py \
  --load_model_name 'run_rlhf_ppo_a_k_seed0' \
  --model_size 360 \
  --temperature 0.95 \
  --batch_size 512 \
  --num_samples 1000000