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

torchrun --nproc_per_node=4 train_kd.py --model_size 135 --train_file_count 100 --load_dir_name 'run_dpo_online_a_k_seed0' --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 1 --type 'distilled' --temp 0.95

torchrun --nproc_per_node=4 train_kd.py --model_size 135 --train_file_count 100 --load_dir_name 'run_rlhf_ppo_a_k_seed0' --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 1 --type 'distilled' --temp 0.95

