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

deepspeed --num_gpus=1 --master_port=29210 train_dpo_online.py \
    --seed 0 \
    --sft_model_path "./pretrained_models/smollm2_360m/seed0" \
    --reward_model_path "./target_models/seed0" \
    --output_dir ./results/run_dpo_online_a_k_seed0 \
    --wandb_project "smollm2-dpo-online" \
    --wandb_run_name "run_dpo_online_a_k_seed0" \
    --model_size 360 \
    --online_iterations 1000 \
    --save_interval 100 \
    --batch_size 64 \
    --steps_per_online_batch 8 \
    --dpo_epochs 1 \
    --mini_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --beta 0.7 \
    --temperature 1.0 

