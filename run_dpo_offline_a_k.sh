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

deepspeed --num_gpus=1 --master_port=29110 pipeline_a_dpo_new_offline.py \
    --seed 0 \
    --sft_model_path "./pretrained_models/smollm2_360m/seed0" \
    --reward_model_path "./target_models/seed0" \
    --output_dir ./results/run_dpo_offline_a_k_seed0 \
    --wandb_project "smollm2-dpo-offline" \
    --wandb_run_name "run_dpo_offline_a_k_seed0" \
    --model_size 360 \
    --online_iterations 1000 \
    --save_interval 10 \
    --batch_size 64 \
    --num_train_epochs 2 \
    --mini_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --beta 0.7 \
    --num_preference_samples 10000 \
    --temperature 1.0 


