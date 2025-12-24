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

deepspeed --num_gpus=1 --master_port=29190 train_rlhf_ppo.py \
    --seed 0 \
    --sft_model_path "./pretrained_models/smollm2_360m/seed0" \
    --reward_model_path "./target_models/seed0" \
    --output_dir ./results/run_rlhf_ppo_a_k_seed0 \
    --wandb_project "smollm2-rlhf-ppo" \
    --wandb_run_name "run_rlhf_ppo_k_a_seed0" \
    --model_size 360 \
    --ppo_iterations 1000 \
    --save_interval 10 \
    --batch_size 64 \
    --mini_batch_size 8 \
    --ppo_epochs 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --beta 0.7 \
    --temperature 1.0 \
    --reward_scaling_factor 0.5 \
    --repetition_penalty 1.0 \
    --reward_tau 8.4636

