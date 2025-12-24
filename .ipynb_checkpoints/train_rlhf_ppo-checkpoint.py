import argparse
import os
import torch
import wandb
import math
from datasets import Dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, AutoConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from torch.nn import CrossEntropyLoss
import numpy as np
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Argument Parsing Configuration ===
def parse_args():
    parser = argparse.ArgumentParser(description="LLM RLHF Pipeline A: KD -> Alignment")

    # --- Model Paths ---
    parser.add_argument(
        "--sft_model_path", type=str, required=True,
        help="Path to the model to be aligned (Actor). For this experiment, it's the 135M model that was distilled from the 360M model.",
    )
    parser.add_argument(
        "--reward_model_path", type=str, required=True,
        help="Path to the target distribution model for reward calculation. For this experiment, it's the pretrained 135M target model.",
    )

    # --- PPO Hyperparameters ---
    parser.add_argument("--hp_dir", type=str, default=None, help="Path to an alternative cache directory for Hugging Face models.")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Path to an alternative directory for wandb data.")
    parser.add_argument("--ppo_iterations", type=int, default=1000, help="Number of PPO training iterations.") 
    parser.add_argument("--ppo_epochs", type=int, default=1, help="Number of optimization epochs per PPO batch.")
    parser.add_argument("--model_size", type=int, default=135, help="Size of the SmolLM2 model to load the tokenizer from Hub (e.g., 135 for 135M).")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the actor and critic.")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient (from PPOConfig).")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="PPO mini-batch size for optimization.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--kl_penalty", type=str, default="kl", choices=["kl", "abs", "mse", "full"], help="KL penalty type.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--reward_pen_lamb", type=float, default=0.5)
    parser.add_argument("--reward_tau", type=float, default=4)

    # --- Generation and Environment Settings ---
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation.")
    parser.add_argument("--max_gen_length", type=int, default=512, help="Max length of generated responses.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    
    # --- Reward Function Hyperparameters ---
    parser.add_argument(
        "--reward_scaling_factor", type=float, default=0.25,
        help="Scaling factor 'C' for the exponential reward function: reward = 10 * exp(-C * nll).",
    )

    # --- Logging and Saving ---
    parser.add_argument("--output_dir", type=str, default="./ppo_aligned_model", help="Directory to save the final aligned model.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save a checkpoint every N iterations.") 
    parser.add_argument("--wandb_project", type=str, default="smollm2-rlhf-pipeline-A", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="kd-align-run", help="W&B run name.")

    # Use parse_known_args() instead of parse_args() to ignore arguments added by DeepSpeed.
    args, _ = parser.parse_known_args()
    return args

@torch.no_grad()
def get_reward_signal_response_only(
    prompts: List[str],
    responses: List[str],
    reward_model,
    reward_tokenizer,
    device,
    scaling_factor: float,
    max_length: int = 512,
) -> torch.FloatTensor:
    # 1) Tokenize prompts and responses
    inputs = reward_tokenizer([p + r for p, r in zip(prompts, responses)],
                              return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    # 2) Calculate prompt lengths (in tokens)
    prompt_lens = []
    for p in prompts:
        enc = reward_tokenizer(p, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
        prompt_lens.append(int(enc["input_ids"].size(1)))

    # 3) Compute logits
    logits = reward_model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits, shift_labels = logits[:, :-1, :].contiguous(), input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()
    B, Tm1 = shift_labels.size()

    # 4) Create response masks (1 only for tokens after the prompt length)
    resp_mask = torch.zeros((B, Tm1), dtype=shift_attn.dtype, device=device)
    for i, pl in enumerate(prompt_lens):
        resp_mask[i, min(pl, Tm1):] = 1
    final_mask = (resp_mask * shift_attn).bool()

    # 5) Compute per-token NLL (Negative Log Likelihood)
    loss_tok = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none"
    ).view(B, Tm1)

    denom = final_mask.sum(dim=1).clamp_min(1)
    nll_per_sample = (loss_tok.masked_fill(~final_mask, 0).sum(dim=1) / denom)
    nll_per_sample = torch.nan_to_num(nll_per_sample, nan=10.0)

    # 6) Calculate exponential reward
    rewards = 10.0 * torch.exp(-scaling_factor * nll_per_sample)
    return rewards

@torch.no_grad()
def apply_reward_folding(
    base_rewards: torch.FloatTensor,
    tau: float
) -> torch.FloatTensor:
    """
    Applies a folding penalty to rewards exceeding a threshold 'tau'.
    - If reward <= tau, the reward is unchanged.
    - If reward > tau, the reward is reflected across the 'tau' threshold.
      (e.g., if tau=8.0 and reward=8.1, final_reward=7.9)
    """
    # condition: base_rewards > tau
    # value_if_true: 2.0 * tau - base_rewards (the folded value)
    # value_if_false: base_rewards (the original value)
    final_rewards = torch.where(
        base_rewards > tau,
        2.0 * tau - base_rewards,
        base_rewards
    )
    return final_rewards
    
def sanitize_stats(stats: Dict[str, any]) -> Dict[str, any]:
    """Cleans up the statistics dictionary by replacing NaN or Inf values with 0."""
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            sanitized_stats[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            sanitized_stats[key] = value
    return sanitized_stats


def main():
    args = parse_args()
    set_seed(args.seed)
    
    wandb_kwargs = {
        "wandb": {
            "name": args.wandb_run_name,
            "config": vars(args),
        }
    }

    ppo_config = PPOConfig(
        model_name=args.sft_model_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        kl_penalty=args.kl_penalty,
        init_kl_coef=args.beta,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        log_with="wandb",
        tracker_project_name=args.wandb_project,
        tracker_kwargs=wandb_kwargs,
        use_score_norm=True,
        max_grad_norm=0.5,
    )

    
    tokenizer_hf_path = f"HuggingFaceTB/SmolLM2-{args.model_size}M"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.float32
    
    sft_config = AutoConfig.from_pretrained(args.sft_model_path, trust_remote_code=True, local_files_only=True)
    sft_base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        config=sft_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_base_model)
    
    reward_config = AutoConfig.from_pretrained(args.reward_model_path, trust_remote_code=True, local_files_only=True)
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model_path,
        config=reward_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True
    )

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model_path, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    )
    
    ppo_trainer = PPOTrainer(
        config=ppo_config, 
        model=model, 
        ref_model=ref_model, 
        tokenizer=tokenizer, 
        dataset=None, 
        data_collator=None,
    )

    if ppo_trainer.accelerator.is_main_process:
        print(f"--- Loading Tokenizer from Hugging Face Hub: {tokenizer_hf_path} ---")
        print(f"--- Loading SFT (Actor) model from local path: {args.sft_model_path} ---")
        print(f"--- Loading Reward model from local path: {args.reward_model_path} ---")
        samples_table = wandb.Table(columns=["iteration", "prompt", "response"])

    reward_model.to(ppo_trainer.accelerator.device)
    reward_model.eval()

    generation_kwargs = {
        "min_length": -1, "top_k": model.config.vocab_size, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128, "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,   
        "no_repeat_ngram_size": 0,    
        "renormalize_logits": True,   
    }
    if ppo_trainer.accelerator.is_main_process:
        print("--- Starting PPO Training (Alignment with fixed 'The' prompt) ---")
        
    prompt_text = "The"
    query_tensor = tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0)
    queries = [query_tensor.clone() for _ in range(args.batch_size)]

    for iteration in tqdm(range(args.ppo_iterations)):
        response_tensors = ppo_trainer.generate(queries, return_prompt=False, **generation_kwargs)
        prompt_texts = [prompt_text] * args.batch_size
        response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        texts_for_reward = [q + r for q, r in zip(prompt_texts, response_texts)]
        
        rewards_base = get_reward_signal_response_only(
            prompts=prompt_texts,
            responses=response_texts,
            reward_model=reward_model,
            reward_tokenizer=tokenizer,
            device=ppo_trainer.accelerator.device,
            scaling_factor=args.reward_scaling_factor,
            max_length=512,
        )

        rewards = apply_reward_folding(
            base_rewards=rewards_base,
            tau=args.reward_tau  
        )

        batch_for_log = {
            "query": prompt_texts,
            "response": response_texts,
        }
        
        rewards_list = [r for r in rewards]
        
        unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
        before_params = sum(p.abs().sum() for p in unwrapped_model.pretrained_model.parameters())

        stats = ppo_trainer.step(queries, response_tensors, rewards_list)
        
        after_params = sum(p.abs().sum() for p in unwrapped_model.pretrained_model.parameters())
            
        mean_reward = rewards.mean().item()
        total_loss = stats.get("ppo/loss/total", 0.0)
        if hasattr(total_loss, "item"):
            total_loss = total_loss.item()
        current_lr = ppo_trainer.optimizer.param_groups[0]['lr']
        stats['ppo/learning_rate'] = current_lr

        lengths = [len(tokenizer.encode(r, add_special_tokens=False)) for r in response_texts]

        if ppo_trainer.accelerator.is_main_process:
            params_changed = not torch.isclose(before_params, after_params)
            tqdm.write(f"Iteration {iteration+1}/{args.ppo_iterations} | Mean Reward: {mean_reward:.4f} | Total Loss: {total_loss:.4f} | LR: {current_lr:.2e} | Param changed: {params_changed} | len mean/std/min/max: {np.mean(lengths):.4f}, {np.std(lengths):.4f}, {np.min(lengths):.4f}, {np.max(lengths):.4f}")
            
        ppo_trainer.log_stats(stats, batch_for_log, rewards)

        if ppo_trainer.accelerator.is_main_process:
            rows = []
            if iteration%10 == 0:
                sample_n = min(8, len(response_texts))
   
                rows.extend([[iteration+1, prompt_texts[i], response_texts[i]] for i in range(sample_n)])
                wandb.log({"samples/responses": wandb.Table(columns=["iteration","prompt","response"], data=rows)},
                      step=iteration+1)

        if (iteration + 1) % args.save_interval == 0:
            if ppo_trainer.accelerator.is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{iteration + 1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"\n--- Saving checkpoint at iteration {iteration + 1} to {checkpoint_dir} ---")

                unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)


    if ppo_trainer.accelerator.is_main_process:
        print("--- PPO Training Finished ---")
        print(f"--- Saving final model to {args.output_dir}/final ---")
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print("--- Experiment Complete ---")


if __name__ == "__main__":
    main()