import argparse
import os
import torch
import wandb
import numpy as np
from datasets import Dataset, disable_caching
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from tqdm import tqdm
from typing import List, Dict

disable_caching()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Argument Parsing Configuration (Optimized for Online DPO) ===
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Online DPO Alignment")

    # --- Model Paths ---
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the model to be aligned.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model for labeling.")

    # --- DPO and Training Loop Hyperparameters ---
    parser.add_argument("--online_iterations", type=int, default=200, help="Total number of online generation/training iterations.")
    parser.add_argument("--steps_per_online_batch", type=int, default=8, help="Number of gradient update steps per online data batch.")
    parser.add_argument("--model_size", type=int, default=135, help="Size of the model to load the tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate for DPO.")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient for DPO.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for online data generation (creates batch_size pairs).")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Mini-batch size for DPO training step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")

    # --- Generation and Environment Settings ---
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation.")
    parser.add_argument("--max_gen_length", type=int, default=128, help="Max length of generated responses.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    
    # --- Logging and Saving ---
    parser.add_argument("--output_dir", type=str, default="./dpo_aligned_model", help="Directory to save the final aligned model.")
    parser.add_argument("--wandb_project", type=str, default="smollm2-online-dpo", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="online-dpo-run", help="W&B run name.")
    
    # Save checkpoint every N online iterations
    parser.add_argument("--save_interval", type=int, default=100, help="Save a checkpoint every N online iterations.")
    parser.add_argument("--reward_scaling_factor", type=float, default=0.5,
                        help="Scaling factor for exp reward logging (PPO-style monitor).")

    
    args, _ = parser.parse_known_args()
    return args

    
@torch.no_grad()
def get_reward_scores(prompts: List[str], responses: List[str], reward_model, reward_tokenizer) -> torch.FloatTensor:
    """
    Returns per-sample scores = -NLL(response | prompt).
    Calculates the average NLL over the response tokens and returns the negated value as the score.
    """
    batch = []
    for p, r in zip(prompts, responses):
        p_ids = reward_tokenizer(p, add_special_tokens=False)["input_ids"]
        r_ids = reward_tokenizer(r, add_special_tokens=False)["input_ids"]
        # [prompt || response[:-1]] / labels = [-100... | response[1:]]
        input_ids = p_ids + r_ids[:-1]
        labels    = [-100] * len(p_ids) + r_ids[1:]
        attn_mask = [1] * len(input_ids)
        batch.append((input_ids, attn_mask, labels))

    max_len = max(len(x[0]) for x in batch)
    pad_id = reward_tokenizer.pad_token_id or reward_tokenizer.eos_token_id

    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    input_ids = torch.tensor([pad(x[0], pad_id) for x in batch], device=device)
    attn_mask = torch.tensor([pad(x[1], 0) for x in batch], device=device)
    labels    = torch.tensor([pad(x[2], -100) for x in batch], device=device)

    outputs = reward_model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    tgt    = input_ids[:, 1:].contiguous()
    mask   = (labels[:, 1:] != -100)  # Mask for response tokens only

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(logits.view(-1, logits.size(-1)), tgt.view(-1)).view(tgt.size())
    token_nll = token_nll * mask

    # Calculate sample-wise average NLL (over response tokens only)
    denom = mask.sum(dim=1).clamp_min(1)
    seq_nll = token_nll.sum(dim=1) / denom
    scores = -seq_nll  # Higher score is better
    return scores  # shape: [batch]


def tokenize_row(feature: Dict, tokenizer) -> Dict:
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)
    chosen_ids = tokenizer(chosen, add_special_tokens=False)
    rejected_ids = tokenizer(rejected, add_special_tokens=False)

    # concat: [prompt || response]
    chosen_input_ids = prompt_ids["input_ids"] + chosen_ids["input_ids"]
    rejected_input_ids = prompt_ids["input_ids"] + rejected_ids["input_ids"]

    chosen_attention_mask = prompt_ids["attention_mask"] + chosen_ids["attention_mask"]
    rejected_attention_mask = prompt_ids["attention_mask"] + rejected_ids["attention_mask"]

    # Mask prompt section with -100
    chosen_labels = [-100] * len(prompt_ids["input_ids"]) + chosen_ids["input_ids"]
    rejected_labels = [-100] * len(prompt_ids["input_ids"]) + rejected_ids["input_ids"]

    return {
        "prompt_input_ids": prompt_ids["input_ids"],
        "prompt_attention_mask": prompt_ids["attention_mask"],

        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,

        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }

# === Main Execution Function ===
def main():
    args = parse_args()
    set_seed(args.seed)

    # --- 1. Load Model and Tokenizer (Executed once) ---
    print("--- Loading models and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(f"HuggingFaceTB/SmolLM2-{args.model_size}M")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    reward_model = AutoModelForCausalLM.from_pretrained(args.reward_model_path)
    reward_model.to(device).eval()
    
    # --- 2. Initialize DPOTrainer (Executed once) ---
    total_steps = args.online_iterations * args.steps_per_online_batch

    
    dpo_config = DPOConfig(
        beta=args.beta,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=total_steps,
        logging_strategy="steps",
        logging_steps=total_steps + 1,  # Set very high to disable automatic logging
        save_strategy="no",   
        save_steps=100,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="wandb"
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_dict({'prompt': [], 'chosen': [], 'rejected': []}),
        max_prompt_length=64,
        max_length=args.max_gen_length + 64,
    )

    dpo_trainer.create_optimizer_and_scheduler(num_training_steps=dpo_config.max_steps)
    
    # Use accelerator.prepare manually for custom training loop
    (dpo_trainer.model,
     dpo_trainer.optimizer,
     dpo_trainer.lr_scheduler) = dpo_trainer.accelerator.prepare(
        dpo_trainer.model, dpo_trainer.optimizer, dpo_trainer.lr_scheduler
    )
    
    dpo_trainer.accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.wandb_run_name}}
    )
    
    if dpo_trainer.accelerator.is_main_process:
        # Define table with 'iteration' column
        cumulative_samples_table = wandb.Table(columns=["iteration", "global_step", "prompt", "chosen", "rejected"])
        
    print(f"--- Starting Online DPO Training for {args.online_iterations} iterations ---")
    prompt_text = "The"

    # --- 3. Online Training Loop ---
    for iteration in tqdm(range(args.online_iterations), desc="Online DPO Iterations"):
        # 3-1. Dynamic Data Pool Generation from Current Policy
        prompt_tensors = tokenizer(prompt_text, return_tensors="pt").input_ids.to(dpo_trainer.accelerator.device)
        generation_kwargs = {
            "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": args.max_gen_length, "temperature": args.temperature,
        }
        
        data_pool_size = args.batch_size * 2
        unwrapped_model = dpo_trainer.accelerator.unwrap_model(dpo_trainer.model)
        response_tensors = unwrapped_model.generate(
            torch.cat([prompt_tensors] * data_pool_size), **generation_kwargs
        )
        responses = tokenizer.batch_decode(response_tensors[:, prompt_tensors.shape[1]:], skip_special_tokens=True)
        
        # 3-2. Generate Preference Pairs from Pool
        
        # Simple prompt list matching pool size
        prompts = [prompt_text] * data_pool_size
        
        scores = get_reward_scores(prompts, responses, reward_model, tokenizer)
        
        # 3-3. Construct Preference Pairs
        preference_data = {'prompt': [], 'chosen': [], 'rejected': []}
        chosen_idx = []
        rejected_idx = []
        for i in range(0, data_pool_size, 2):
            if scores[i] > scores[i + 1]:
                c, r = i, i + 1
            else:
                c, r = i + 1, i
            preference_data['prompt'].append(prompt_text)
            preference_data['chosen'].append(responses[c])
            preference_data['rejected'].append(responses[r])
            chosen_idx.append(c)
            rejected_idx.append(r)

        # === Reward/NLL Pre-logging (Quality before update) ===
        sf = args.reward_scaling_factor
        with torch.no_grad():
            rewards_all = 10.0 * torch.exp(sf * scores)   # = 10 * exp(-sf * nll)
            nll_all     = (-scores)

        cr = rewards_all[chosen_idx].detach().float().cpu().numpy()
        rr = rewards_all[rejected_idx].detach().float().cpu().numpy()
        cn = nll_all[chosen_idx].detach().float().cpu().numpy()
        rn = nll_all[rejected_idx].detach().float().cpu().numpy()

        reward_margin = cr - rr
        nll_margin = rn - cn  # (rejected_nll - chosen_nll), higher is better

        if dpo_trainer.accelerator.is_main_process:
            gs_pre = int(dpo_trainer.state.global_step)
            log_blob = {
                "reward/pre/mean_chosen": float(np.mean(cr)),
                "reward/pre/mean_rejected": float(np.mean(rr)),
                "reward/pre/mean_margin": float(np.mean(reward_margin)),
                "reward/pre/std_chosen": float(np.std(cr)),
                "reward/pre/std_rejected": float(np.std(rr)),
                "reward/pre/std_margin": float(np.std(reward_margin)),
                "reward/pre/nll_mean_chosen": float(np.mean(cn)),
                "reward/pre/nll_mean_rejected": float(np.mean(rn)),
                "reward/pre/nll_margin_mean": float(np.mean(nll_margin)),
                "iteration/idx": iteration + 1,
            }
            if (iteration + 1) % 50 == 0:
                log_blob.update({
                    "reward/pre/hist_chosen": wandb.Histogram(cr),
                    "reward/pre/hist_rejected": wandb.Histogram(rr),
                    "reward/pre/hist_margin": wandb.Histogram(reward_margin),
                })
            dpo_trainer.log(log_blob)

        # Memory Cleanup
        del response_tensors
        # scores are kept as they are used later, but rewards/nll calculation is done
        torch.cuda.empty_cache()
            
        online_dataset_raw = Dataset.from_dict(preference_data)
        online_dataset = online_dataset_raw.map(tokenize_row, fn_kwargs={"tokenizer": tokenizer}, num_proc=4)

        # 1. Replace trainer's dataset with new online dataset
        dpo_trainer.train_dataset = online_dataset
        
        # 2. Get dataloader (function call without args uses the replaced dataset)
        dataloader = dpo_trainer.get_train_dataloader()
        
        # Ensure model is in train mode
        dpo_trainer.model.train()
    
        dataloader_iterator = iter(dataloader)
        metrics_this_iteration = []

        
        for step in range(args.steps_per_online_batch):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                batch = next(dataloader_iterator)
    
            # Use accumulate context for proper gradient accumulation/sync
            with dpo_trainer.accelerator.accumulate(dpo_trainer.model):
                loss, metrics = dpo_trainer.get_batch_loss_metrics(
                    dpo_trainer.model, batch, train_eval="train"
                )
                dpo_trainer.accelerator.backward(loss)
                
                if dpo_trainer.accelerator.sync_gradients:
                    dpo_trainer.accelerator.clip_grad_norm_(dpo_trainer.model.parameters(), 1.0)
                    
                    dpo_trainer.optimizer.step()
                    dpo_trainer.optimizer.zero_grad()
                    dpo_trainer.lr_scheduler.step()
    
                    # Manually increment global step
                    dpo_trainer.state.global_step += 1

            if dpo_trainer.accelerator.is_main_process:
                metrics_as_floats = {k: v.item() for k, v in metrics.items()}
                metrics_as_floats["loss"] = loss.item()
                metrics_this_iteration.append(metrics_as_floats)
                
                dpo_trainer.log(metrics_as_floats)
                      
        # 3-6. Iteration Summary and Sample Logging
        if dpo_trainer.accelerator.is_main_process:
            if metrics_this_iteration:
                sum_logs = {}
                count_logs = {}
                for log_dict in metrics_this_iteration:
                    for key, value in log_dict.items():
                        if isinstance(value, (int, float)):
                            sum_logs[key] = sum_logs.get(key, 0.0) + value
                            count_logs[key] = count_logs.get(key, 0) + 1

                avg_log = {f"iteration/{key}": sum_logs[key] / count_logs[key] for key in sum_logs}
                avg_log["iteration/idx"] = iteration + 1
                gs_now = int(dpo_trainer.state.global_step)
                avg_log["trainer/global_step"] = gs_now

                dpo_trainer.log(avg_log)

                # (2) Sample table: create new table and commit with step
                if (iteration + 1) % 50 == 0:
                    samples_table = wandb.Table(columns=["iteration", "global_step", "prompt", "chosen", "rejected"])
                    sample_n = min(4, len(preference_data['chosen']))
                    for i in range(sample_n):
                        samples_table.add_data(
                            iteration + 1,
                            gs_now,
                            preference_data['prompt'][i],
                            preference_data['chosen'][i],
                            preference_data['rejected'][i],
                        )
                    # Log directly to W&B (safe in multi-process via accelerator.log)
                    dpo_trainer.log({"samples/dpo_pairs": samples_table})
            

        # 3-4. Save checkpoint per iteration interval
        if (iteration + 1) % args.save_interval == 0:
            if dpo_trainer.accelerator.is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{iteration + 1}")
                dpo_trainer.save_model(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\n--- Saved checkpoint for iteration {iteration + 1} to {checkpoint_dir} ---")
                    
    # --- 4. Save Final Model ---
    if dpo_trainer.accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        dpo_trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"--- DPO Training Finished. Saving final model to {final_dir} ---")

if __name__ == "__main__":
    main()