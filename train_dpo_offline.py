import argparse
import os
import torch
import wandb
from datasets import Dataset, disable_caching
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, TrainerCallback
from trl import DPOConfig, DPOTrainer
from tqdm import tqdm
from typing import List, Dict

# Disable Hugging Face dataset caching
disable_caching()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Argument Parsing Configuration (Optimized for Offline DPO) ===
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Offline DPO Alignment")

    # --- Model Paths ---
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the model to be aligned.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model for labeling.")

    # --- Offline DPO Hyperparameters ---
    parser.add_argument("--num_preference_samples", type=int, default=50000, help="Number of preference pairs for the offline dataset.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs to train on the offline dataset.")

    parser.add_argument("--model_size", type=int, default=135, help="Size of the model to load the tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate for DPO.")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient for DPO.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data generation.")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Mini-batch size for DPO training step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation.")
    parser.add_argument("--max_gen_length", type=int, default=128, help="Max length of generated responses.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    
    # --- Logging and Saving ---
    parser.add_argument("--output_dir", type=str, default="./dpo_aligned_model", help="Directory to save the final aligned model.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=10, help="Save a checkpoint every N steps.")
    parser.add_argument("--wandb_project", type=str, default="smollm2-offline-dpo", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="offline-dpo-run", help="W&B run name.")
    
    args, _ = parser.parse_known_args()
    return args

# === Reward Scoring Function ===
@torch.no_grad()
def batched_reward_scores(prompts, responses, reward_model, reward_tokenizer, batch_size=64, max_length=512):
    reward_model.eval()
    scores_list = []
    for i in range(0, len(responses), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_resps   = responses[i:i+batch_size]
        with torch.no_grad():
            inputs = reward_tokenizer(
                [p + r for p, r in zip(batch_prompts, batch_resps)],
                return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            logits = reward_model(input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"]).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size(0), -1)
            scores_list.append(-loss.mean(dim=1).detach().cpu())
        # Memory cleanup
        del inputs, logits, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()
    return torch.cat(scores_list)
    

# === DPO Dataset Tokenization Function ===
def tokenize_row(feature: Dict, tokenizer) -> Dict:
    """Tokenizes a single row for DPO, creating labels as copies of input_ids."""
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(prompt + chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(prompt + rejected, add_special_tokens=False)

    return {
        "prompt_input_ids": prompt_tokens["input_ids"],
        "prompt_attention_mask": prompt_tokens["attention_mask"],
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_tokens["input_ids"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_tokens["input_ids"],
    }

# === Callback for Periodic Sample Logging ===
class SampleLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, log_interval=50):
        self.tokenizer = tokenizer
        self.log_interval = log_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Execute only on the main process and at specified intervals
        if state.is_world_process_zero and (state.global_step > 0) and (state.global_step % self.log_interval == 0):
            model = kwargs["model"]
            prompt_text = "The"
            prompt_tensors = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
            
            generation_kwargs = { "max_new_tokens": 128, "temperature": 1.0, "do_sample": True, "pad_token_id": self.tokenizer.pad_token_id }
            
            # Generate 4 samples
            response_tensors = model.generate(torch.cat([prompt_tensors] * 4), **generation_kwargs)
            responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Log to WandB table
            table = wandb.Table(columns=["global_step", "prompt", "generated_response"])
            for resp in responses:
                table.add_data(state.global_step, prompt_text, resp)
            wandb.log({"samples/generations": table}, step=state.global_step)

# === Main Execution Function ===
def main():
    args = parse_args()
    set_seed(args.seed)

    # --- 1. Load Model and Tokenizer ---
    print("--- Loading models and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(f"HuggingFaceTB/SmolLM2-{args.model_size}M")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    reward_model = AutoModelForCausalLM.from_pretrained(args.reward_model_path)
    
    ref_model.to(device).eval()
    reward_model.to(device).eval()
    
    # --- 2. Generate Offline Preference Dataset ---
    print(f"--- Generating {args.num_preference_samples} preference pairs for offline training ---")
    prompt_text = "The"
    
    # Use the initial model (ref_model) to generate data before DPO starts
    prompt_tensors = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generation_kwargs = {
        "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_gen_length, "temperature": args.temperature,
    }
    
    # Generate 2x samples to create pairs
    data_pool_size = args.num_preference_samples * 2
    
    all_responses = []
    # Generate in chunks to prevent memory issues
    # Note: This process might take time if generating a large amount of data.
    for _ in tqdm(range(data_pool_size // args.batch_size), desc="Generating samples"):
        with torch.no_grad():
            response_tensors = ref_model.generate(
                torch.cat([prompt_tensors] * args.batch_size), **generation_kwargs
            )
        all_responses.extend(tokenizer.batch_decode(response_tensors[:, prompt_tensors.shape[1]:], skip_special_tokens=True))

    # After generation
    N = len(all_responses)                 # Actual number generated
    if N % 2 == 1:                         # If odd, remove the last one to match pairs
        all_responses = all_responses[:-1]
    N = len(all_responses)                 # Update to even number
    
    prompts = [prompt_text] * N            # Match actual N, not data_pool_size
    scores  = batched_reward_scores(prompts, all_responses, reward_model, tokenizer,
                                    batch_size=64, max_length=512)

    del reward_model
    torch.cuda.empty_cache()
    
    preference_data = {'prompt': [], 'chosen': [], 'rejected': []}
    for i in range(0, N, 2):               # Use N instead of data_pool_size
        if scores[i] > scores[i+1]:
            chosen, rejected = all_responses[i], all_responses[i+1]
        else:
            chosen, rejected = all_responses[i+1], all_responses[i]
        preference_data['prompt'].append(prompt_text)
        preference_data['chosen'].append(chosen)
        preference_data['rejected'].append(rejected)

    offline_dataset_raw = Dataset.from_dict(preference_data)
    offline_dataset = offline_dataset_raw.map(tokenize_row, fn_kwargs={"tokenizer": tokenizer}, num_proc=4)
    
    print("--- Offline dataset generation complete ---")
    
    # --- 3. Initialize DPOTrainer ---
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    dpo_config = DPOConfig(
        beta=args.beta,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="wandb"
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=offline_dataset,
        max_prompt_length=64,
        max_length=args.max_gen_length + 64,
        callbacks=[SampleLoggingCallback(tokenizer, log_interval=50)]
    )
    
    # --- 4. Start Training ---
    print("--- Starting Offline DPO Training ---")
    dpo_trainer.train()
    
    # --- 5. Save Final Model ---
    if dpo_trainer.accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        dpo_trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"--- DPO Training Finished. Saving final model to {final_dir} ---")

if __name__ == "__main__":
    main()