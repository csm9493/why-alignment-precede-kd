import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import uuid
import os
import argparse
from huggingface_hub import login

import numpy as np
from transformers import set_seed

def seed_everything(seed):
    """
    Fixes random seeds for reproducibility across all libraries.
    """
    # Basic seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA seed setting
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # Use deterministic algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Hugging Face Transformers seed setting
    set_seed(seed)

# =======================
# Argument Parser
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=int, required=True, help="e.g., 135, 360, 1700")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=600000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--kd_iteration", type=int, default=0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--load_model_name", type=str, default=None)
parser.add_argument("--data_type", type=str, default='train')
args = parser.parse_args()

# =======================
# Configuration
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.seed != 0:
    seed_everything(args.seed)

if '135M' not in args.load_model_name and '360M' not in args.load_model_name:

    if 'kd' not in args.load_model_name:
    
        model_ckpt_dir = f"results/{args.load_model_name}/checkpoint-{args.iteration}/"

    else:

        if args.kd_iteration == 0:

            model_ckpt_dir = f"results/{args.load_model_name}/checkpoint-epoch-1/"

        else:

            model_ckpt_dir = f"results/{args.load_model_name}/checkpoint-{args.kd_iteration}/"
        
else:
    model_ckpt_dir = f"pretrained_models/{args.load_model_name}/seed{args.seed}"
    
tokenizer_hf_path = f"HuggingFaceTB/SmolLM2-{args.model_size}M"

if args.data_type == 'validation':
    output_dir = f"./generated_data/{args.load_model_name}_validation"
    args.temperature = 1.0
    
else:

    if args.temperature == 0.8:
        output_dir = f"./generated_data/{args.load_model_name}"
    else:
        output_dir = f"./generated_data/{args.load_model_name}_kd_temp_{args.temperature}"
    
os.makedirs(output_dir, exist_ok=True)

# =======================
# Load model & tokenizer
# =======================

# Login with token
login('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')

model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_path)

# Set pad_token to eos_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("BOS:", tokenizer.bos_token, "| PAD:", tokenizer.pad_token, "| EOS:", tokenizer.eos_token)

# =======================
# Generation settings
# =======================
temperature = args.temperature
top_p = 1.0
top_k = model.config.vocab_size
num_return_sequences = 1

inputs = tokenizer(["The"] * args.batch_size, return_tensors="pt").to(device)

# =======================
# Generation loop
# =======================
generated_data = []
for i in range(0, args.num_samples, args.batch_size):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_length=512,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j in range(args.batch_size):
        responses = output[j * num_return_sequences : (j + 1) * num_return_sequences]
        decoded_responses = [tokenizer.decode(res, skip_special_tokens=True) for res in responses]
        for response in decoded_responses:
            generated_data.append({"prompt": "The", "response": response})

    # Save every 100,000 samples
    if len(generated_data) >= 100_000:
        random_id = uuid.uuid4().hex
        output_file = os.path.join(output_dir, f"generated_data_part_{random_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)
        generated_data = []

    print(f"Generated: {i + args.batch_size} / {args.num_samples}")

# Final save
if generated_data:
    random_id = uuid.uuid4().hex
    output_file = os.path.join(output_dir, f"generated_data_part_{random_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

print("Data generation complete.")