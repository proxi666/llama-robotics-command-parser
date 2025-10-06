import torch
import os
from unsloth import FastLanguageModel

# The model adapter you want to push to the hub
model_path = "./llama-robotics-finetuned"

# Your Hugging Face username and the desired repo name
hf_repo_name = "Proxiii/Llama-3.1-8b-path-planner-parser"

# Check for Hugging Face token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    # Fallback to reading from the huggingface-cli token file
    try:
        with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
            hf_token = f.read().strip()
    except FileNotFoundError:
        raise ValueError(
            "Hugging Face token not found. Please log in via `huggingface-cli login` "
            "or set the HF_TOKEN environment variable."
        )

if not hf_token:
    raise ValueError(
        "Hugging Face token is empty. Please ensure you are logged in correctly."
    )

print(f"Loading model from {model_path}...")

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print(f"Pushing model and tokenizer to Hugging Face Hub: {hf_repo_name}")

# Push the model adapter to the Hub
model.push_to_hub(hf_repo_name, token=hf_token)

# Push the tokenizer to the Hub
tokenizer.push_to_hub(hf_repo_name, token=hf_token)

print("\n✅ Successfully pushed to the Hub!")
print(f"Model available at: https://huggingface.co/{hf_repo_name}")
