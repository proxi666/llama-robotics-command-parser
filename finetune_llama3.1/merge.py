# merge_unsloth_fixed.py
from unsloth import FastLanguageModel
import os

# Create output directory first
output_dir = "./llama-3.1-8b-merged"
os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./llama-robotics-finetuned",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Merging and saving...")
model.save_pretrained_merged(
    output_dir,
    tokenizer,
    save_method="merged_16bit"
)

print(f"✅ Model merged and saved to {output_dir}")
