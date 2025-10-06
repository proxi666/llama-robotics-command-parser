import torch
import os
import pandas as pd
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = "./llama-robotics-finetuned"
DATASET_PATH = "training_data.json"
MAX_SEQ_LENGTH = 2048  # Choose any! We auto support RoPE Scaling internally!
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.

def format_instruction(example):
    """Format the data for instruction following with Alpaca format"""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def load_model_and_tokenizer():
    """Load model and tokenizer using Unsloth"""
    print("Loading model and tokenizer with Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    # Add LoRA adapters using Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )
    
    # Print model info
    FastLanguageModel.for_training(model)  # Enable native 2x faster training
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Prepare dataset for training"""
    # Load your data
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    
    # Clean the data - remove entries with NaN/empty values
    cleaned_data = []
    for item in data:
        # Skip if any field is NaN, None, or empty
        if (pd.isna(item.get('instruction')) or pd.isna(item.get('input')) or pd.isna(item.get('output')) or
            item.get('instruction') == '' or item.get('input') == '' or item.get('output') == ''):
            continue
            
        cleaned_item = {
            'instruction': str(item['instruction']).strip(),
            'input': str(item['input']).strip(),
            'output': str(item['output']).strip()
        }
        cleaned_data.append(cleaned_item)
    
    print(f"After cleaning: {len(cleaned_data)} training examples")
    
    if len(cleaned_data) == 0:
        raise ValueError("No valid training examples found after cleaning!")
    
    # Convert to DataFrame
    df = pd.DataFrame(cleaned_data)
    
    print(f"Sample data:")
    print(f"Instruction: {df.iloc[0]['instruction'][:100]}...")
    print(f"Input: {df.iloc[0]['input'][:100]}...")
    print(f"Output: {df.iloc[0]['output'][:100]}...")
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    
    # Format the text for each example using Alpaca format
    def format_examples(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = format_instruction({
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i]
            })
            texts.append(text)
        return {'text': texts}
    
    # Apply formatting
    dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset

def main():
    """Main training function"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your PyTorch installation.")
    
    print(f"Using device: cuda")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    print(f"Training dataset size: {len(dataset['train'])}")
    print(f"Validation dataset size: {len(dataset['test'])}")
    
    # Training arguments optimized for Unsloth
    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # Unsloth can handle larger batch sizes
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Set to -1 for full training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=30,
        eval_strategy="steps",
        eval_steps=30,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )
    
    # Initialize SFTTrainer with Unsloth optimizations
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Start training
    print("Starting training with Unsloth optimizations...")
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    # Save model
    model.save_pretrained(OUTPUT_DIR)  # Local saving
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")
    
    # Optional: Save to 16bit for inference
    # model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    main()
