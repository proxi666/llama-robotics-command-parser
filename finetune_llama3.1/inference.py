# inference.py
import torch
from unsloth import FastLanguageModel
import os

def load_model():
    """Load the fine-tuned model properly"""
    model_path = "./llama-robotics-finetuned"
    
    # Check if the fine-tuned model exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return None, None
    
    print(f"Loading from: {model_path}")
    
    
    # Load the fine-tuned model (this should include the merged LoRA weights)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def get_response(model, tokenizer, user_input):
    """Format prompt properly and get response"""
    # Use the same format as training (Alpaca format)
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Convert the following navigation instruction into a structured JSON format with specific actions

### Input:
{user_input}

### Response:
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,  # Very low temperature for consistent output
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Get only the response part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in full_response:
        response = full_response.split("### Response:\n")[1].strip()
    else:
        response = full_response.strip()
    
    return response

# Load model
print("Loading fine-tuned model...")
model, tokenizer = load_model()

if model is None:
    print("Failed to load model!")
    exit(1)

print("Model loaded! Testing with training format...\n")

# Test with the exact same prompt that worked during training
test_prompt = "Go to Room 1 and Pick Red Cube and Go to Room 2 and Place It"
print(f"Test prompt: {test_prompt}")
response = get_response(model, tokenizer, test_prompt)
print(f"Response: {response}\n")

if __name__ == "__main__":
    while True:
        user_input = input("Enter navigation instruction (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        response = get_response(model, tokenizer, user_input)
        print(f"Response: {response}\n")
