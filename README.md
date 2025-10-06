# Fine-Tuning Llama 3.1 for Robotics Path Planning 

## Overview

This project demonstrates how to fine-tune the Meta Llama 3.1 8B model to act as a specialized parser for robotics commands. The goal is to convert high-level, natural-language instructions (e.g., "Go to the kitchen, grab the apple, and bring it to the living room") into a structured JSON format that a robot or other automated agent can understand and execute.

This project uses the `unsloth` library for highly efficient, memory-saving fine-tuning, allowing the model to be trained on consumer-grade GPUs.

## Pre-trained Model

The fine-tuned model adapter is available on the Hugging Face Hub at the following URL:
[https://huggingface.co/Proxiii/Llama-3.1-8b-path-planner-parser](https://huggingface.co/Proxiii/Llama-3.1-8b-path-planner-parser)

You can use the `inference.py` script to load and run this model directly from the Hub by changing the `model_path` to `"Proxiii/Llama-3.1-8b-path-planner-parser"`.

## How it Works

The project follows a clear, three-step pipeline:

1.  **Data Preprocessing:** A dataset of natural language commands and their corresponding JSON representations is prepared. The script `data_preprocessing.py` handles this conversion from an Excel file.
2.  **Fine-Tuning:** The `finetune_llama.py` script uses the prepared dataset to fine-tune the Llama 3.1 8B model. It employs a technique called QLoRA (Quantized Low-Rank Adaptation), which is a form of Parameter-Efficient Fine-Tuning (PEFT). This means we only train a small "adapter" layer on top of the frozen base model, drastically reducing computational and memory requirements.
3.  **Inference:** Once the model is fine-tuned, the `inference.py` script loads the model and its new adapter to perform the command parsing task. It provides an interactive prompt for you to test the model with new commands.

## Directory Structure

Here is a breakdown of the key files in this project:

```
/
└── finetune_llama3.1/
    ├── data_preprocessing.py   # Converts your Excel data into a training-ready JSON file.
    ├── finetune_llama.py       # The main script to run the model fine-tuning process.
    ├── merge.py                # Optional script to merge the trained adapter into the base model.
    ├── inference.py            # Script to run the fine-tuned model and test it with new commands.
    ├── prompts_output.xlsx     # Your source data file containing prompts and desired outputs.
    └── training_data.json      # The auto-generated dataset used for training.
```

## Setup and Installation

### Prerequisites

*   An NVIDIA GPU with at least 8GB of VRAM is recommended.
*   CUDA Toolkit installed.

### Installation

1.  **Clone the repository or download the project files.**

2.  **Create a Python virtual environment.** This is recommended to keep dependencies isolated.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries.** This project relies on `unsloth`, which handles the installation of many necessary packages like `torch`, `transformers`, and `trl`.
    ```bash
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install "unsloth[conda-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install pandas openpyxl
    ```
    *Note: The two `unsloth` commands cover different dependencies that might be needed. It's safe to run both.*

## Step-by-Step Usage

Follow these steps to run the complete pipeline from data preparation to inference. All commands should be run from the `finetune_llama3.1` directory.

```bash
cd finetune_llama3.1
```

### Step 1: Prepare Your Data

1.  Open `prompts_output.xlsx`.
2.  Fill the spreadsheet with your own data.
    *   The **`prompts`** column should contain the natural language instructions you want to parse.
    *   The **`output`** column should contain the perfect, structured JSON you want the model to generate for each corresponding prompt.
3.  Save the file.

### Step 2: Run the Data Preprocessing Script

This script will convert your Excel data into the `training_data.json` file used for training.

```bash
python data_preprocessing.py
```

### Step 3: Run the Fine-Tuning Process

This is the main event. The script will load the Llama 3.1 model, apply the LoRA adapter, and train it on your data.

```bash
python finetune_llama.py
```

This process will save the trained adapter to a new directory named `llama-robotics-finetuned`.

### Step 4: Test the Model with Inference

Now you can test your newly fine-tuned model. This script provides an interactive prompt.

```bash
python inference.py
```

Once the model is loaded, you can type in a command and press Enter to see the model's JSON output.

```
Enter navigation instruction (or 'quit' to exit): Go to Room 1 and Pick Red Cube
```

### Example Output

Here is a sample output from the model when given the test prompt:

```json
[
  {
    "instruction": "Go to Room 1",
    "action": "nav",
    "target": "Room 1",
    "location": "Room 1"
  },
  {
    "instruction": "Pick Red Cube",
    "action": "pick",
    "target": "Red Cube",
    "location": "Room 1"
  },
  {
    "instruction": "Go to Room 2",
    "action": "nav",
    "target": "Room 2",
    "location": "Room 2"
  },
  {
    "instruction": "Place It",
    "action": "place",
    "target": "Red Cube",
    "location": "Room 2"
  }
]
```

### (Optional) Step 5: Merge the Adapter

If you want to create a single, standalone model for easier deployment, you can merge the LoRA adapter with the base model.

```bash
python merge.py
```

This will create a new directory, `llama-3.1-8b-merged`, containing the full, merged model.

## Customization

You can easily customize this project for your own needs.

### Using a Different Model

To use a different model, change the `MODEL_NAME` variable in `finetune_llama.py`. You can find many compatible models on the [Unsloth Hugging Face page](https://huggingface.co/unsloth).

### Adjusting Training Parameters

In `finetune_llama.py`, you can modify the `TrainingArguments` to control the training process:

*   `max_steps`: The total number of training steps. Set to `-1` to train for a specific number of epochs (defined by `num_train_epochs`).
*   `per_device_train_batch_size`: How many training examples to process at once. Increase this if you have more VRAM.
*   `learning_rate`: How quickly the model learns. `2e-4` is a good starting point for LoRA.

### Modifying the LoRA Configuration

In the `load_model_and_tokenizer` function within `finetune_llama.py`, you can change the LoRA parameters:

*   `r`: The "rank" or dimension of the LoRA adapter. A higher rank means more trainable parameters and potentially better performance, but also more memory usage. Common values are 8, 16, 32, 64.

