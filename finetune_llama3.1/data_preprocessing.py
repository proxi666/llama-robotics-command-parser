# data_preprocessing.py
import pandas as pd
import json

def prepare_dataset():
    # Load the Excel file
    df = pd.read_excel('prompts_output.xlsx')
    
    # Create training data in the format expected by the model
    training_data = []
    
    for _, row in df.iterrows():
        prompt = row['prompts']
        output = row['output']
        
        # Format as instruction-following dataset
        entry = {
            "instruction": "Convert the following navigation instruction into a structured JSON format with specific actions:",
            "input": prompt,
            "output": output
        }
        training_data.append(entry)
    
    # Save as JSON
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    return training_data

# Run preprocessing
prepare_dataset()
