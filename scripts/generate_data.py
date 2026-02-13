import sys
import os
import json
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import load_model_config
from src.models.model_loader import load_model

def main():
    # 1. Load Configs
    print("Loading configurations...")
    teacher_cfg = load_model_config("config/model/teacher.yaml")
    
    # 2. Load Teacher Model (Inference Mode)
    print(f"Loading Teacher: {teacher_cfg.model.name}...")
    model, tokenizer = load_model(teacher_cfg, inference_mode=True)

    # 3. Load Dataset (Use TRAIN split to avoid benchmark contamination)
    print("Loading MBPP dataset (train split)...")
    dataset = load_dataset("google-research-datasets/mbpp", split="train")

    # 4. Define Prompt Template for CoT
    # We force the model to explain itself before writing code.
    prompt_template = """You are an expert Python programmer.

Problem: {prompt}

Instructions:
1. Analyze the problem step-by-step.
2. Write the Python code to solve it.
3. Ensure the code is enclosed in ```python ... ``` blocks.

### Reasoning:
"""

    output_file = "data/raw/teacher_generated.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Generating responses for {len(dataset)} examples...")
    
    with open(output_file, 'w') as f:
        for item in tqdm(dataset):
            task_id = item['task_id']
            problem_text = item['text']
            tests = item['test_list'] # MBPP has a list of assert statements

            # Format input
            input_text = prompt_template.format(prompt=problem_text)
            
            # Tokenize
            inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
            input_len = inputs.input_ids.shape[1]
            
            # Generate
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                use_cache=True,
                temperature=0.7
            )
            
            # Decode
            # Slice outputs to exclude the input tokens so we don't repeat the prompt
            generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            # Save raw output
            entry = {
                "task_id": task_id,
                "prompt": problem_text,
                "generated_text": generated_text,
                "tests": tests
            }
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()