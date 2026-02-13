import argparse
import os
import json
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

def extract_code(text):
    """Extracts code from markdown blocks (Same as filter_data.py)."""
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return text # Fallback: return raw text if no block found

def format_prompt(instruction):
    """Wraps the benchmark prompt in the training format."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def main():
    parser = argparse.ArgumentParser(description="Evaluate model with vLLM")
    parser.add_argument("--model_path", type=str, default="results/final_student_model/merged")
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize vLLM
    print(f"Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path, tensor_parallel_size=1, trust_remote_code=True)
    
    # Greedy decoding for benchmarks (deterministic)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    # --- Benchmark 1: HumanEval ---
    print("\n--- Running HumanEval ---")
    ds = load_dataset("openai_humaneval", split="test")
    
    prompts = [format_prompt(f"Complete the following Python function:\n```python\n{item['prompt']}\n```") for item in ds]
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, item in enumerate(ds):
        generated_text = outputs[i].outputs[0].text
        code = extract_code(generated_text)
        # HumanEval expects the function body combined with the signature if we extracted just the body
        # But usually, we save the completion to be evaluated by a harness.
        results.append({"task_id": item["task_id"], "completion": code})
    
    with open(os.path.join(args.output_dir, "humaneval_output.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    # --- Benchmark 2: BigCodeBench ---
    print("\n--- Running BigCodeBench ---")
    try:
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2_test") # Use latest version
    except:
        ds = load_dataset("bigcode/bigcodebench", split="test")
        
    prompts = [format_prompt(item["complete_prompt"]) for item in ds]
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, item in enumerate(ds):
        generated_text = outputs[i].outputs[0].text
        code = extract_code(generated_text)
        results.append({"task_id": item["task_id"], "completion": code})
        
    with open(os.path.join(args.output_dir, "bigcodebench_output.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    # --- Benchmark 3: LiveCodeBench (2024.07 - 2024.11) ---
    print("\n--- Running LiveCodeBench (Hardest) ---")
    ds = load_dataset("livecodebench/lcb_v1", split="test")
    
    # Filter by date
    def filter_date(x):
        return "2024-07-01" <= x["date"] <= "2024-11-30"
    
    ds_filtered = ds.filter(filter_date)
    print(f"Filtered LiveCodeBench to {len(ds_filtered)} examples (2024.07-2024.11).")

    if len(ds_filtered) > 0:
        # LCB prompts are usually plain text descriptions
        prompts = [format_prompt(item["question_content"]) for item in ds_filtered]
        outputs = llm.generate(prompts, sampling_params)
        
        results = []
        for i, item in enumerate(ds_filtered):
            generated_text = outputs[i].outputs[0].text
            code = extract_code(generated_text)
            results.append({
                "question_id": item["question_id"], 
                "completion": code,
                "date": item["date"]
            })
            
        with open(os.path.join(args.output_dir, "livecodebench_output.jsonl"), "w") as f:
            for r in results: f.write(json.dumps(r) + "\n")

    print(f"\nDone! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()