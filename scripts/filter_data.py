import json
import re
import os
import signal
import argparse
import multiprocessing
from tqdm import tqdm

# Timeout handler to prevent infinite loops in generated code
class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException

def extract_code(text):
    """Extracts code from markdown blocks."""
    # More robust regex: handles 'python', 'py', or no label, and whitespace
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1] # Return the last block (usually the final solution)
    return None

def run_tests(code, tests, timeout=2):
    """
    Executes code with tests. 
    WARNING: executing arbitrary code is dangerous. Run in a container in prod.
    """
    # Inject common imports so code doesn't fail on missing standard libs
    header = "import math\nimport re\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\nimport random\n"
    
    # Combine code and tests
    full_script = header + code + "\n" + "\n".join(tests)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Create a restricted scope (not a full sandbox, but better than nothing)
        exec_globals = {}
        exec(full_script, exec_globals)
        signal.alarm(0)
        return True
    except TimeoutException:
        return False
    except Exception as e:
        signal.alarm(0)
        return False

def process_line(line):
    """Worker function for multiprocessing."""
    try:
        data = json.loads(line)
        code = extract_code(data['generated_text'])
        if not code:
            return None

        if run_tests(code, data['tests']):
            return {
                "instruction": data['prompt'],
                "output": data['generated_text']
            }
    except Exception:
        return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Filter generated code by running tests.")
    parser.add_argument("--input_file", type=str, default="data/raw/teacher_generated.jsonl")
    parser.add_argument("--output_file", type=str, default="data/processed/filtered_data.jsonl")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if not os.path.exists(args.input_file):
        print(f"File {args.input_file} not found.")
        return

    print(f"Filtering code from {args.input_file} using {args.workers} workers...")
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    passed_examples = []
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        # Use imap to process in parallel and show progress
        for result in tqdm(pool.imap(process_line, lines), total=len(lines)):
            if result:
                passed_examples.append(result)

    print(f"Saving {len(passed_examples)} passed examples to {args.output_file}...")
    with open(args.output_file, 'w') as fout:
        for entry in passed_examples:
            fout.write(json.dumps(entry) + "\n")

    print(f"Filtering complete. {len(passed_examples)}/{len(lines)} examples passed.")

if __name__ == "__main__":
    main()