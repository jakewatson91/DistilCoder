import sys
import os
import argparse
from datasets import load_dataset, concatenate_datasets

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import load_model_config, load_training_config
from src.models.model_loader import load_model
from src.training.trainer import train, merge_and_save_hf

def formatting_prompts_func(examples):
    """Formats data for SFT. Maps 'instruction' and 'output' to chat format."""
    output_texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        # Simple Alpaca-style format or ChatML
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        output_texts.append(text)
    return output_texts

def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--model_config", type=str, default="config/model/student.yaml", help="Path to model config")
    parser.add_argument("--train_config", type=str, default="config/training/finetune.yaml", help="Path to training config")
    args = parser.parse_args()

    # 1. Load Configs
    print("Loading configurations...")
    student_cfg = load_model_config(args.model_config)
    full_train_cfg = load_training_config(args.train_config)
    train_cfg = full_train_cfg.training
    
    # Inject LoRA config from training settings into student model config
    student_cfg.lora = full_train_cfg.lora
    
    # 2. Load Student Model (Training Mode)
    print(f"Loading Student: {student_cfg.model.name}...")
    model, tokenizer = load_model(student_cfg, inference_mode=False)

    # 3. Load Datasets
    # A. Load External Dataset (Nvidia)
    dataset_name = "nvidia/OpenCodeInstruct"
    print(f"Loading external dataset: {dataset_name}...")
    nvidia_dataset = load_dataset(dataset_name, split="train")

    # Map columns if necessary (e.g. problem -> instruction)
    if "problem" in nvidia_dataset.column_names and "solution" in nvidia_dataset.column_names:
        nvidia_dataset = nvidia_dataset.rename_columns({"problem": "instruction", "solution": "output"})

    # B. Load Synthetic/Filtered Data (Distillation)
    filtered_file = "data/processed/filtered_data.jsonl"
    if os.path.exists(filtered_file):
        print(f"Loading synthetic data: {filtered_file}...")
        synthetic_dataset = load_dataset("json", data_files=filtered_file, split="train")
        print(f"Merging {len(synthetic_dataset)} synthetic examples with {len(nvidia_dataset)} external examples.")
        dataset = concatenate_datasets([nvidia_dataset, synthetic_dataset])
    else:
        print("No synthetic data found. Training on external dataset only.")
        dataset = nvidia_dataset

    # Split into train/val (5% for validation)
    print("Splitting dataset into train/validation...")
    dataset_split = dataset.train_test_split(test_size=0.05, seed=train_cfg.seed)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    print(f"Training on {len(train_dataset)} examples, Validating on {len(eval_dataset)} examples.")

    # 4. Run Training
    print("Starting Distillation (Fine-tuning)...")
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="results/checkpoints",
        logging_dir="results/logs",
        formatting_func=formatting_prompts_func,
        max_length=train_cfg.max_length,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_accum=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        logging_steps=train_cfg.logging_steps,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        max_steps=train_cfg.max_steps,
        grad_checkp=train_cfg.gradient_checkpointing,
        seed=train_cfg.seed,
        dataset_num_proc=train_cfg.dataset_num_proc,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
    )
    
    # Save final model
    print("Saving and Merging Final Model...")
    final_dir = "results/final_student_model"
    
    # Merge to 16bit for vLLM/Inference (Best practice from finetuning.py)
    merge_and_save_hf(model, tokenizer, os.path.join(final_dir, "merged"), method="merged_16bit")
    
    # Also save the adapter separately just in case
    model.save_pretrained(os.path.join(final_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(final_dir, "adapter"))

if __name__ == "__main__":
    main()