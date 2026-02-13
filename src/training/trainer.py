from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
import torch
import os
from typing import Union, List, Optional, Callable, Any

def train(
    *,
    model,
    tokenizer,
    train_dataset: Union[Dataset, List[str]],
    eval_dataset: Union[Dataset, List[str]],
    output_dir: str,
    logging_dir: str,
    formatting_func: Optional[Callable] = None,
    max_length: int = 2048,
    epochs: int = 3,
    batch_size: int = 2,  # Bumped to 2 (Unsloth handles this well)
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,  
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    fp16: bool = False,
    bf16: bool = True,
    max_steps: int = -1,      
    grad_checkp: bool = True,
    seed: int = 3407,      # Added seed argument
    report_to: str = "none", # Added report_to argument
    optim: str = "adamw_8bit",
    dataset_num_proc: int = os.cpu_count(),
    dataloader_num_workers: int = 2,
    callbacks: Optional[List[Any]] = None,
):
    # 1. Convert lists to Datasets
    if isinstance(train_dataset, list):
        train_dataset = Dataset.from_dict({"text": train_dataset})
    if isinstance(eval_dataset, list):
        eval_dataset = Dataset.from_dict({"text": eval_dataset})

    # 2. Configure Training (All args moved here for modern TRL support)
    sft_config = SFTConfig(
        output_dir = output_dir,
        dataset_text_field = "text" if formatting_func is None else None,
        max_seq_length = max_length,
        dataset_num_proc = dataset_num_proc,
        packing = False, # Set to True if you want faster training on long contexts
        
        # Training Stats
        num_train_epochs = epochs,
        max_steps = max_steps,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = learning_rate,
        optim = optim,
        weight_decay = weight_decay,
        warmup_ratio = warmup_ratio,
        lr_scheduler_type = "linear",
        
        # Hardware
        fp16 = fp16,
        bf16 = bf16,
        gradient_checkpointing = grad_checkp,
        seed = seed,
        
        # Logging
        logging_dir = str(logging_dir),
        logging_steps = logging_steps,
        report_to = report_to,
        dataloader_num_workers = dataloader_num_workers,
        
        # Evaluation & Saving
        evaluation_strategy = "steps",
        eval_steps = logging_steps,
        save_strategy = "steps",
        save_steps = logging_steps * 2,
        load_best_model_at_end = True,
    )

    # 3. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        formatting_func = formatting_func,
        args = sft_config,
        callbacks = callbacks,
    )

    trainer.train()
    return trainer

def save_lora_adapter(model, tokenizer, adapter_dir: str):
    # Always save tokenizer with adapter for reproducibility
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

def merge_and_save_hf(model, tokenizer, merged_dir: str, method: str = "merged_16bit"):
    # Options: "merged_16bit", "merged_4bit", "lora"
    print(f"Merging model using method: {method}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method=method)