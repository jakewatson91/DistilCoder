from unsloth import FastLanguageModel
import torch
from typing import Tuple, Any

def load_model(model_config: Any, inference_mode: bool = True) -> Tuple[Any, Any]:
    """
    Loads a model using Unsloth.
    Args:
        model_config: ModelConfig Pydantic object
        inference_mode: If True, optimizes for inference (faster). If False, prepares for training (LoRA).
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model.name,
        max_seq_length=model_config.model.max_seq_length,
        dtype=None, # Auto-detect
        load_in_4bit=model_config.quant.use_4bit,
    )

    if inference_mode:
        FastLanguageModel.for_inference(model)
    else:
        # Prepare for LoRA training
        # Use lora config if present, otherwise defaults
        lora = model_config.lora
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora.r if lora else 16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora.alpha if lora else 16,
            lora_dropout=lora.dropout if lora else 0,
            bias="none",
        )
    
    return model, tokenizer