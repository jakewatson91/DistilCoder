import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

# --- Pydantic Models ---

class ModelParams(BaseModel):
    name: str
    use_fast_tokenizer: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    use_unsloth: bool = True
    max_seq_length: int = 2048

class QuantParams(BaseModel):
    enabled: bool = True
    use_4bit: bool = True
    use_8bit: bool = False

    @model_validator(mode='after')
    def check_exclusive_quantization(self):
        if self.use_4bit and self.use_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization simultaneously.")
        return self

class HFParams(BaseModel):
    token_env: str = "HF_TOKEN"
    hf_home: str = "~/.cache/huggingface"

class LoraParams(BaseModel):
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class ModelConfig(BaseModel):
    model: ModelParams
    quant: QuantParams = Field(default_factory=QuantParams)
    huggingface: HFParams = Field(default_factory=HFParams)
    # Lora might be injected from training config
    lora: Optional[LoraParams] = None

class TrainingParams(BaseModel):
    max_length: int = 2048
    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.01
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 50
    dataloader_num_workers: int = 2
    dataset_num_proc: int = 2

    @model_validator(mode='after')
    def check_precision_conflict(self):
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16 mixed precision simultaneously.")
        return self

class TrainingConfig(BaseModel):
    training: TrainingParams
    lora: LoraParams = Field(default_factory=LoraParams)

def load_model_config(yaml_path: str) -> ModelConfig:
    with open(yaml_path, "r") as f:
        return ModelConfig(**yaml.safe_load(f))

def load_training_config(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, "r") as f:
        return TrainingConfig(**yaml.safe_load(f))