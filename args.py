from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2")
    path_to_save_model: str = field(default="my_privacy_gpt2_model")
    sequence_len: int = field(default=512)

@dataclass
class DataArguments:
    dataset_name: str = field(default="med-summ")
    ds_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    path_to_dataset: str = field(default="/data/train.csv")
    path_to_test_dataset: Optional[str] = field(default=None)
    control_field: str = field(default="label")
    text_field: str = field(default="text")
    label_field: str = field(default="text")
    prompt_begin: str = field(default="")
    prompt_end: str = field(default="")
    def __post_init__(self):
        if self.path_to_test_dataset == "None":
            self.path_to_test_dataset = None

@dataclass
class TrainArguments:
    epochs: int = field(default=1)
    lr: float = field(default=5e-5)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: float = field(default=128)  # Gradient Accumulation Steps
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    project_name: str = field(default="dp-fact")
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if self.lr <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient Accumulation Steps must be greater than 0.")

@dataclass
class PrivacyArguments:
    target_epsilon: float = field(default=None)
    max_grad_norm: float = field(default=0.5)
    target_delta: float = field(default=1e-5)
    clipping: str = field(default="per_layer")  # Options: 'flat', 'per_layer'
    noise_multiplier: float = field(default=None)
    clipbound_learning_rate: float = field(default=0.2)
    target_unclipped_quantile: float = field(default=0.5)
    min_clipbound: float = field(default=0.1)  # Minimum clip bound
    max_clipbound: float = field(default=1.0)  # Maximum clip bound
    unclipped_num_std: float = field(default=0.1)
    pld: bool = field(default=False)  # Use PLD accountant