import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class LoRAConfig:
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_text_encoder_r: int = 8
    lora_text_encoder_alpha: int = 32
    lora_text_encoder_dropout: float = 0.0
    lora_text_encoder_bias: str = "none"


@dataclass
class InferenceConfig:
    prompt: str = "A photo of sks person"
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    height: int = 512
    width: int = 512
    generator_seed: int = None


@dataclass
class TrainingConfig:
    # Model
    pretrained_model_name_or_path: str = None
    revision: str = None
    tokenizer_name: str = None
    
    # Data
    instance_data_dir: str = None
    class_data_dir: str = None
    instance_prompt: str = None
    class_prompt: str = None
    
    # Prior preservation
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    
    # Validation
    validation_prompt: str = None
    num_validation_images: int = 4
    validation_steps: int = 100
    
    # Output
    output_dir: str = "dreambooth-lora-model"
    seed: int = None
    resolution: int = 512
    center_crop: bool = False
    
    # Text encoder
    train_text_encoder: bool = False
    
    # Training
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: int = None
    checkpointing_steps: int = 500
    resume_from_checkpoint: str = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_dataloader_workers: int = 1
    no_tracemalloc: bool = False
    
    # Optimization
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Hub
    push_to_hub: bool = False
    hub_token: str = None
    hub_model_id: str = None
    
    # Logging
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    wandb_key: str = None
    wandb_project_name: str = None
    
    # Hardware
    allow_tf32: bool = False
    mixed_precision: str = None
    prior_generation_precision: str = None
    local_rank: int = -1
    enable_xformers_memory_efficient_attention: bool = False
    
    # LoRA and Inference configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class Config:
    """Simple configuration class that allows attribute access to dict items."""
    def __init__(self, config_dict):
        # Store the original dict
        object.__setattr__(self, '_config', config_dict)
        
        # Create lora sub-config
        lora_config = {}
        for key, value in config_dict.items():
            if key.startswith('lora_') or key == 'use_lora':
                lora_config[key] = value
        object.__setattr__(self, 'lora', Config._create_simple_config(lora_config))
        
        # Create inference sub-config
        if 'inference' in config_dict:
            object.__setattr__(self, 'inference', Config._create_simple_config(config_dict['inference']))
        else:
            # Default inference config
            default_inference = {
                'prompts': ['A photo of sks person'],
                'negative_prompt': '',
                'num_inference_steps': 50,
                'guidance_scale': 7.5,
                'num_images_per_prompt': 1,
                'height': 512,
                'width': 512,
                'generator_seed': None,
                'output_dir': 'generated_images',
                'compare_with_base_model': False
            }
            object.__setattr__(self, 'inference', Config._create_simple_config(default_inference))
        
        # Create nullbooth sub-config
        if 'nullbooth' in config_dict:
            object.__setattr__(self, 'nullbooth', Config._create_simple_config(config_dict['nullbooth']))
        else:
            # Default nullbooth config
            default_nullbooth = {
                'enable': False,
                'original_knowledge_prompts': './dataset/prompts.txt',
                'cov_matrices_output_dir': './cov_matrices',
                'visual_attention_map': False,
                'num_denoising_steps': 50,
                'nullspace_threshold': 2e-2,
                'collect_features': {
                    'q_features': True,
                    'k_features': True,
                    'v_features': True,
                    'out_features': True
                },
                'cross_attention_layers': 'all'
            }
            object.__setattr__(self, 'nullbooth', Config._create_simple_config(default_nullbooth))
    
    @staticmethod
    def _create_simple_config(config_dict):
        """Create a simple config object that allows attribute access."""
        class SimpleConfig:
            def __init__(self, d):
                for key, value in d.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        setattr(self, key, Config._create_simple_config(value))
                    else:
                        setattr(self, key, value)
            def __getattr__(self, key):
                return None
        return SimpleConfig(config_dict)
    
    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        else:
            return None
    
    def __setattr__(self, key, value):
        if key.startswith('_') or key in ['lora', 'inference', 'nullbooth']:
            object.__setattr__(self, key, value)
        else:
            self._config[key] = value


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert string scientific notation to float for specific keys
    float_keys = [
        'learning_rate', 'adam_weight_decay', 'adam_epsilon', 'prior_loss_weight',
        'adam_beta1', 'adam_beta2', 'max_grad_norm', 'lr_power', 'lora_dropout',
        'lora_text_encoder_dropout'
    ]
    
    for key in float_keys:
        if key in config_dict and isinstance(config_dict[key], str):
            try:
                config_dict[key] = float(config_dict[key])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails
    
    return Config(config_dict)


def save_config(config: Dict[Any, Any], save_path: str):
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)