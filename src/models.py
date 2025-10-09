import itertools
from typing import Union

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    LCMScheduler,  # Add LCM scheduler
    LatentConsistencyModelPipeline,  # Add LCM pipeline
)
from transformers import PretrainedConfig
from peft import LoraConfig, get_peft_model

# Target modules for LoRA
UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """Import the appropriate text encoder class based on the model configuration."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_models(config):
    """Load all the required models for training (SD or LCM)."""
    # Check if we're loading an LCM model
    model_type = getattr(config, 'model_type', 'SD')

    if model_type == 'LCM':
        return load_lcm_models(config)
    else:
        return load_sd_models(config)


def load_sd_models(config):
    """Load Stable Diffusion models for training."""
    # Import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path,
        config.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    text_encoder = text_encoder_cls.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision
    )

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision
    )

    return noise_scheduler, text_encoder, vae, unet


def load_lcm_models(config):
    """Load LCM (Latent Consistency Model) models for training."""
    print(f"Loading LCM model from: {config.pretrained_model_name_or_path}")

    # Import text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path,
        config.revision
    )

    # Load LCM scheduler
    noise_scheduler = LCMScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Load components
    text_encoder = text_encoder_cls.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision
    )

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision
    )

    print(f"LCM model loaded successfully")
    print(f"  Scheduler: {noise_scheduler.__class__.__name__}")
    print(f"  Text Encoder: {text_encoder.__class__.__name__}")
    print(f"  UNet: {unet.__class__.__name__}")

    return noise_scheduler, text_encoder, vae, unet


def setup_lora(unet, text_encoder, config):
    """Setup LoRA for UNet and optionally text encoder."""
    if config.lora.use_lora:
        # Setup LoRA for UNet
        unet_lora_config = LoraConfig(
            r=config.lora.lora_r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.lora_bias,
        )
        unet = get_peft_model(unet, unet_lora_config)
        unet.print_trainable_parameters()
        print(unet)
        
        # Setup LoRA for text encoder if training
        if config.train_text_encoder:
            text_encoder_lora_config = LoraConfig(
                r=config.lora.lora_text_encoder_r,
                lora_alpha=config.lora.lora_text_encoder_alpha,
                target_modules=TEXT_ENCODER_TARGET_MODULES,
                lora_dropout=config.lora.lora_text_encoder_dropout,
                bias=config.lora.lora_text_encoder_bias,
            )
            text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
            text_encoder.print_trainable_parameters()
            print(text_encoder)
    
    return unet, text_encoder


def setup_model_for_training(unet, text_encoder, vae, config):
    """Setup models for training - freeze VAE and optionally text encoder."""
    vae.requires_grad_(False)
    if not config.train_text_encoder:
        text_encoder.requires_grad_(False)
    
    return unet, text_encoder, vae


def get_trainable_parameters(unet, text_encoder, train_text_encoder: bool):
    """Get parameters to optimize."""
    if train_text_encoder:
        return itertools.chain(unet.parameters(), text_encoder.parameters())
    else:
        return unet.parameters()


def setup_optimizer(params_to_optimize, config):
    """Setup optimizer based on configuration."""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    return optimizer