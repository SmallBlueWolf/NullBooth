import sys
from pathlib import Path
from accelerate.utils import set_seed

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import (
    load_config,
    setup_logging,
    setup_accelerator,
    validate_config,
    load_tokenizer,
    load_models,
    setup_lora,
    setup_model_for_training,
    get_trainable_parameters,
    setup_optimizer,
    create_datasets_and_dataloaders,
    setup_training_params,
    setup_lr_scheduler,
    setup_hub_and_output_dir,
    enable_optimizations,
    generate_class_images,
    training_loop,
    save_model
)
from src.logger import log_script_execution


def main():
    """Main training function."""
    # Setup logging
    with log_script_execution("train"):
        # Load configuration
        config_path = "configs/config.yaml"
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Validate configuration
        validate_config(config)
        
        # Setup accelerator
        accelerator = setup_accelerator(config)
        
        # Setup logging
        setup_logging(accelerator)
        
        # Set seed if specified
        if config.seed is not None:
            set_seed(config.seed)
        
        # Generate class images if needed
        generate_class_images(config, accelerator)
        
        # Setup hub and output directory
        api, repo_id = setup_hub_and_output_dir(config, accelerator)
        
        # Load tokenizer
        tokenizer = load_tokenizer(config)
        
        # Load models
        noise_scheduler, text_encoder, vae, unet = load_models(config)
        
        # Setup LoRA if enabled
        if config.lora.use_lora:
            unet, text_encoder = setup_lora(unet, text_encoder, config)
        
        # Setup models for training
        unet, text_encoder, vae = setup_model_for_training(unet, text_encoder, vae, config)
        
        # Enable optimizations
        enable_optimizations(unet, text_encoder, config, accelerator)
        
        # Create datasets and dataloaders
        train_dataset, train_dataloader = create_datasets_and_dataloaders(config, tokenizer)
        
        # Setup training parameters
        overrode_max_train_steps, num_update_steps_per_epoch = setup_training_params(
            config, train_dataloader, accelerator
        )
        
        # Setup optimizer
        params_to_optimize = get_trainable_parameters(unet, text_encoder, config.train_text_encoder)
        optimizer = setup_optimizer(params_to_optimize, config)
        
        # Setup learning rate scheduler
        lr_scheduler = setup_lr_scheduler(config, optimizer)

        # Prepare everything with accelerator
        if config.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        # Move models to device with appropriate dtype
        weight_dtype = accelerator.mixed_precision
        if weight_dtype == "fp16":
            weight_dtype = "float16"
        elif weight_dtype == "bf16":
            weight_dtype = "bfloat16"
        else:
            weight_dtype = "float32"
        
        # Move vae and text_encoder to device and cast to weight_dtype  
        import torch
        if weight_dtype == "float16":
            dtype = torch.float16
        elif weight_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        vae.to(accelerator.device, dtype=dtype)
        if not config.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=dtype)
        
        # Recalculate training steps after dataloader preparation
        if overrode_max_train_steps:
            import math
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
            config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
            config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
        
        # Run training loop
        training_loop(
            config,
            accelerator,
            unet,
            text_encoder,
            vae,
            noise_scheduler,
            optimizer,
            lr_scheduler,
            train_dataloader,
            train_dataset,
        )
        
        # Save model
        save_model(config, accelerator, unet, text_encoder, api, repo_id)
        
        # End training
        accelerator.end_training()
        print("Training completed successfully!")


if __name__ == "__main__":
    main()