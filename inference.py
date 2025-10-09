#!/usr/bin/env python3
"""
DreamBooth Inference Script
Load and run inference with trained DreamBooth models (LoRA or full fine-tuned).
"""

import sys
import torch
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import load_config
from src.logger import log_script_execution
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, LatentConsistencyModelPipeline, LCMScheduler
from peft import PeftModel


def is_lcm_model(base_model_path: str) -> bool:
    """
    Check if the base model is an LCM model by looking for LCMScheduler.

    Args:
        base_model_path: Path to the base model directory

    Returns:
        True if it's an LCM model
    """
    model_path = Path(base_model_path)
    model_index_path = model_path / "model_index.json"

    if model_index_path.exists():
        import json
        with open(model_index_path) as f:
            model_index = json.load(f)
        # Check if it uses LCMScheduler
        scheduler_class = model_index.get("scheduler", [None, None])[1]
        return scheduler_class == "LCMScheduler" or model_index.get("_class_name") == "LatentConsistencyModelPipeline"

    return False


def is_lora_model(model_path: str) -> bool:
    """
    Check if the model is a LoRA model by looking for adapter_config.json files.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        True if it's a LoRA model, False if it's a full fine-tuned model
    """
    model_path = Path(model_path)
    
    # Check for LoRA adapter config files
    unet_adapter_config = model_path / "unet" / "adapter_config.json"
    text_encoder_adapter_config = model_path / "text_encoder" / "adapter_config.json"
    
    return unet_adapter_config.exists() or text_encoder_adapter_config.exists()


def is_full_finetuned_model(model_path: str) -> bool:
    """
    Check if the model is a full fine-tuned model by looking for model_index.json.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        True if it's a full fine-tuned model
    """
    model_path = Path(model_path)
    return (model_path / "model_index.json").exists()


def create_output_directory(config_path: str, base_dir: str = "inference_results") -> Path:
    """
    Create timestamped output directory with config file name.
    
    Args:
        config_path: Path to the config file
        base_dir: Base directory for outputs
    
    Returns:
        Path to the created output directory
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract config file name (without extension)
    config_name = Path(config_path).stem
    
    # Create directory name: timestamp_configname
    dir_name = f"{timestamp}_{config_name}"
    
    # Create full output path
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def load_pipeline(
    base_model_path: str,
    model_path: str,
    device: str = "cuda",
    torch_dtype=torch.float16,
    model_type: str = "SD"  # Add model_type parameter
) -> DiffusionPipeline:
    """
    Load a DiffusionPipeline, automatically detecting whether it's LoRA, full fine-tuned, SD or LCM.

    Args:
        base_model_path: Path to the base model
        model_path: Path to the fine-tuned model directory (LoRA or full)
        device: Device to load the model on
        torch_dtype: Torch dtype for the model
        model_type: Model type ("SD" or "LCM")

    Returns:
        DiffusionPipeline with fine-tuned weights applied
    """
    model_path = Path(model_path)

    # Determine if base model is LCM
    is_lcm = model_type == "LCM" or is_lcm_model(base_model_path)

    # Choose pipeline class based on model type
    PipelineClass = LatentConsistencyModelPipeline if is_lcm else DiffusionPipeline

    # Check if it's a full fine-tuned model
    if is_full_finetuned_model(model_path):
        print(f"Detected full fine-tuned {'LCM' if is_lcm else 'SD'} model at: {model_path}")
        print("Loading full fine-tuned pipeline...")
        pipeline = PipelineClass.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    # Check if it's a LoRA model
    elif is_lora_model(model_path):
        print(f"Detected LoRA model at: {model_path}")
        print(f"Loading base {'LCM' if is_lcm else 'SD'} pipeline and applying LoRA weights...")

        # Load the base pipeline
        pipeline = PipelineClass.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Load LoRA weights for UNet
        unet_lora_path = model_path / "unet"
        if unet_lora_path.exists() and (unet_lora_path / "adapter_config.json").exists():
            print(f"Loading UNet LoRA from: {unet_lora_path}")
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(unet_lora_path))
            # Merge LoRA weights for faster inference
            pipeline.unet = pipeline.unet.merge_and_unload()

        # Load LoRA weights for text encoder if available
        text_encoder_lora_path = model_path / "text_encoder"
        if text_encoder_lora_path.exists() and (text_encoder_lora_path / "adapter_config.json").exists():
            print(f"Loading Text Encoder LoRA from: {text_encoder_lora_path}")
            pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, str(text_encoder_lora_path))
            # Merge LoRA weights for faster inference
            pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()

    else:
        print(f"Could not determine model type at: {model_path}")
        print(f"Falling back to loading as base {'LCM' if is_lcm else 'SD'} model...")
        pipeline = PipelineClass.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    # Set scheduler based on model type
    if is_lcm:
        # LCM uses its own scheduler - keep it as is
        print("Using LCM scheduler")
    else:
        # Use DPM Solver for faster inference (SD models)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        print("Using DPM Solver scheduler")

    # Move to device
    pipeline = pipeline.to(device)

    return pipeline


def load_lora_pipeline(
    base_model_path: str,
    lora_model_path: str,
    device: str = "cuda",
    torch_dtype=torch.float16
) -> DiffusionPipeline:
    """
    Load a DiffusionPipeline with LoRA weights applied.
    DEPRECATED: Use load_pipeline() instead for automatic model type detection.
    
    Args:
        base_model_path: Path to the base model
        lora_model_path: Path to the LoRA model directory
        device: Device to load the model on
        torch_dtype: Torch dtype for the model
    
    Returns:
        DiffusionPipeline with LoRA weights applied
    """
    print("Warning: load_lora_pipeline() is deprecated. Use load_pipeline() for better model type detection.")
    return load_pipeline(base_model_path, lora_model_path, device, torch_dtype)


def generate_images(
    pipeline: DiffusionPipeline,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    height: int = 512,
    width: int = 512,
    generator_seed: Optional[int] = None,
) -> List:
    """
    Generate images using the pipeline.
    
    Args:
        pipeline: The loaded DiffusionPipeline
        prompt: The prompt to generate images for
        negative_prompt: Negative prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        num_images_per_prompt: Number of images to generate per prompt
        height: Height of generated images
        width: Width of generated images
        generator_seed: Seed for reproducible generation
    
    Returns:
        List of generated PIL Images and the generator used
    """
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(generator_seed)
    
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        generator=generator,
    ).images
    
    return images, generator


def main(config_path: str = "configs/config.yaml"):
    """Main inference function."""
    # Setup logging
    with log_script_execution("inference"):
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Setup paths
        base_model_path = config.pretrained_model_name_or_path
        lora_model_path = config.output_dir
        
        print(f"Base model: {base_model_path}")
        print(f"Fine-tuned model path: {lora_model_path}")
        
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Using device: {device}")
        print(f"Using dtype: {torch_dtype}")
        
        # Get inference config
        inference_config = config.inference
        
        # Ensure prompts is a list
        if hasattr(inference_config, 'prompts') and inference_config.prompts:
            prompts = inference_config.prompts
        elif hasattr(inference_config, 'prompt') and inference_config.prompt:
            prompts = [inference_config.prompt]  # Convert single prompt to list
        else:
            prompts = ["A photo of sks person"]  # Default
        
        print(f"Will generate images for {len(prompts)} prompts")
        
        # Create timestamped output directory
        output_dir = create_output_directory(config_path)
        print(f"Output directory: {output_dir}")

        # Get model type from config
        model_type = getattr(config, 'model_type', 'SD')
        print(f"Model type: {model_type}")

        # Load fine-tuned pipeline (auto-detect LoRA vs full fine-tuned, SD vs LCM)
        print("Loading fine-tuned pipeline...")
        pipeline = load_pipeline(
            base_model_path=base_model_path,
            model_path=lora_model_path,
            device=device,
            torch_dtype=torch_dtype,
            model_type=model_type
        )
        
        # Load base model pipeline if comparison is requested
        base_pipeline = None
        if hasattr(inference_config, 'compare_with_base_model') and inference_config.compare_with_base_model:
            print("Loading base model pipeline for comparison...")
            PipelineClass = LatentConsistencyModelPipeline if model_type == "LCM" else DiffusionPipeline
            base_pipeline = PipelineClass.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
        
        # Generate images for each prompt
        total_images = 0
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n[{prompt_idx+1}/{len(prompts)}] Generating images for prompt: '{prompt}'")
            
            # Generate with fine-tuned model
            print("  -> Generating with fine-tuned model...")
            finetuned_images, generator = generate_images(
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=inference_config.negative_prompt,
                num_inference_steps=inference_config.num_inference_steps,
                guidance_scale=inference_config.guidance_scale,
                num_images_per_prompt=inference_config.num_images_per_prompt,
                height=inference_config.height,
                width=inference_config.width,
                generator_seed=inference_config.generator_seed,
            )
            
            # Save fine-tuned model images
            for img_idx, image in enumerate(finetuned_images):
                filename = f"finetuned_prompt{prompt_idx:02d}_img{img_idx:02d}.png"
                output_path = output_dir / filename
                image.save(output_path)
                print(f"     Saved: {output_path}")
                total_images += 1
            
            # Generate with base model if requested
            if base_pipeline:
                print("  -> Generating with base model...")
                # Use the same generator seed for comparison
                base_images, _ = generate_images(
                    pipeline=base_pipeline,
                    prompt=prompt,
                    negative_prompt=inference_config.negative_prompt,
                    num_inference_steps=inference_config.num_inference_steps,
                    guidance_scale=inference_config.guidance_scale,
                    num_images_per_prompt=inference_config.num_images_per_prompt,
                    height=inference_config.height,
                    width=inference_config.width,
                    generator_seed=inference_config.generator_seed,
                )
                
                # Save base model images
                for img_idx, image in enumerate(base_images):
                    filename = f"base_prompt{prompt_idx:02d}_img{img_idx:02d}.png"
                    output_path = output_dir / filename
                    image.save(output_path)
                    print(f"     Saved: {output_path}")
                    total_images += 1
        
        print(f"\nGenerated {total_images} images successfully in {output_dir}!")
        
        # Create a summary file
        summary_path = output_dir / "generation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Image Generation Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Base Model: {base_model_path}\n")
            f.write(f"Fine-tuned Model: {lora_model_path}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Torch dtype: {torch_dtype}\n\n")
            f.write(f"Generation Settings:\n")
            f.write(f"- Steps: {inference_config.num_inference_steps}\n")
            f.write(f"- Guidance Scale: {inference_config.guidance_scale}\n")
            f.write(f"- Image Size: {inference_config.width}x{inference_config.height}\n")
            f.write(f"- Seed: {inference_config.generator_seed}\n")
            f.write(f"- Negative Prompt: '{inference_config.negative_prompt}'\n\n")
            f.write(f"Prompts Used:\n")
            for i, prompt in enumerate(prompts):
                f.write(f"{i+1}. {prompt}\n")
            f.write(f"\nTotal Images Generated: {total_images}\n")
            
            if base_pipeline:
                f.write(f"\nComparison with base model: Yes\n")
                f.write(f"- finetuned_promptXX_imgXX.png: Images generated with fine-tuned model\n")
                f.write(f"- base_promptXX_imgXX.png: Images generated with base model\n")
            else:
                f.write(f"\nComparison with base model: No\n")
        
        print(f"Summary saved to: {summary_path}")


def interactive_mode(config_path: str = "configs/config.yaml"):
    """Interactive mode for generating images with custom prompts."""
    # Setup logging
    with log_script_execution("inference_interactive"):
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Setup paths and device
        base_model_path = config.pretrained_model_name_or_path
        lora_model_path = config.output_dir
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Get model type
        model_type = getattr(config, 'model_type', 'SD')

        print("Loading pipeline...")
        pipeline = load_pipeline(
            base_model_path=base_model_path,
            model_path=lora_model_path,
            device=device,
            torch_dtype=torch_dtype,
            model_type=model_type
        )
        
        print("Pipeline loaded! Enter prompts to generate images (type 'quit' to exit):")
        
        # Create timestamped output directory for interactive mode
        output_dir = create_output_directory(config_path, "inference_results/interactive")
        print(f"Output directory: {output_dir}")
        
        image_counter = 0
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not prompt:
                    continue
                
                print(f"Generating image for: '{prompt}'")
                
                images, _ = generate_images(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=config.inference.negative_prompt,
                    num_inference_steps=config.inference.num_inference_steps,
                    guidance_scale=config.inference.guidance_scale,
                    num_images_per_prompt=1,
                    height=config.inference.height,
                    width=config.inference.width,
                )
                
                for image in images:
                    output_path = output_dir / f"interactive_{image_counter:03d}.png"
                    image.save(output_path)
                    print(f"Saved: {output_path}")
                    image_counter += 1
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DreamBooth Inference (LoRA and Full Fine-tuned)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.config)
    else:
        main(args.config)