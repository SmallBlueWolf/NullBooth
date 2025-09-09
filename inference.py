#!/usr/bin/env python3
"""
DreamBooth LoRA Inference Script
Load and run inference with trained DreamBooth LoRA models.
"""

import sys
import torch
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import load_config
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


def load_lora_pipeline(
    base_model_path: str,
    lora_model_path: str,
    device: str = "cuda",
    torch_dtype=torch.float16
) -> DiffusionPipeline:
    """
    Load a DiffusionPipeline with LoRA weights applied.
    
    Args:
        base_model_path: Path to the base model
        lora_model_path: Path to the LoRA model directory
        device: Device to load the model on
        torch_dtype: Torch dtype for the model
    
    Returns:
        DiffusionPipeline with LoRA weights applied
    """
    # Load the base pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # Load LoRA weights for UNet
    unet_lora_path = Path(lora_model_path) / "unet"
    if unet_lora_path.exists():
        print(f"Loading UNet LoRA from: {unet_lora_path}")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)
        # Merge LoRA weights for faster inference
        pipeline.unet = pipeline.unet.merge_and_unload()
    
    # Load LoRA weights for text encoder if available
    text_encoder_lora_path = Path(lora_model_path) / "text_encoder"
    if text_encoder_lora_path.exists():
        print(f"Loading Text Encoder LoRA from: {text_encoder_lora_path}")
        pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, text_encoder_lora_path)
        # Merge LoRA weights for faster inference
        pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
    
    # Use DPM Solver for faster inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Move to device
    pipeline = pipeline.to(device)
    
    return pipeline


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
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Setup paths
    base_model_path = config.pretrained_model_name_or_path
    lora_model_path = config.output_dir
    
    print(f"Base model: {base_model_path}")
    print(f"LoRA model path: {lora_model_path}")
    
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
    
    # Setup output directory
    output_dir = Path(inference_config.output_dir) if hasattr(inference_config, 'output_dir') else Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    
    # Load LoRA pipeline
    print("Loading LoRA pipeline...")
    lora_pipeline = load_lora_pipeline(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Load base model pipeline if comparison is requested
    base_pipeline = None
    if hasattr(inference_config, 'compare_with_base_model') and inference_config.compare_with_base_model:
        print("Loading base model pipeline for comparison...")
        base_pipeline = DiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
    
    # Generate images for each prompt
    total_images = 0
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] Generating images for prompt: '{prompt}'")
        
        # Generate with LoRA model
        print("  -> Generating with LoRA model...")
        lora_images, generator = generate_images(
            pipeline=lora_pipeline,
            prompt=prompt,
            negative_prompt=inference_config.negative_prompt,
            num_inference_steps=inference_config.num_inference_steps,
            guidance_scale=inference_config.guidance_scale,
            num_images_per_prompt=inference_config.num_images_per_prompt,
            height=inference_config.height,
            width=inference_config.width,
            generator_seed=inference_config.generator_seed,
        )
        
        # Save LoRA images
        for img_idx, image in enumerate(lora_images):
            filename = f"lora_prompt{prompt_idx:02d}_img{img_idx:02d}.png"
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
        f.write(f"LoRA Model: {lora_model_path}\n")
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
            f.write(f"- lora_promptXX_imgXX.png: Images generated with LoRA model\n")
            f.write(f"- base_promptXX_imgXX.png: Images generated with base model\n")
        else:
            f.write(f"\nComparison with base model: No\n")
    
    print(f"Summary saved to: {summary_path}")


def interactive_mode(config_path: str = "configs/config.yaml"):
    """Interactive mode for generating images with custom prompts."""
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Setup paths and device
    base_model_path = config.pretrained_model_name_or_path
    lora_model_path = config.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print("Loading pipeline...")
    pipeline = load_lora_pipeline(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        device=device,
        torch_dtype=torch_dtype
    )
    
    print("Pipeline loaded! Enter prompts to generate images (type 'quit' to exit):")
    
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    
    image_counter = 0
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
            
            print(f"Generating image for: '{prompt}'")
            
            images = generate_images(
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
    
    parser = argparse.ArgumentParser(description="DreamBooth LoRA Inference")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.config)
    else:
        main(args.config)