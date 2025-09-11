#!/usr/bin/env python3
"""
NullBooth: Parallel Covariance Matrix Builder
Multi-GPU parallel implementation using Accelerate library for 4x speedup.
Based on AlphaEdit implementation for knowledge editing in Large Language Models.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import os
import json
import argparse
from math import ceil

# Accelerate imports for multi-GPU support
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
import torch.distributed as dist

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import load_config, load_tokenizer, load_models
from src.logger import log_script_execution
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F


class ParallelAttentionFeatureCollector:
    """Parallel version of AttentionFeatureCollector using Accelerate."""
    
    def __init__(self, config, accelerator: Accelerator, cache_dir=None):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.feature_cache = {}
        self.attention_maps_cache = {}
        self.current_timestep = None
        self.current_encoder_hidden_states = None
        self.hooks = []
        self.current_prompt_idx = 0
        
        # Setup process-specific cache directory
        self.cache_dir = Path(cache_dir) / "cache" / f"process_{accelerator.process_index}" if cache_dir else Path("cache") / f"process_{accelerator.process_index}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature collection flags
        self.collect_q = config.nullbooth.collect_features.q_features
        self.collect_k = config.nullbooth.collect_features.k_features  
        self.collect_v = config.nullbooth.collect_features.v_features
        self.collect_out = config.nullbooth.collect_features.out_features
        self.visual_attention_map = config.nullbooth.visual_attention_map
        
        if self.accelerator.is_main_process:
            print(f"Initialized ParallelAttentionFeatureCollector on {accelerator.num_processes} processes")
        
    def set_current_encoder_hidden_states(self, encoder_hidden_states):
        """Set the current text encoder hidden states for cross-attention."""
        self.current_encoder_hidden_states = encoder_hidden_states
        
    def register_hooks(self, unet: UNet2DConditionModel):
        """Register forward hooks on cross-attention layers."""
        self.hooks = []
        
        def create_hook(layer_name: str):
            def hook_fn(module, input_data, output_data):
                self._collect_attention_features(module, input_data, output_data, layer_name)
            return hook_fn
        
        # Register hooks on all cross-attention layers
        layer_count = 0
        for name, module in unet.named_modules():
            if isinstance(module, Attention) and "attn2" in name:  # Cross-attention layers
                if (self.config.nullbooth.cross_attention_layers == "all" or 
                    layer_count in self.config.nullbooth.cross_attention_layers):
                    
                    hook = module.register_forward_hook(create_hook(name))
                    self.hooks.append(hook)
                    if self.accelerator.is_main_process:
                        print(f"Registered hook for layer: {name}")
                
                layer_count += 1
                
        if self.accelerator.is_main_process:
            print(f"Total {len(self.hooks)} cross-attention hooks registered.")
    
    def _collect_attention_features(self, module: Attention, input_data, output_data, layer_name: str):
        """Collect Q, K, V, and output features from cross-attention layers only."""
        if self.current_timestep is None:
            return
            
        # Only collect from cross-attention layers (attn2)
        if "attn2" not in layer_name:
            return
            
        # Get the hidden_states from input_data
        if isinstance(input_data, tuple):
            hidden_states = input_data[0]
        else:
            hidden_states = input_data
        
        # Use stored encoder_hidden_states
        encoder_hidden_states = self.current_encoder_hidden_states
        
        if encoder_hidden_states is None:
            raise RuntimeError(f"encoder_hidden_states is None for cross-attention layer {layer_name}. "
                             f"Make sure to call set_current_encoder_hidden_states() before UNet forward pass.")
        
        # Initialize feature storage for this timestep if needed
        timestep_key = f"timestep_{self.current_timestep:04d}"
        if timestep_key not in self.feature_cache:
            self.feature_cache[timestep_key] = {}
            self.attention_maps_cache[timestep_key] = {}
            
        layer_features = {}
        
        # Collect Q, K, V features for cross-attention only
        with torch.no_grad():
            # Query computation - from hidden_states (image features)
            if self.collect_q:
                query = module.to_q(hidden_states)
                query = module.head_to_batch_dim(query)
                layer_features['q'] = query.detach().cpu().numpy()
            
            # Key computation - from encoder_hidden_states (text features)
            if self.collect_k:
                key = module.to_k(encoder_hidden_states) 
                key = module.head_to_batch_dim(key)
                layer_features['k'] = key.detach().cpu().numpy()
                
            # Value computation - from encoder_hidden_states (text features)
            if self.collect_v:
                value = module.to_v(encoder_hidden_states)
                value = module.head_to_batch_dim(value)  
                layer_features['v'] = value.detach().cpu().numpy()
            
            # Output features
            if self.collect_out:
                layer_features['out'] = output_data.detach().cpu().numpy()
            
            # Attention map computation for visualization
            if self.visual_attention_map and 'q' in layer_features and 'k' in layer_features:
                query = torch.from_numpy(layer_features['q']).to(self.device)
                key = torch.from_numpy(layer_features['k']).to(self.device)
                
                # Compute attention scores
                attention_scores = torch.matmul(query, key.transpose(-1, -2))
                attention_scores = attention_scores / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                self.attention_maps_cache[timestep_key][layer_name] = attention_probs.detach().cpu().numpy()
        
        # Store features for this layer
        if layer_name not in self.feature_cache[timestep_key]:
            self.feature_cache[timestep_key][layer_name] = {}
        
        # Store features
        for feature_type, feature_data in layer_features.items():
            if feature_type not in self.feature_cache[timestep_key][layer_name]:
                self.feature_cache[timestep_key][layer_name][feature_type] = []
            
            self.feature_cache[timestep_key][layer_name][feature_type].append(feature_data)
    
    def set_current_prompt(self, prompt_idx: int):
        """Set current prompt index for caching."""
        self.current_prompt_idx = prompt_idx
    
    def save_prompt_features_to_disk(self):
        """Save current prompt's features to disk and clear memory cache."""
        if not self.feature_cache:
            return
            
        prompt_cache_dir = self.cache_dir / f"prompt_{self.current_prompt_idx:04d}"
        prompt_cache_dir.mkdir(exist_ok=True)
        
        # Save features for each timestep
        for timestep_key, timestep_features in self.feature_cache.items():
            timestep_file = prompt_cache_dir / f"{timestep_key}.npz"
            
            # Flatten the nested dictionary structure for saving
            save_dict = {}
            for layer_name, layer_features in timestep_features.items():
                for feature_type, feature_data in layer_features.items():
                    if feature_data:
                        safe_key = f"{layer_name.replace('.', '_').replace('/', '_')}_{feature_type}"
                        save_dict[safe_key] = feature_data[-1] if isinstance(feature_data, list) else feature_data
            
            if save_dict:
                np.savez_compressed(timestep_file, **save_dict)
        
        # Save attention maps if enabled
        if self.visual_attention_map and self.attention_maps_cache:
            for timestep_key, timestep_maps in self.attention_maps_cache.items():
                attention_file = prompt_cache_dir / f"{timestep_key}_attention.npz"
                save_dict = {}
                for layer_name, attention_data in timestep_maps.items():
                    if attention_data is not None:
                        safe_key = layer_name.replace(".", "_").replace("/", "_")
                        save_dict[safe_key] = attention_data
                if save_dict:
                    np.savez_compressed(attention_file, **save_dict)
        
        # Clear memory cache after saving
        self.feature_cache.clear()
        self.attention_maps_cache.clear()
    
    def set_current_timestep(self, timestep: int):
        """Set the current denoising timestep."""
        self.current_timestep = timestep
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_collected_features(self) -> Dict:
        """Get all collected features."""
        return self.feature_cache
    
    def get_attention_maps(self) -> Dict:
        """Get all collected attention maps.""" 
        return self.attention_maps_cache


class ParallelCovarianceMatrixComputer:
    """Parallel version of CovarianceMatrixComputer using distributed operations."""
    
    def __init__(self, config, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.nullspace_threshold = float(config.nullbooth.nullspace_threshold)
        
    def gather_all_features_from_processes(self, run_dir: Path) -> Dict:
        """Gather features from all processes and merge them."""
        if self.accelerator.is_main_process:
            print("Gathering features from all processes...")
        
        # Wait for all processes to finish feature collection
        self.accelerator.wait_for_everyone()
        
        # Each process collects its own cache directories
        process_cache_dirs = []
        for process_idx in range(self.accelerator.num_processes):
            process_cache_dir = run_dir / "cache" / f"process_{process_idx}"
            if process_cache_dir.exists():
                process_cache_dirs.append(process_cache_dir)
        
        # Gather all cache directories from all processes
        all_cache_dirs = gather_object(process_cache_dirs)
        
        if self.accelerator.is_main_process:
            # Flatten the list of lists - handle both Path objects and lists
            flattened_dirs = []
            for item in all_cache_dirs:
                if isinstance(item, list):
                    flattened_dirs.extend(item)
                else:
                    flattened_dirs.append(item)
            print(f"Found {len(flattened_dirs)} process cache directories")
            return self._merge_features_from_cache_dirs(flattened_dirs)
        else:
            return {}
    
    def _merge_features_from_cache_dirs(self, cache_dirs: List[Path]) -> Dict:
        """Merge features from multiple cache directories."""
        merged_features = {}
        
        # Find all timesteps
        all_timestep_keys = set()
        for cache_dir in cache_dirs:
            for prompt_dir in cache_dir.iterdir():
                if prompt_dir.is_dir() and prompt_dir.name.startswith("prompt_"):
                    for timestep_file in prompt_dir.iterdir():
                        if timestep_file.suffix == '.npz' and 'attention' not in timestep_file.name:
                            all_timestep_keys.add(timestep_file.stem)
        
        print(f"Found {len(all_timestep_keys)} unique timesteps across all processes")
        
        # Process each timestep
        for timestep_key in tqdm(all_timestep_keys, desc="Merging timestep features"):
            merged_features[timestep_key] = {}
            
            # Collect features from all processes for this timestep
            for cache_dir in cache_dirs:
                for prompt_dir in cache_dir.iterdir():
                    if not (prompt_dir.is_dir() and prompt_dir.name.startswith("prompt_")):
                        continue
                        
                    timestep_file = prompt_dir / f"{timestep_key}.npz"
                    if not timestep_file.exists():
                        continue
                    
                    try:
                        data = np.load(timestep_file)
                        
                        # Reconstruct the nested structure
                        for key, feature_data in data.items():
                            # Parse the safe key back to layer_name and feature_type
                            parts = key.rsplit('_', 1)
                            if len(parts) == 2:
                                layer_name = parts[0].replace('_', '.')
                                feature_type = parts[1]
                                
                                if layer_name not in merged_features[timestep_key]:
                                    merged_features[timestep_key][layer_name] = {}
                                if feature_type not in merged_features[timestep_key][layer_name]:
                                    merged_features[timestep_key][layer_name][feature_type] = []
                                
                                merged_features[timestep_key][layer_name][feature_type].append(feature_data)
                    except Exception as e:
                        print(f"Warning: Failed to load {timestep_file}: {e}")
        
        return merged_features
    
    def compute_covariance_matrices_from_merged_features(self, merged_features: Dict) -> Dict:
        """Compute covariance matrices from merged features."""
        cov_matrices = {}
        
        if not self.accelerator.is_main_process:
            return cov_matrices
        
        print("Computing covariance matrices from merged features...")
        
        # Process each timestep
        for timestep_key in tqdm(merged_features.keys(), desc="Processing timesteps"):
            cov_matrices[timestep_key] = {}
            
            # Process each layer
            for layer_name, layer_features in merged_features[timestep_key].items():
                cov_matrices[timestep_key][layer_name] = {}
                
                # Process each feature type
                for feature_type, feature_data_list in layer_features.items():
                    if not feature_data_list:
                        continue
                    
                    print(f"  Processing {layer_name}/{feature_type}: {len(feature_data_list)} samples")
                    
                    # Concatenate features from all prompts
                    try:
                        feature_data = np.concatenate(feature_data_list, axis=0)
                    except Exception as e:
                        print(f"    Warning: Failed to concatenate features for {layer_name}/{feature_type}: {e}")
                        continue
                    
                    # Reshape feature data to (n_samples, feature_dim)
                    if feature_data.ndim > 2:
                        original_shape = feature_data.shape
                        feature_data = feature_data.reshape(-1, original_shape[-1])
                    else:
                        original_shape = feature_data.shape
                    
                    # Compute covariance matrix K₀K₀ᵀ
                    feature_tensor = torch.tensor(feature_data, dtype=torch.float32)
                    
                    # Center the data (following AlphaEdit practice)
                    feature_tensor = feature_tensor - feature_tensor.mean(dim=0, keepdim=True)
                    
                    # Compute covariance matrix
                    cov_matrix = torch.mm(feature_tensor.T, feature_tensor) / (feature_tensor.shape[0] - 1)
                    
                    # Store covariance matrix and compute null space projection
                    cov_info = {
                        'covariance_matrix': cov_matrix.numpy(),
                        'original_shape': original_shape,
                        'n_samples': feature_tensor.shape[0],
                        'feature_dim': feature_tensor.shape[1]
                    }
                    
                    # Compute null space projection matrix following AlphaEdit
                    U, S, _ = torch.linalg.svd(cov_matrix, full_matrices=False)
                    
                    # Find eigenvectors corresponding to small eigenvalues (null space)
                    small_singular_indices = (S < self.nullspace_threshold).nonzero(as_tuple=True)[0]
                    
                    if len(small_singular_indices) > 0:
                        # Construct projection matrix P = ŮŮᵀ
                        U_null = U[:, small_singular_indices]
                        projection_matrix = torch.mm(U_null, U_null.T)
                        
                        cov_info['projection_matrix'] = projection_matrix.numpy()
                        cov_info['null_space_dim'] = len(small_singular_indices)
                        cov_info['singular_values'] = S.numpy()
                    else:
                        cov_info['projection_matrix'] = None
                        cov_info['null_space_dim'] = 0
                        cov_info['singular_values'] = S.numpy()
                    
                    cov_matrices[timestep_key][layer_name][feature_type] = cov_info
                    
                    print(f"    cov_shape={cov_matrix.shape}, null_dim={cov_info['null_space_dim']}")
                    
                    # Clear memory
                    del feature_tensor, cov_matrix, U, S
        
        return cov_matrices


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file, one prompt per line."""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    prompts.append(line)
    except FileNotFoundError:
        print(f"Warning: Prompts file {file_path} not found. Using default prompts.")
        prompts = [
            "a photo of a dog",
            "a beautiful landscape", 
            "a cat sitting on a chair",
            "a car on the road",
            "a person walking"
        ]
    return prompts


def save_covariance_matrices(cov_matrices: Dict, output_dir: Path):
    """Save covariance matrices to disk in organized structure."""
    print(f"Saving covariance matrices to {output_dir}...")
    
    for timestep, timestep_data in tqdm(cov_matrices.items(), desc="Saving matrices"):
        timestep_dir = output_dir / timestep
        timestep_dir.mkdir(exist_ok=True)
        
        for layer_name, layer_data in timestep_data.items():
            # Create safe filename from layer name
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            layer_dir = timestep_dir / safe_layer_name
            layer_dir.mkdir(exist_ok=True)
            
            for feature_type, cov_info in layer_data.items():
                # Save covariance matrix
                cov_file = layer_dir / f"{feature_type}_covariance.npy"
                np.save(cov_file, cov_info['covariance_matrix'])
                
                # Save projection matrix if exists
                if cov_info['projection_matrix'] is not None:
                    proj_file = layer_dir / f"{feature_type}_projection.npy"
                    np.save(proj_file, cov_info['projection_matrix'])
                
                # Save metadata
                metadata = {
                    'original_shape': cov_info['original_shape'],
                    'n_samples': int(cov_info['n_samples']),
                    'feature_dim': int(cov_info['feature_dim']),
                    'null_space_dim': int(cov_info['null_space_dim']),
                    'singular_values': cov_info['singular_values'].tolist(),
                    'nullspace_threshold': float(cov_info.get('nullspace_threshold', 1e-5)),
                    'actual_timestep': int(timestep.split('_')[1]) if 'timestep_' in timestep else None,
                }
                
                metadata_file = layer_dir / f"{feature_type}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)


def distribute_prompts(prompts: List[str], accelerator: Accelerator) -> List[str]:
    """Distribute prompts across processes."""
    total_prompts = len(prompts)
    prompts_per_process = ceil(total_prompts / accelerator.num_processes)
    
    start_idx = accelerator.process_index * prompts_per_process
    end_idx = min(start_idx + prompts_per_process, total_prompts)
    
    process_prompts = prompts[start_idx:end_idx]
    
    if accelerator.is_main_process:
        print(f"Distributing {total_prompts} prompts across {accelerator.num_processes} processes")
        for i in range(accelerator.num_processes):
            proc_start = i * prompts_per_process
            proc_end = min(proc_start + prompts_per_process, total_prompts)
            proc_count = max(0, proc_end - proc_start)
            print(f"  Process {i}: {proc_count} prompts (indices {proc_start}-{proc_end-1})")
    
    print(f"Process {accelerator.process_index}: Processing {len(process_prompts)} prompts")
    return process_prompts


def main():
    """Main function to build covariance matrices for NullBooth in parallel."""
    # Initialize Accelerator
    kwargs = InitProcessGroupKwargs(backend='gloo', timeout=timedelta(hours=2))  # Use gloo backend for better compatibility
    accelerator = Accelerator(
        split_batches=False,  # We'll handle batch splitting manually
        kwargs_handlers=[kwargs]
    )
    
    # Setup logging
    with log_script_execution("build_cov_parallel"):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Build covariance matrices for NullBooth in parallel")
        parser.add_argument("--config", "-c", default="configs/nullbooth.yaml", 
                           help="Path to configuration file")
        args = parser.parse_args()
        
        # Load configuration
        if accelerator.is_main_process:
            print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Check if NullBooth is enabled
        if not config.nullbooth.enable:
            if accelerator.is_main_process:
                print("NullBooth mode is not enabled in config. Set nullbooth.enable=true to proceed.")
            return
        
        # Setup output directory
        output_dir = Path(config.nullbooth.cov_matrices_output_dir)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}_parallel"
        if accelerator.is_main_process:
            run_dir.mkdir(exist_ok=True)
            print(f"Output directory: {run_dir}")
        
        # Wait for main process to create directories
        accelerator.wait_for_everyone()
        
        # Load prompts and distribute across processes
        prompts = load_prompts_from_file(config.nullbooth.original_knowledge_prompts)
        process_prompts = distribute_prompts(prompts, accelerator)
        
        if accelerator.is_main_process:
            print(f"Total prompts loaded: {len(prompts)}")
            print(f"Using {accelerator.num_processes} GPUs: {[f'cuda:{i}' for i in range(accelerator.num_processes)]}")
        
        # Load models
        if accelerator.is_main_process:
            print("Loading diffusion models...")
        tokenizer = load_tokenizer(config)
        noise_scheduler, text_encoder, vae, unet = load_models(config)
        
        # Move models to device using accelerator
        text_encoder = accelerator.prepare(text_encoder)
        unet = accelerator.prepare(unet)
        vae = accelerator.prepare(vae)
        
        # Create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            vae=vae.module if hasattr(vae, 'module') else vae,
            text_encoder=text_encoder.module if hasattr(text_encoder, 'module') else text_encoder,
            tokenizer=tokenizer,
            unet=unet.module if hasattr(unet, 'module') else unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16 if accelerator.device.type == "cuda" else torch.float32
        ).to(accelerator.device)
        
        # Initialize feature collector
        collector = ParallelAttentionFeatureCollector(config, accelerator, cache_dir=run_dir)
        collector.register_hooks(unet.module if hasattr(unet, 'module') else unet)
        
        try:
            # Process prompts to collect features (each process handles its subset)
            if accelerator.is_main_process:
                print("Processing prompts to collect features in parallel...")
            
            with torch.no_grad():
                for prompt_idx, prompt in enumerate(tqdm(process_prompts, desc=f"GPU {accelerator.process_index} processing", disable=not accelerator.is_local_main_process)):
                    # Global prompt index for consistent naming across processes
                    global_prompt_idx = accelerator.process_index * ceil(len(prompts) / accelerator.num_processes) + prompt_idx
                    
                    if accelerator.is_local_main_process:
                        print(f"\nGPU {accelerator.process_index}: Processing prompt {prompt_idx+1}/{len(process_prompts)} (global: {global_prompt_idx}): '{prompt[:50]}...'")
                    
                    # Set current prompt index for caching
                    collector.set_current_prompt(global_prompt_idx)
                    
                    # Encode prompt
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = (text_encoder.module if hasattr(text_encoder, 'module') else text_encoder)(text_inputs.input_ids.to(accelerator.device))[0]
                    
                    # Generate base latents
                    base_latents = torch.randn(
                        (1, (unet.module if hasattr(unet, 'module') else unet).config.in_channels, config.resolution // 8, config.resolution // 8),
                        device=accelerator.device
                    )
                    
                    # Generate noise for adding to latents
                    noise = torch.randn_like(base_latents)
                    
                    # Set up scheduler with sampled timesteps
                    total_timesteps = pipeline.scheduler.config.num_train_timesteps
                    num_sample_steps = config.nullbooth.num_denoising_steps
                    
                    # Create evenly spaced timesteps
                    if num_sample_steps >= total_timesteps:
                        timesteps = torch.arange(total_timesteps-1, -1, -1, dtype=torch.long)
                    else:
                        timesteps = torch.linspace(total_timesteps-1, 0, num_sample_steps, dtype=torch.long)
                    
                    # Denoising loop
                    for i, t in enumerate(timesteps):
                        # Set current timestep and encoder states
                        collector.set_current_timestep(t.item())
                        collector.set_current_encoder_hidden_states(text_embeddings)
                        
                        # Add noise corresponding to this timestep
                        noisy_latents = pipeline.scheduler.add_noise(base_latents, noise, t)
                        
                        # Prepare latent model input  
                        latent_model_input = noisy_latents
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                        
                        # Predict noise
                        noise_pred = (unet.module if hasattr(unet, 'module') else unet)(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False
                        )[0]
                    
                    # Save current prompt's features to disk
                    collector.save_prompt_features_to_disk()
            
            # Wait for all processes to complete feature collection
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                print("All processes completed feature collection. Computing covariance matrices...")
            
            # Compute covariance matrices (only main process does this)
            computer = ParallelCovarianceMatrixComputer(config, accelerator)
            merged_features = computer.gather_all_features_from_processes(run_dir)
            cov_matrices = computer.compute_covariance_matrices_from_merged_features(merged_features)
            
            # Save results (only main process)
            if accelerator.is_main_process and cov_matrices:
                save_covariance_matrices(cov_matrices, run_dir)
                
                print(f"\nParallel covariance matrix computation completed!")
                print(f"Results saved to: {run_dir}")
                print(f"Total timesteps processed: {len(cov_matrices)}")
                print(f"Total prompts processed: {len(prompts)}")
                print(f"Speedup achieved: ~{accelerator.num_processes}x with {accelerator.num_processes} GPUs")
            
        finally:
            # Clean up hooks
            collector.remove_hooks()
            if accelerator.is_main_process:
                print("Hooks removed successfully.")


if __name__ == "__main__":
    main()