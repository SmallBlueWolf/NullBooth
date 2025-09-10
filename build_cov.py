#!/usr/bin/env python3
"""
NullBooth: Covariance Matrix Builder
Build covariance matrices for AlphaEdit-style null-space constrained editing in Diffusion models.
Based on AlphaEdit implementation for knowledge editing in Large Language Models.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from tqdm.auto import tqdm
import os
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import load_config, load_tokenizer, load_models
from src.logger import log_script_execution
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F


class AttentionFeatureCollector:
    """Collects Q, K, V, and output features from cross-attention layers during diffusion."""
    
    def __init__(self, config, device="cuda", cache_dir=None):
        self.config = config
        self.device = device
        self.feature_cache = {}
        self.attention_maps_cache = {}
        self.current_timestep = None
        self.current_encoder_hidden_states = None  # Store current text embeddings
        self.hooks = []
        self.current_prompt_idx = 0
        
        # Setup disk cache directory
        self.cache_dir = Path(cache_dir) / "cache" if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature collection flags
        self.collect_q = config.nullbooth.collect_features.q_features
        self.collect_k = config.nullbooth.collect_features.k_features  
        self.collect_v = config.nullbooth.collect_features.v_features
        self.collect_out = config.nullbooth.collect_features.out_features
        self.visual_attention_map = config.nullbooth.visual_attention_map
        
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
                    print(f"Registered hook for layer: {name}")
                
                layer_count += 1  # Increment after processing each cross-attention layer
                
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
        
        # Use stored encoder_hidden_states (set before each UNet forward pass)
        encoder_hidden_states = self.current_encoder_hidden_states
        
        # For cross-attention, encoder_hidden_states must not be None
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
        
        # Store features for this layer (accumulate across prompts)
        if layer_name not in self.feature_cache[timestep_key]:
            self.feature_cache[timestep_key][layer_name] = {}
        
        # Accumulate features across different prompts
        for feature_type, feature_data in layer_features.items():
            if feature_type not in self.feature_cache[timestep_key][layer_name]:
                self.feature_cache[timestep_key][layer_name][feature_type] = []
            
            # Append current prompt's features to the list
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
                    if feature_data:  # Only save if we have data
                        safe_key = f"{layer_name.replace('.', '_').replace('/', '_')}_{feature_type}"
                        # Take the last item since we only have current prompt's data
                        save_dict[safe_key] = feature_data[-1] if isinstance(feature_data, list) else feature_data
            
            if save_dict:  # Only save if we have data
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
        print(f"Saved features for prompt {self.current_prompt_idx} to disk and cleared memory cache.")
    
    def load_cached_features_batch(self, timestep_key: str, batch_size: int = 50) -> Dict:
        """Load cached features for a specific timestep in batches to avoid memory issues."""
        print(f"Loading cached features for {timestep_key} in batches of {batch_size}...")
        
        # Find all prompt cache directories
        prompt_dirs = sorted([d for d in self.cache_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")])
        
        # Initialize result structure
        result = {}
        
        # Process in batches
        for batch_start in range(0, len(prompt_dirs), batch_size):
            batch_end = min(batch_start + batch_size, len(prompt_dirs))
            batch_dirs = prompt_dirs[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//batch_size + 1}/{(len(prompt_dirs) + batch_size - 1)//batch_size} "
                  f"(prompts {batch_start}-{batch_end-1})")
            
            # Load features from this batch
            batch_features = {}
            for prompt_dir in batch_dirs:
                timestep_file = prompt_dir / f"{timestep_key}.npz"
                if timestep_file.exists():
                    try:
                        data = np.load(timestep_file)
                        
                        # Reconstruct the nested structure
                        for key, feature_data in data.items():
                            # Parse the safe key back to layer_name and feature_type
                            parts = key.rsplit('_', 1)
                            if len(parts) == 2:
                                layer_name = parts[0].replace('_', '.')
                                feature_type = parts[1]
                                
                                if layer_name not in batch_features:
                                    batch_features[layer_name] = {}
                                if feature_type not in batch_features[layer_name]:
                                    batch_features[layer_name][feature_type] = []
                                
                                batch_features[layer_name][feature_type].append(feature_data)
                    except Exception as e:
                        print(f"Warning: Failed to load {timestep_file}: {e}")
            
            # Merge batch results into main result
            for layer_name, layer_features in batch_features.items():
                if layer_name not in result:
                    result[layer_name] = {}
                for feature_type, feature_list in layer_features.items():
                    if feature_type not in result[layer_name]:
                        result[layer_name][feature_type] = []
                    result[layer_name][feature_type].extend(feature_list)
        
        print(f"Loaded features from {len(prompt_dirs)} prompts for {timestep_key}")
        return result
    
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


class CovarianceMatrixComputer:
    """Computes covariance matrices from collected features using AlphaEdit methodology."""
    
    def __init__(self, config):
        self.config = config
        # Convert nullspace_threshold to float to ensure proper comparison
        self.nullspace_threshold = float(config.nullbooth.nullspace_threshold)
        
    def compute_covariance_matrices_from_cache(self, collector: AttentionFeatureCollector) -> Dict:
        """
        Compute covariance matrices from cached features using batch loading.
        
        Following AlphaEdit methodology:
        1. Load features in batches to avoid memory issues
        2. Compute K₀K₀ᵀ covariance matrix
        3. Perform SVD decomposition
        4. Identify null space based on threshold
        """
        cov_matrices = {}
        
        print("Computing covariance matrices from disk cache...")
        
        # Get all timesteps from cache directories
        prompt_dirs = sorted([d for d in collector.cache_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")])
        if not prompt_dirs:
            print("No cached features found!")
            return cov_matrices
        
        # Find all timestep keys by checking the first prompt directory
        first_prompt_dir = prompt_dirs[0]
        timestep_files = [f for f in first_prompt_dir.iterdir() if f.suffix == '.npz' and 'attention' not in f.name]
        timestep_keys = [f.stem for f in timestep_files]
        
        print(f"Found {len(timestep_keys)} timesteps in cache: {timestep_keys[:5]}..." if len(timestep_keys) > 5 else f"Found timesteps: {timestep_keys}")
        
        # Process each timestep
        for timestep_key in tqdm(timestep_keys, desc="Processing timesteps"):
            print(f"\nProcessing {timestep_key}...")
            cov_matrices[timestep_key] = {}
            
            # Load features for this timestep in batches
            timestep_features = collector.load_cached_features_batch(timestep_key, batch_size=100)
            
            # Process each layer
            for layer_name, layer_features in timestep_features.items():
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


class VisualizationManager:
    """Handles visualization of attention maps and covariance matrices."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def save_attention_maps(self, attention_maps: Dict):
        """Save attention maps as heatmap visualizations."""
        print("Saving attention map visualizations...")
        
        for timestep, timestep_maps in tqdm(attention_maps.items(), desc="Visualizing attention"):
            timestep_dir = self.viz_dir / timestep
            timestep_dir.mkdir(exist_ok=True)
            
            for layer_name, attention_data in timestep_maps.items():
                if attention_data is None:
                    continue
                    
                # Average across heads and batch dimension for visualization
                if attention_data.ndim == 4:  # [batch, heads, seq_len, seq_len]
                    avg_attention = attention_data.mean(axis=(0, 1))
                elif attention_data.ndim == 3:  # [heads, seq_len, seq_len] 
                    avg_attention = attention_data.mean(axis=0)
                else:
                    avg_attention = attention_data
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(avg_attention, cmap='viridis', cbar=True)
                plt.title(f"Attention Map - {timestep} - {layer_name}")
                plt.xlabel("Key Sequence Position")
                plt.ylabel("Query Sequence Position")
                
                # Save plot
                safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                output_path = timestep_dir / f"attention_{safe_layer_name}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
        print(f"Attention maps saved to {self.viz_dir}")
    
    def save_covariance_summaries(self, cov_matrices: Dict):
        """Save summary visualizations of covariance matrices."""
        summary_dir = self.viz_dir / "covariance_summaries"
        summary_dir.mkdir(exist_ok=True)
        
        # Collect statistics across all timesteps and layers
        stats = {
            'null_space_dims': [],
            'total_dims': [],
            'timesteps': [],
            'layers': [],
            'feature_types': []
        }
        
        for timestep, timestep_data in cov_matrices.items():
            for layer_name, layer_data in timestep_data.items():
                for feature_type, cov_info in layer_data.items():
                    stats['null_space_dims'].append(cov_info['null_space_dim'])
                    stats['total_dims'].append(cov_info['feature_dim'])
                    stats['timesteps'].append(timestep)
                    stats['layers'].append(layer_name)
                    stats['feature_types'].append(feature_type)
        
        # Plot null space dimension distribution
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(stats['null_space_dims'], bins=30, alpha=0.7)
        plt.title("Distribution of Null Space Dimensions")
        plt.xlabel("Null Space Dimension")
        plt.ylabel("Frequency")
        
        plt.subplot(2, 2, 2)
        unique_feature_types = list(set(stats['feature_types']))
        null_dims_by_type = {ft: [] for ft in unique_feature_types}
        for i, ft in enumerate(stats['feature_types']):
            null_dims_by_type[ft].append(stats['null_space_dims'][i])
        
        for ft, dims in null_dims_by_type.items():
            plt.hist(dims, bins=20, alpha=0.5, label=ft)
        plt.title("Null Space Dims by Feature Type")
        plt.xlabel("Null Space Dimension")
        plt.ylabel("Frequency")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        ratios = [nd/td for nd, td in zip(stats['null_space_dims'], stats['total_dims'])]
        plt.hist(ratios, bins=30, alpha=0.7)
        plt.title("Null Space Ratio Distribution")
        plt.xlabel("Null Space Dim / Total Dim")
        plt.ylabel("Frequency")
        
        plt.subplot(2, 2, 4)
        timestep_nulls = {}
        for i, ts in enumerate(stats['timesteps']):
            if ts not in timestep_nulls:
                timestep_nulls[ts] = []
            timestep_nulls[ts].append(stats['null_space_dims'][i])
        
        ts_keys = sorted(timestep_nulls.keys())
        avg_nulls = [np.mean(timestep_nulls[ts]) for ts in ts_keys]
        plt.plot(range(len(ts_keys)), avg_nulls, 'o-')
        plt.title("Average Null Space Dim vs Timestep")
        plt.xlabel("Denoising Step")
        plt.ylabel("Avg Null Space Dimension")
        
        plt.tight_layout()
        plt.savefig(summary_dir / "covariance_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()


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
        timestep_dir = output_dir / timestep  # Use the full timestep key (e.g., "timestep_0980")
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
                    'nullspace_threshold': 1e-5,  # Default threshold from config
                    'actual_timestep': int(timestep.split('_')[1]) if 'timestep_' in timestep else None,  # Store actual timestep value
                }
                
                metadata_file = layer_dir / f"{feature_type}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)


def main():
    """Main function to build covariance matrices for NullBooth."""
    # Setup logging
    with log_script_execution("build_cov"):
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Build covariance matrices for NullBooth")
        parser.add_argument("--config", "-c", default="configs/nullbooth.yaml", 
                           help="Path to configuration file")
        args = parser.parse_args()
        
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Check if NullBooth is enabled
        if not config.nullbooth.enable:
            print("NullBooth mode is not enabled in config. Set nullbooth.enable=true to proceed.")
            return
        
        # Setup output directory
        output_dir = Path(config.nullbooth.cov_matrices_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {run_dir}")
        
        # Load prompts
        prompts = load_prompts_from_file(config.nullbooth.original_knowledge_prompts)
        print(f"Loaded {len(prompts)} prompts for covariance matrix computation")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load models
        print("Loading diffusion models...")
        tokenizer = load_tokenizer(config)
        noise_scheduler, text_encoder, vae, unet = load_models(config)
        
        # Move models to device
        text_encoder = text_encoder.to(device)
        unet = unet.to(device)
        vae = vae.to(device)
        
        # Create pipeline using from_pretrained for proper initialization
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Initialize feature collector with cache directory
        collector = AttentionFeatureCollector(config, device, cache_dir=run_dir)
        collector.register_hooks(unet)
        
        try:
            # Process prompts to collect features
            print("Processing prompts to collect features...")
            
            with torch.no_grad():
                for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
                    print(f"\nProcessing prompt {prompt_idx+1}/{len(prompts)}: '{prompt[:50]}...'")
                    
                    # Set current prompt index for caching
                    collector.set_current_prompt(prompt_idx)
                    # Encode prompt
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
                    
                    # Generate base latents (clean image representation)
                    base_latents = torch.randn(
                        (1, unet.config.in_channels, config.resolution // 8, config.resolution // 8),
                        device=device
                    )
                    
                    # Generate noise for adding to latents
                    noise = torch.randn_like(base_latents)
                    
                    # Set up scheduler with sampled timesteps
                    # Use the specified number of steps as sampling points from the full range
                    total_timesteps = pipeline.scheduler.config.num_train_timesteps  # Usually 1000
                    num_sample_steps = config.nullbooth.num_denoising_steps
                    
                    # Create evenly spaced timesteps from the full range (from high noise to low noise)
                    if num_sample_steps >= total_timesteps:
                        # If requested steps >= total steps, use all timesteps
                        timesteps = torch.arange(total_timesteps-1, -1, -1, dtype=torch.long)
                    else:
                        # Sample evenly spaced timesteps from full range: [999, 980, 960, ..., 20, 0]
                        timesteps = torch.linspace(total_timesteps-1, 0, num_sample_steps, dtype=torch.long)
                    
                    print(f"Sampling {len(timesteps)} timesteps from range [0, {total_timesteps}]")
                    print(f"Selected timesteps: {timesteps.tolist()[:10]}..." if len(timesteps) > 10 else f"Selected timesteps: {timesteps.tolist()}")
                    
                    # Denoising loop
                    for i, t in enumerate(timesteps):
                        # Use actual timestep value for feature collection
                        collector.set_current_timestep(t.item())
                        
                        # Set current encoder hidden states for cross-attention feature collection
                        collector.set_current_encoder_hidden_states(text_embeddings)
                        
                        # Add noise corresponding to this timestep
                        # This creates the correct noise level for timestep t
                        noisy_latents = pipeline.scheduler.add_noise(base_latents, noise, t)
                        
                        # Prepare latent model input  
                        latent_model_input = noisy_latents
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                        
                        # Predict noise
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False
                        )[0]
                        
                        # Update latents (we don't actually need the result, just the forward pass)
                        # latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    
                    # Save current prompt's features to disk and clear memory cache
                    collector.save_prompt_features_to_disk()
                    print(f"Completed processing and caching prompt {prompt_idx+1}/{len(prompts)}")
            
            # Compute covariance matrices from disk cache
            print("\nComputing covariance matrices from cached features...")
            computer = CovarianceMatrixComputer(config)
            cov_matrices = computer.compute_covariance_matrices_from_cache(collector)
            
            # Save covariance matrices
            save_covariance_matrices(cov_matrices, run_dir)
            
            # Handle visualizations
            if config.nullbooth.visual_attention_map:
                print("Creating visualizations...")
                viz_manager = VisualizationManager(run_dir)
                
                # Save attention maps
                attention_maps = collector.get_attention_maps()
                viz_manager.save_attention_maps(attention_maps)
                
                # Save covariance summaries
                viz_manager.save_covariance_summaries(cov_matrices)
            
            print(f"\nCovariance matrix computation completed!")
            print(f"Results saved to: {run_dir}")
            print(f"Total timesteps processed: {len(cov_matrices)}")
            print(f"Total prompts processed: {len(prompts)}")
            
        finally:
            # Clean up hooks
            collector.remove_hooks()
            print("Hooks removed successfully.")


if __name__ == "__main__":
    main()