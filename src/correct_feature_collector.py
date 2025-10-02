"""
Correct Feature Collection for AlphaEdit-style NullBooth
Collects input features to attention components and builds covariance matrices
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import gc


class AlphaEditFeatureCollector:
    """
    Collects input features to attention components (to_q, to_k, to_v, to_out)
    following the exact AlphaEdit methodology.
    """

    def __init__(
        self,
        config,
        accelerator,
        cache_dir=None
    ):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device

        # Cache management
        self.cache_dir = Path(cache_dir) / "cache" if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature storage - key: (timestep, layer_name, component_type)
        self.collected_features = {}
        self.hooks = []

        print("AlphaEditFeatureCollector initialized")
        print("  Collecting INPUT features to attention components")
        print(f"  Cache directory: {self.cache_dir}")

    def register_hooks(self, unet, timestep: int):
        """
        Register hooks to collect INPUT features to attention components.

        Args:
            unet: UNet model
            timestep: Current denoising timestep
        """
        self.remove_hooks()  # Clear existing hooks

        def create_input_hook(layer_name: str, component_type: str):
            """Create hook to capture input features to a component."""
            def hook_fn(module, input_data, output_data):
                # Input data is what goes INTO the component (to_q, to_k, etc.)
                # This is the feature we need for AlphaEdit
                if isinstance(input_data, tuple):
                    input_feature = input_data[0]
                else:
                    input_feature = input_data

                # Store the input feature
                key = (timestep, layer_name, component_type)
                if key not in self.collected_features:
                    self.collected_features[key] = []

                # Detach and move to CPU to save memory
                self.collected_features[key].append(
                    input_feature.detach().cpu()
                )

            return hook_fn

        # Register hooks on cross-attention components
        hook_count = 0
        for name, module in unet.named_modules():
            if "attn2" in name:  # Cross-attention only
                # Find the attention module
                if hasattr(module, 'to_q'):
                    # Register hooks for each component
                    components = {
                        'q': module.to_q,
                        'k': module.to_k,
                        'v': module.to_v,
                        'out': module.to_out[0] if hasattr(module.to_out, '__getitem__') else module.to_out
                    }

                    for comp_name, comp_module in components.items():
                        if comp_module is not None:
                            hook = comp_module.register_forward_hook(
                                create_input_hook(name, comp_name)
                            )
                            self.hooks.append(hook)
                            hook_count += 1

        if hook_count > 0:
            print(f"  Registered {hook_count} hooks for timestep {timestep}")

        return hook_count > 0

    def collect_features_for_prompts(
        self,
        pipeline,
        prompts: List[str],
        timesteps: List[int]
    ):
        """
        Collect input features for all prompts and timesteps.

        Args:
            pipeline: Stable Diffusion pipeline
            prompts: List of original knowledge prompts
            timesteps: List of timesteps to collect features for
        """
        print(f"\nCollecting features for {len(prompts)} prompts at {len(timesteps)} timesteps")

        # Set to eval mode for clean features
        unet_training = pipeline.unet.training
        pipeline.unet.eval()

        try:
            with torch.no_grad():
                for timestep in tqdm(timesteps, desc="Processing timesteps"):
                    # Register hooks for this timestep
                    if not self.register_hooks(pipeline.unet, timestep):
                        print(f"Warning: No hooks registered for timestep {timestep}")
                        continue

                    # Process each prompt
                    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Timestep {timestep}", leave=False)):
                        # Encode prompt
                        text_inputs = pipeline.tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=pipeline.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = pipeline.text_encoder(
                            text_inputs.input_ids.to(self.device)
                        )[0]

                        # Generate random latents
                        latents = torch.randn(
                            (1, pipeline.unet.config.in_channels, 64, 64),
                            device=self.device,
                            dtype=pipeline.unet.dtype
                        )

                        # Add noise for this timestep
                        t = torch.tensor([timestep], device=self.device)
                        noise = torch.randn_like(latents)
                        noisy_latents = pipeline.scheduler.add_noise(latents, noise, t)

                        # Forward pass to collect features
                        _ = pipeline.unet(
                            noisy_latents,
                            t,
                            encoder_hidden_states=text_embeddings
                        ).sample

                    # Remove hooks after processing this timestep
                    self.remove_hooks()

                    # Process collected features into covariance matrix
                    self.compute_and_save_covariance(timestep, len(prompts))

                    # Clear features to save memory
                    self.clear_features_for_timestep(timestep)

        finally:
            # Restore training mode
            if unet_training:
                pipeline.unet.train()
            self.remove_hooks()

        print("✅ Feature collection completed")

    def compute_and_save_covariance(self, timestep: int, num_prompts: int):
        """
        Compute covariance matrix K₀K₀ᵀ for each layer and component.

        For each attention layer and component (q, k, v, out):
        - Collect all input features across prompts
        - Stack them into matrix K₀ of shape [dim, num_prompts]
        - Compute covariance K₀K₀ᵀ of shape [dim, dim]
        """
        print(f"\nComputing covariance matrices for timestep {timestep}")

        # Group features by layer and component
        layer_components = {}
        for (ts, layer_name, comp_type), features in self.collected_features.items():
            if ts != timestep:
                continue

            if layer_name not in layer_components:
                layer_components[layer_name] = {}

            # Stack features from all prompts
            # features is a list of tensors, each of shape [batch, seq_len, dim]
            # We need to extract and reshape appropriately
            feature_list = []
            for feat in features:
                # Flatten spatial dimensions and take mean across sequence
                if len(feat.shape) == 3:  # [batch, seq_len, dim]
                    feat = feat.mean(dim=1)  # Average across sequence
                elif len(feat.shape) == 4:  # [batch, h, w, dim]
                    feat = feat.reshape(feat.shape[0], -1, feat.shape[-1]).mean(dim=1)
                feature_list.append(feat.squeeze(0))  # Remove batch dimension

            # Stack into matrix K₀ of shape [dim, num_prompts]
            if feature_list:
                K0 = torch.stack(feature_list, dim=1)  # [dim, num_prompts]
                layer_components[layer_name][comp_type] = K0

        # Compute and save covariance matrices
        timestep_dir = self.cache_dir.parent / f"timestep_{timestep:04d}"
        timestep_dir.mkdir(parents=True, exist_ok=True)

        for layer_name, components in layer_components.items():
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            layer_dir = timestep_dir / f"module_{safe_layer_name}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            for comp_type, K0 in components.items():
                # Ensure K0 is [dim, num_prompts]
                if K0.shape[1] != num_prompts:
                    print(f"  Warning: Expected {num_prompts} prompts, got {K0.shape[1]} for {layer_name}/{comp_type}")

                # Compute covariance matrix K₀K₀ᵀ
                K0 = K0.float()  # Ensure float32 for stability
                cov_matrix = torch.mm(K0, K0.T)  # [dim, dim]

                # Save covariance matrix
                cov_file = layer_dir / f"{comp_type}_covariance.npy"
                np.save(cov_file, cov_matrix.cpu().numpy())

                print(f"  Saved {layer_name}/{comp_type}: K₀={K0.shape} → Cov={cov_matrix.shape}")

        print(f"✅ Covariance matrices saved for timestep {timestep}")

    def clear_features_for_timestep(self, timestep: int):
        """Clear collected features for a specific timestep to free memory."""
        keys_to_remove = [
            key for key in self.collected_features.keys()
            if key[0] == timestep
        ]
        for key in keys_to_remove:
            del self.collected_features[key]
        gc.collect()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def build_covariance_matrices_correct(
    config,
    accelerator,
    pipeline,
    prompts_file: str,
    timesteps: List[int],
    output_dir: str
):
    """
    Main function to build covariance matrices following AlphaEdit.

    Args:
        config: Configuration object
        accelerator: Accelerator for distributed training
        pipeline: Stable Diffusion pipeline
        prompts_file: Path to file containing original knowledge prompts
        timesteps: List of timesteps to process
        output_dir: Output directory for covariance matrices
    """
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Create feature collector
    collector = AlphaEditFeatureCollector(
        config=config,
        accelerator=accelerator,
        cache_dir=output_dir
    )

    # Collect features and compute covariance matrices
    collector.collect_features_for_prompts(
        pipeline=pipeline,
        prompts=prompts,
        timesteps=timesteps
    )

    print(f"\n✅ All covariance matrices saved to {output_dir}")