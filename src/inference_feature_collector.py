"""
Improved Feature Collection for NullBooth
Collects clean features using inference mode for better representation of original knowledge
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import torch.nn.functional as F


class InferenceFeatureCollector:
    """
    Collects features in inference mode to get clean representations of original knowledge.
    This avoids contamination from training dynamics.
    """

    def __init__(
        self,
        config,
        accelerator,
        cache_dir=None,
        inference_mode: str = "partial_denoise"  # "partial_denoise" or "full_denoise"
    ):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.inference_mode = inference_mode

        # Cache management
        self.cache_dir = Path(cache_dir) / "cache" if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature storage
        self.feature_cache = {}
        self.current_timestep = None
        self.hooks = []

        print(f"InferenceFeatureCollector initialized:")
        print(f"  Mode: {inference_mode}")
        print(f"  Cache directory: {self.cache_dir}")

    def collect_clean_features(
        self,
        pipeline,
        prompts: List[str],
        timesteps: List[int],
        timestep_to_alpha_mapping: Optional[Dict] = None
    ):
        """
        Collect clean features using the diffusion pipeline in inference mode.

        Args:
            pipeline: StableDiffusionPipeline instance
            prompts: List of original knowledge prompts
            timesteps: List of timesteps to collect features for
            timestep_to_alpha_mapping: Mapping for alpha naming scheme
        """
        print(f"Collecting clean features for {len(prompts)} prompts at {len(timesteps)} timesteps")

        # Set models to eval mode
        unet_training = pipeline.unet.training
        pipeline.unet.eval()

        # Register hooks for feature collection
        self._register_hooks(pipeline.unet)

        try:
            with torch.no_grad():  # Ensure we're in inference mode
                for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
                    self._collect_prompt_features(
                        pipeline=pipeline,
                        prompt=prompt,
                        prompt_idx=prompt_idx,
                        timesteps=timesteps,
                        timestep_to_alpha_mapping=timestep_to_alpha_mapping
                    )

                    # Save features periodically to manage memory
                    if (prompt_idx + 1) % 10 == 0:
                        self._save_and_clear_cache()

                # Save any remaining features
                self._save_and_clear_cache()

        finally:
            # Restore original training mode
            if unet_training:
                pipeline.unet.train()
            # Remove hooks
            self._remove_hooks()

        print(f"✅ Clean feature collection completed")

    def _collect_prompt_features(
        self,
        pipeline,
        prompt: str,
        prompt_idx: int,
        timesteps: List[int],
        timestep_to_alpha_mapping: Optional[Dict] = None
    ):
        """Collect features for a single prompt across all timesteps."""

        # Encode prompt once
        text_embeddings = self._encode_prompt(pipeline, prompt)

        # Generate base latents
        latents = self._generate_base_latents(pipeline)

        for timestep in timesteps:
            self.current_timestep = timestep

            if self.inference_mode == "partial_denoise":
                # Partial denoising: Start from noise and denoise to timestep
                features = self._collect_partial_denoise_features(
                    pipeline, latents, text_embeddings, timestep
                )
            elif self.inference_mode == "full_denoise":
                # Full denoising: Generate clean image then add noise to timestep
                features = self._collect_full_denoise_features(
                    pipeline, latents, text_embeddings, timestep, prompt
                )
            else:
                raise ValueError(f"Unknown inference mode: {self.inference_mode}")

            # Store features with appropriate naming
            if timestep_to_alpha_mapping:
                timestep_key = timestep_to_alpha_mapping.get(f"timestep_{timestep:04d}", f"timestep_{timestep:04d}")
            else:
                timestep_key = f"timestep_{timestep:04d}"

            self._store_features(prompt_idx, timestep_key, features)

    def _encode_prompt(self, pipeline, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return text_embeddings

    def _generate_base_latents(self, pipeline) -> torch.Tensor:
        """Generate base random latents."""
        shape = (
            1,
            pipeline.unet.config.in_channels,
            self.config.resolution // 8,
            self.config.resolution // 8
        )
        latents = torch.randn(shape, device=self.device, dtype=pipeline.unet.dtype)
        return latents

    def _collect_partial_denoise_features(
        self,
        pipeline,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_timestep: int
    ) -> Dict:
        """
        Collect features by denoising from maximum noise to target timestep.
        Performs proper step-by-step denoising without simplification.
        """
        # Handle the case where target_timestep might be a float (for alpha-based sampling)
        if isinstance(target_timestep, float):
            target_timestep_int = int(target_timestep)
        else:
            target_timestep_int = target_timestep

        # Set up scheduler for step-by-step denoising
        pipeline.scheduler.set_timesteps(pipeline.scheduler.config.num_train_timesteps, device=self.device)

        # Get all timesteps from max to target
        all_timesteps = pipeline.scheduler.timesteps.cpu().numpy().tolist()

        # Find the index of our target timestep (or closest)
        target_idx = len(all_timesteps) - 1  # Default to last (smallest noise)
        for idx, ts in enumerate(all_timesteps):
            if ts <= target_timestep_int:
                target_idx = idx
                break

        # Get timesteps from start (max noise) to target
        timesteps_to_denoise = all_timesteps[:target_idx + 1]

        if len(timesteps_to_denoise) == 0:
            # Target is at max noise, no denoising needed
            timesteps_to_denoise = [all_timesteps[0]]

        # Start from pure noise
        current_latents = torch.randn_like(latents)

        # Perform step-by-step denoising
        for i, t in enumerate(timesteps_to_denoise):
            # Prepare timestep tensor
            timestep_tensor = torch.tensor([t], device=self.device)

            # Scale model input if needed
            latent_model_input = pipeline.scheduler.scale_model_input(current_latents, timestep_tensor)

            # Predict noise residual
            noise_pred = pipeline.unet(
                latent_model_input,
                timestep_tensor,
                encoder_hidden_states=text_embeddings
            ).sample

            # Compute the previous noisy sample (denoising step)
            if i < len(timesteps_to_denoise) - 1:
                # Not the last step, continue denoising
                current_latents = pipeline.scheduler.step(
                    noise_pred, t, current_latents
                ).prev_sample
            # On the last step, we just run the UNet to collect features
            # without updating latents

        # Features are collected via hooks during the last UNet pass
        return self.feature_cache.get(f"timestep_{target_timestep_int:04d}", {})

    def _collect_full_denoise_features(
        self,
        pipeline,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_timestep: int,
        prompt: str
    ) -> Dict:
        """
        Collect features by first generating a clean image, then adding noise to target timestep.
        This represents how the model encodes a clean concept at different noise levels.
        """
        # Handle the case where target_timestep might be a float
        if isinstance(target_timestep, float):
            target_timestep_int = int(target_timestep)
        else:
            target_timestep_int = target_timestep

        # Step 1: Generate clean image using full denoising pipeline
        with torch.no_grad():
            # Set up scheduler for full denoising
            pipeline.scheduler.set_timesteps(50, device=self.device)  # Use 50 steps for quality

            # Start from random noise
            clean_latents = torch.randn_like(latents)

            # Full denoising loop
            for t in tqdm(pipeline.scheduler.timesteps, desc="Generating clean image", leave=False):
                # Scale model input
                latent_model_input = pipeline.scheduler.scale_model_input(clean_latents, t)

                # Predict noise residual
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

                # Compute previous noisy sample
                clean_latents = pipeline.scheduler.step(
                    noise_pred, t, clean_latents
                ).prev_sample

        # Step 2: Add noise to reach target timestep
        noise = torch.randn_like(clean_latents)
        t = torch.tensor([target_timestep_int], device=self.device)

        # Add noise using the scheduler's add_noise method
        noisy_latents = pipeline.scheduler.add_noise(clean_latents, noise, t)

        # Step 3: Run UNet at target timestep to collect features
        latent_model_input = pipeline.scheduler.scale_model_input(noisy_latents, t)

        _ = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample

        # Features are collected via hooks
        return self.feature_cache.get(f"timestep_{target_timestep_int:04d}", {})

    def _register_hooks(self, unet):
        """Register forward hooks to collect features from cross-attention layers."""
        def make_hook(layer_name, feature_type):
            def hook(module, input, output):
                if self.current_timestep is not None:
                    key = f"timestep_{self.current_timestep:04d}"
                    if key not in self.feature_cache:
                        self.feature_cache[key] = {}
                    if layer_name not in self.feature_cache[key]:
                        self.feature_cache[key][layer_name] = {}

                    # Store the feature
                    if feature_type == 'q':
                        self.feature_cache[key][layer_name]['q'] = input[0].detach().cpu()
                    elif feature_type in ['k', 'v']:
                        self.feature_cache[key][layer_name][feature_type] = output.detach().cpu()
                    elif feature_type == 'out':
                        self.feature_cache[key][layer_name]['out'] = output.detach().cpu()

            return hook

        # Register hooks on cross-attention layers
        for name, module in unet.named_modules():
            if "attn2" in name:  # Cross-attention layers
                if hasattr(module, 'to_q'):
                    hook = module.to_q.register_forward_hook(make_hook(name, 'q'))
                    self.hooks.append(hook)
                if hasattr(module, 'to_k'):
                    hook = module.to_k.register_forward_hook(make_hook(name, 'k'))
                    self.hooks.append(hook)
                if hasattr(module, 'to_v'):
                    hook = module.to_v.register_forward_hook(make_hook(name, 'v'))
                    self.hooks.append(hook)
                if hasattr(module, 'to_out'):
                    if hasattr(module.to_out, '__getitem__'):
                        hook = module.to_out[0].register_forward_hook(make_hook(name, 'out'))
                    else:
                        hook = module.to_out.register_forward_hook(make_hook(name, 'out'))
                    self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _store_features(self, prompt_idx: int, timestep_key: str, features: Dict):
        """Store collected features to disk."""
        if not features:
            return

        prompt_dir = self.cache_dir / f"prompt_{prompt_idx:04d}"
        prompt_dir.mkdir(exist_ok=True)

        # Save features as compressed numpy file
        feature_file = prompt_dir / f"{timestep_key}_features.npz"

        # Convert features to numpy arrays
        save_dict = {}
        for layer_name, layer_features in features.items():
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            for feature_type, feature_tensor in layer_features.items():
                key = f"{safe_layer_name}_{feature_type}"
                save_dict[key] = feature_tensor.numpy() if hasattr(feature_tensor, 'numpy') else feature_tensor

        np.savez_compressed(feature_file, **save_dict)

    def _save_and_clear_cache(self):
        """Save cached features to disk and clear memory."""
        # Features are saved immediately in _store_features, so just clear cache
        self.feature_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()