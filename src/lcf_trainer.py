"""
Latent Consistency Fine-tuning (LCF) implementation for LCM models.
Based on Algorithm 4 from the LCM paper (Luo et al., 2023).

This implements non-distillation fine-tuning for pre-trained LCMs,
which is different from Latent Consistency Distillation (LCD).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LCFLoss:
    """
    Latent Consistency Fine-tuning Loss.
    Enforces self-consistency between different timesteps without requiring a teacher model.
    """

    def __init__(
        self,
        consistency_weight: float = 1.0,
        skipping_step: int = 20,
        huber_c: float = 0.001,
    ):
        """
        Args:
            consistency_weight: Weight for consistency loss
            skipping_step: Number of steps to skip (k in the paper)
            huber_c: Huber loss parameter for robustness
        """
        self.consistency_weight = consistency_weight
        self.skipping_step = skipping_step
        self.huber_c = huber_c

    def compute_lcf_loss(
        self,
        model,
        z_0: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_t_prev: torch.Tensor,
        w: float = 8.0,
    ) -> torch.Tensor:
        """
        Compute LCF loss based on self-consistency.

        Key difference from LCD:
        - No ODE solver needed
        - No teacher model needed
        - Direct consistency between f(z_{t_n+k}) and f(z_{t_n})
        """

        # Add noise at two different timesteps (t_n+k and t_n)
        # Using the SAME noise epsilon for both (key for consistency)
        z_t_nk = alpha_prod_t * z_0 + sigma_t * noise  # z at t_{n+k}
        z_t_n = alpha_prod_t_prev * z_0 + sigma_t_prev * noise  # z at t_n

        # Get consistency function outputs at both timesteps
        # The model should predict the same z_0 from both points
        with torch.no_grad():
            # Target model (EMA) prediction at t_n
            target_pred = model(z_t_n, timesteps - self.skipping_step, text_embeddings, w)
            # Convert to predicted z_0
            target_z0 = self.pred_to_z0(z_t_n, target_pred, alpha_prod_t_prev, sigma_t_prev)

        # Online model prediction at t_{n+k}
        online_pred = model(z_t_nk, timesteps, text_embeddings, w)
        # Convert to predicted z_0
        online_z0 = self.pred_to_z0(z_t_nk, online_pred, alpha_prod_t, sigma_t)

        # Consistency loss: both should predict the same z_0
        # Using Huber loss for robustness
        consistency_loss = self.huber_loss(online_z0, target_z0.detach())

        return consistency_loss * self.consistency_weight

    def pred_to_z0(self, z_t, noise_pred, alpha, sigma):
        """Convert noise prediction to z_0 prediction."""
        return (z_t - sigma * noise_pred) / alpha

    def huber_loss(self, pred, target):
        """Pseudo-Huber loss for robustness."""
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + self.huber_c**2) - self.huber_c)


def get_lcf_config():
    """Get recommended LCF configuration for fine-tuning LCMs."""
    return {
        # LCF specific parameters
        "lcf_mode": True,  # Enable LCF mode
        "consistency_weight": 1.0,
        "skipping_step": 20,  # k=20 as recommended
        "huber_c": 0.001,

        # Training parameters optimized for LCF
        "learning_rate": 1e-6,  # Much lower than LCD!
        "max_train_steps": 50,  # Very few steps needed
        "gradient_accumulation_steps": 1,

        # No teacher model needed
        "use_teacher_model": False,
        "use_ode_solver": False,

        # CFG scale range
        "w_min": 3.0,
        "w_max": 15.0,

        # EMA for target model
        "ema_decay": 0.95,  # Faster EMA update for fine-tuning

        # Important: Don't change the noise schedule!
        "preserve_noise_schedule": True,
    }


class LCFTrainer:
    """
    Trainer for Latent Consistency Fine-tuning.
    This is specifically for fine-tuning pre-trained LCMs on custom datasets.
    """

    def __init__(
        self,
        model,
        ema_model,
        config: Dict,
        nullbooth_wrapper=None,
    ):
        self.model = model
        self.ema_model = ema_model
        self.config = config
        self.nullbooth_wrapper = nullbooth_wrapper

        # Initialize LCF loss
        self.lcf_loss_fn = LCFLoss(
            consistency_weight=config.get("consistency_weight", 1.0),
            skipping_step=config.get("skipping_step", 20),
            huber_c=config.get("huber_c", 0.001),
        )

        self.ema_decay = config.get("ema_decay", 0.95)

        logger.info("Initialized LCF Trainer (non-distillation mode)")
        logger.info(f"  Skipping step k = {config.get('skipping_step', 20)}")
        logger.info(f"  EMA decay = {self.ema_decay}")
        logger.info("  No teacher model required")

    def training_step(self, batch):
        """
        LCF training step.
        Key differences from standard training:
        1. No teacher model needed
        2. Use same noise for two timesteps
        3. Enforce consistency between predictions
        """

        z_0 = batch["latents"]
        text_embeddings = batch["text_embeddings"]

        # Sample timestep n
        n = torch.randint(
            1,
            self.model.num_train_timesteps - self.lcf_loss_fn.skipping_step,
            (z_0.shape[0],),
            device=z_0.device
        )

        # Get timesteps t_n and t_{n+k}
        timesteps_nk = n + self.lcf_loss_fn.skipping_step
        timesteps_n = n

        # Sample CFG scale
        w = torch.rand(1).item() * (self.config["w_max"] - self.config["w_min"]) + self.config["w_min"]

        # Sample noise (same for both timesteps - critical!)
        noise = torch.randn_like(z_0)

        # Get alpha and sigma values
        alpha_prod_t = self.model.alphas_cumprod[timesteps_nk]
        alpha_prod_t_prev = self.model.alphas_cumprod[timesteps_n]
        sigma_t = ((1 - alpha_prod_t) ** 0.5)
        sigma_t_prev = ((1 - alpha_prod_t_prev) ** 0.5)

        # Reshape for broadcasting
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        sigma_t_prev = sigma_t_prev.view(-1, 1, 1, 1)

        # Compute LCF loss
        loss = self.lcf_loss_fn.compute_lcf_loss(
            self.ema_model,  # Use EMA model as target
            z_0,
            text_embeddings,
            timesteps_nk,
            noise,
            alpha_prod_t,
            alpha_prod_t_prev,
            sigma_t,
            sigma_t_prev,
            w,
        )

        # Apply NullBooth projection if enabled
        if self.nullbooth_wrapper is not None:
            # LCF is compatible with NullBooth
            # The null-space projection preserves consistency
            loss = self.nullbooth_wrapper.apply_projection(loss)

        return loss

    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, online_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    online_param.data, alpha=1 - self.ema_decay
                )