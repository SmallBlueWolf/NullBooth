"""
Corrected NullBooth Optimizer for AlphaEdit Implementation
Applies projection to weight UPDATES (not gradients) following the paper exactly
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import numpy as np


class AlphaEditOptimizer:
    """
    Optimizer wrapper that applies AlphaEdit null-space projection to weight updates.

    Key insight from AlphaEdit paper:
    - We project the weight UPDATE Δ, not the gradient
    - The update is: W_new = W_old + Δ·P
    - Where Δ = -lr * gradient (for SGD) or the actual update from Adam
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cov_manager,
        unet: nn.Module,
        enable_projection: bool = True,
        debug: bool = False
    ):
        """
        Args:
            optimizer: Base optimizer (e.g., AdamW)
            cov_manager: CovarianceMatrixManager instance
            unet: UNet model to track attention layers
            enable_projection: Whether to apply projection
            debug: Enable debug output
        """
        self.base_optimizer = optimizer
        self.cov_manager = cov_manager
        self.unet = unet
        self.enable_projection = enable_projection
        self.debug = debug
        self.current_timestep = None

        # Map parameters to layer info
        self.param_info = self._build_param_mapping()

        # Store original parameters to compute actual updates
        self.original_params = {}
        self._store_original_params()

        print(f"AlphaEditOptimizer initialized:")
        print(f"  Tracked cross-attention parameters: {len(self.param_info)}")
        print(f"  Projection enabled: {enable_projection}")

    def _build_param_mapping(self) -> Dict[int, Tuple[str, str, str]]:
        """
        Build mapping from parameter ID to (layer_name, component_type, param_type).

        Returns:
            Dict mapping parameter ID to (layer_name, component_type, param_type)
            where component_type is 'q', 'k', 'v', or 'out'
            and param_type is 'weight' or 'bias'
        """
        param_info = {}

        for name, module in self.unet.named_modules():
            if "attn2" not in name:  # Only cross-attention
                continue

            # Check if this is an attention module with components
            if hasattr(module, 'to_q'):
                components = {
                    'q': module.to_q,
                    'k': module.to_k,
                    'v': module.to_v,
                    'out': module.to_out[0] if hasattr(module.to_out, '__getitem__') else module.to_out
                }

                for comp_type, comp_module in components.items():
                    if comp_module is None:
                        continue

                    # Track both weight and bias
                    if hasattr(comp_module, 'weight') and comp_module.weight is not None:
                        param_id = id(comp_module.weight)
                        param_info[param_id] = (name, comp_type, 'weight')

                    if hasattr(comp_module, 'bias') and comp_module.bias is not None:
                        param_id = id(comp_module.bias)
                        param_info[param_id] = (name, comp_type, 'bias')

        return param_info

    def _store_original_params(self):
        """Store original parameters before optimization step."""
        for param_group in self.base_optimizer.param_groups:
            for param in param_group['params']:
                param_id = id(param)
                if param_id in self.param_info:
                    self.original_params[param_id] = param.data.clone()

    def set_current_timestep(self, timestep: int):
        """Set the current denoising timestep for projection matrix selection."""
        self.current_timestep = timestep
        if self.debug:
            closest = self.cov_manager.find_closest_timestep(timestep)
            print(f"Timestep set: {timestep} -> closest: {closest}")

    def step(self, closure=None):
        """
        Perform optimization step with AlphaEdit projection.

        Process:
        1. Store original parameters
        2. Perform standard optimization step
        3. Compute actual updates Δ = W_new - W_old
        4. Apply projection: Δ_projected = Δ @ P
        5. Update parameters: W_final = W_old + Δ_projected
        """
        if not self.enable_projection or self.current_timestep is None:
            # Standard optimization without projection
            return self.base_optimizer.step(closure)

        # Store original parameters
        self._store_original_params()

        # Perform standard optimization step
        loss = self.base_optimizer.step(closure)

        # Apply projection to the updates
        self._apply_update_projection()

        return loss

    def _apply_update_projection(self):
        """
        Apply null-space projection to weight updates.
        This is the core of AlphaEdit: Δ_projected = Δ @ P
        """
        projection_count = 0

        for param_group in self.base_optimizer.param_groups:
            for param in param_group['params']:
                param_id = id(param)

                # Check if this parameter needs projection
                if param_id not in self.param_info:
                    continue

                layer_name, comp_type, param_type = self.param_info[param_id]

                # Get projection matrix for this component
                P = self.cov_manager.get_projection_matrix(
                    self.current_timestep, layer_name, comp_type
                )

                if P is None:
                    continue

                # Compute actual update: Δ = W_new - W_old
                W_old = self.original_params[param_id]
                W_new = param.data
                Delta = W_new - W_old

                # Apply projection based on parameter type
                if param_type == 'weight':
                    # Weight matrix update
                    if len(Delta.shape) == 2:
                        # Delta shape: [out_features, in_features]
                        # P shape: [in_features, in_features]

                        # Apply projection: Δ_projected = Δ @ P
                        if Delta.shape[1] == P.shape[0]:
                            # Standard case: project input dimension
                            Delta_projected = torch.mm(Delta, P)
                        elif Delta.shape[0] == P.shape[0]:
                            # Alternative: project output dimension
                            Delta_projected = torch.mm(P, Delta)
                        else:
                            if self.debug:
                                print(f"  ⚠️ Dimension mismatch: Δ={Delta.shape} vs P={P.shape}")
                            continue
                    else:
                        continue

                elif param_type == 'bias':
                    # Bias vector update
                    if len(Delta.shape) == 1 and Delta.shape[0] == P.shape[0]:
                        # Apply projection to bias: Δ_projected = P @ Δ
                        Delta_projected = torch.mv(P, Delta)
                    else:
                        continue

                else:
                    continue

                # Compute statistics
                if self.debug and projection_count < 5:
                    original_norm = torch.norm(Delta).item()
                    projected_norm = torch.norm(Delta_projected).item()
                    reduction = 1 - (projected_norm / original_norm if original_norm > 0 else 0)

                    print(f"  ✓ Projected {layer_name}/{comp_type}/{param_type}")
                    print(f"    Update norm: {original_norm:.6f} → {projected_norm:.6f}")
                    print(f"    Norm reduction: {100*reduction:.1f}%")

                # Apply projected update: W_final = W_old + Δ_projected
                param.data = W_old + Delta_projected
                projection_count += 1

        if projection_count > 0 and self.debug:
            print(f"  Total projections applied: {projection_count}")

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.base_optimizer.zero_grad(set_to_none)

    @property
    def param_groups(self):
        """Get parameter groups."""
        return self.base_optimizer.param_groups

    def state_dict(self):
        """Get state dict."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.base_optimizer.load_state_dict(state_dict)


class AlphaEditProjectionManager:
    """
    Manages the computation of projection matrices from covariance matrices
    following the exact AlphaEdit algorithm.
    """

    @staticmethod
    def compute_projection_matrix(
        cov_matrix: torch.Tensor,
        threshold: float = 1e-3,
        threshold_percent: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute projection matrix P from covariance matrix K₀K₀ᵀ.

        Following AlphaEdit:
        1. SVD decomposition: K₀K₀ᵀ = UΣUᵀ
        2. Find null space: eigenvectors with small eigenvalues
        3. Construct projection: P = ŮŮᵀ where Ů contains null space eigenvectors

        Args:
            cov_matrix: Covariance matrix K₀K₀ᵀ of shape [dim, dim]
            threshold: Absolute threshold for small eigenvalues
            threshold_percent: Percentile threshold (overrides absolute if set)

        Returns:
            Projection matrix P of shape [dim, dim]
        """
        # Perform SVD
        U, S, _ = torch.linalg.svd(cov_matrix, full_matrices=False)

        # Determine threshold for null space
        if threshold_percent is not None:
            # Use percentile-based threshold
            threshold = torch.quantile(S, threshold_percent / 100.0)
            threshold_mode = f"percentile({threshold_percent}%)"
        else:
            # Use absolute threshold
            threshold_mode = f"absolute({threshold:.2e})"

        # Find eigenvectors corresponding to small eigenvalues (null space)
        small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]

        if len(small_singular_indices) > 0:
            # Construct projection matrix P = ŮŮᵀ
            U_null = U[:, small_singular_indices]
            P = torch.mm(U_null, U_null.T)

            # Ensure symmetry (for numerical stability)
            P = 0.5 * (P + P.T)

            print(f"  Projection matrix computed:")
            print(f"    Null space dimension: {len(small_singular_indices)}/{len(S)}")
            print(f"    Threshold: {threshold_mode}")
            print(f"    Eigenvalue range: [{S.min():.2e}, {S.max():.2e}]")
        else:
            # No null space found - use zero projection
            P = torch.zeros_like(cov_matrix)
            print(f"  No null space found (threshold: {threshold_mode})")

        return P