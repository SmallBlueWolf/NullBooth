"""
NullBooth Training Implementation
Implementation of AlphaEdit null-space projection for DreamBooth training.
Based on: ALPHAEDIT: NULL-SPACE CONSTRAINED KNOWLEDGE EDITING FOR LANGUAGE MODELS
"""

import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import json
import hashlib
from diffusers.models.attention_processor import Attention
from .correct_optimizer import AlphaEditOptimizer


class CovarianceMatrixManager:
    """
    Manages covariance matrices for NullBooth training.
    Loads precomputed covariance matrices and computes projection matrices.
    """

    def __init__(
        self,
        cov_matrices_dir: Union[str, Path],
        device: torch.device,
        nullspace_threshold: float = 2e-2,
        threshold_percent: float = None,
        cache_size: int = 50,
        use_interpolation: bool = True,
        debug: bool = False
    ):
        self.cov_matrices_dir = Path(cov_matrices_dir)
        self.device = device
        self.nullspace_threshold = nullspace_threshold
        self.threshold_percent = threshold_percent  # New parameter with higher priority
        self.cache_size = cache_size
        self.use_interpolation = use_interpolation
        self.debug = debug

        # Cache for projection matrices
        self.projection_cache = {}
        self.access_count = {}

        # Validate directory exists
        if not self.cov_matrices_dir.exists():
            raise FileNotFoundError(f"Covariance matrices directory not found: {self.cov_matrices_dir}")

        # Load available timesteps
        self._load_available_timesteps()

        print(f"CovarianceMatrixManager initialized:")
        print(f"  Directory: {self.cov_matrices_dir}")
        print(f"  Available timesteps: {len(self.available_timesteps)}")
        print(f"  Nullspace threshold: {self.nullspace_threshold}")
        print(f"  Use interpolation: {self.use_interpolation}")

    def _load_available_timesteps(self):
        """Discover available timesteps from directory structure.

        NEW: Adapted for current file structure where covariance matrices are organized as:
        cov_matrices_dir/covariance_matrices/module_layer_name/alpha_XX_YYYY_feature_cov.npy

        Supports both timestep_XXXX and alpha_XX_YYYY naming conventions.
        """
        self.available_timesteps = []
        self.timestep_to_key = {}
        self.alpha_mappings = {}  # Store alpha key mappings
        self.timestep_files = {}  # Store actual file locations

        # Check if we have the new structure: cov_matrices_dir/covariance_matrices/
        covariance_dir = self.cov_matrices_dir / "covariance_matrices"

        if covariance_dir.exists():
            # NEW STRUCTURE: Scan layer directories for timestep files
            print(f"  Detected new structure: scanning layer directories...")

            for layer_dir in covariance_dir.iterdir():
                if not layer_dir.is_dir():
                    continue

                # Scan files in this layer directory to find available timesteps
                for file_path in layer_dir.glob("*_cov.npy"):
                    filename = file_path.stem  # Remove .npy extension

                    # Parse filename: alpha_XX_YYYY_feature or timestep_XXXX_feature
                    if filename.startswith("alpha_"):
                        # Format: alpha_XX_YYYY_feature_cov
                        parts = filename.split("_")
                        if len(parts) >= 4:  # alpha, XX, YYYY, feature
                            timestep_key = f"{parts[0]}_{parts[1]}_{parts[2]}"  # alpha_XX_YYYY

                            # Extract actual timestep value from YYYY
                            try:
                                actual_timestep = int(parts[2])

                                # Add to available timesteps if not already present
                                if actual_timestep not in self.available_timesteps:
                                    self.available_timesteps.append(actual_timestep)
                                    self.timestep_to_key[actual_timestep] = timestep_key

                                    # Store alpha mapping
                                    step_idx = int(parts[1])
                                    self.alpha_mappings[timestep_key] = (step_idx, actual_timestep)

                            except (ValueError, IndexError):
                                continue

                    elif filename.startswith("timestep_"):
                        # Format: timestep_XXXX_feature_cov
                        parts = filename.split("_")
                        if len(parts) >= 3:  # timestep, XXXX, feature
                            try:
                                timestep_num = int(parts[1])
                                timestep_key = f"timestep_{timestep_num:04d}"

                                if timestep_num not in self.available_timesteps:
                                    self.available_timesteps.append(timestep_num)
                                    self.timestep_to_key[timestep_num] = timestep_key
                            except (ValueError, IndexError):
                                continue

                # Break after first layer - we just need to know which timesteps exist
                if self.available_timesteps:
                    break

        else:
            # OLD STRUCTURE: Scan timestep directories
            print(f"  Using old structure: scanning timestep directories...")

            for timestep_dir in self.cov_matrices_dir.iterdir():
                if not timestep_dir.is_dir():
                    continue

                dir_name = timestep_dir.name

                if dir_name.startswith("timestep_"):
                    # Standard timestep naming
                    try:
                        timestep_num = int(dir_name.split("_")[1])
                        self.available_timesteps.append(timestep_num)
                        self.timestep_to_key[timestep_num] = dir_name
                    except (ValueError, IndexError):
                        continue

                elif dir_name.startswith("alpha_"):
                    # Alpha naming: alpha_XX_YYYY directory
                    try:
                        parts = dir_name.split("_")
                        if len(parts) >= 3:
                            step_idx = int(parts[1])
                            actual_timestep = int(parts[2])

                            self.available_timesteps.append(actual_timestep)
                            self.timestep_to_key[actual_timestep] = dir_name
                            self.alpha_mappings[dir_name] = (step_idx, actual_timestep)

                            if self.debug:
                                print(f"  Alpha dir: {dir_name} -> step {step_idx} -> timestep {actual_timestep}")

                    except (ValueError, IndexError) as e:
                        if self.debug:
                            print(f"Failed to parse alpha directory {dir_name}: {e}")
                        continue

        self.available_timesteps.sort(reverse=True)  # Sort in descending order (high noise to low)

        if len(self.available_timesteps) > 0:
            print(f"  Loaded {len(self.available_timesteps)} timesteps")
            if self.alpha_mappings:
                print(f"  Including {len(self.alpha_mappings)} alpha-named timesteps")
            print(f"  Timestep range: {max(self.available_timesteps)} to {min(self.available_timesteps)}")
        else:
            print(f"  WARNING: No timesteps found in {self.cov_matrices_dir}")

    def find_closest_timestep(self, target_timestep: int) -> int:
        """Find the closest available timestep to the target."""
        if not self.available_timesteps:
            raise ValueError("No available timesteps found")

        # If exact match exists, use it
        if target_timestep in self.available_timesteps:
            return target_timestep

        # Find closest timestep
        closest_timestep = min(
            self.available_timesteps,
            key=lambda x: abs(x - target_timestep)
        )
        return closest_timestep

    def get_interpolated_projection_matrix(
        self,
        timestep: int,
        layer_name: str,
        feature_type: str
    ) -> Optional[torch.Tensor]:
        """
        Get projection matrix with optional interpolation between neighboring timesteps.
        This provides smoother transitions for timesteps not in the cache.
        """
        # Check if exact timestep exists
        if timestep in self.available_timesteps:
            return self.get_projection_matrix(timestep, layer_name, feature_type)

        # Find neighboring timesteps for interpolation
        lower_ts = [t for t in self.available_timesteps if t < timestep]
        upper_ts = [t for t in self.available_timesteps if t > timestep]

        if not lower_ts or not upper_ts:
            # Can't interpolate, use closest
            return self.get_projection_matrix(timestep, layer_name, feature_type)

        # Get nearest neighbors
        lower_timestep = max(lower_ts)
        upper_timestep = min(upper_ts)

        # Get projection matrices for neighbors
        P_lower = self._load_and_compute_projection(lower_timestep, layer_name, feature_type)
        P_upper = self._load_and_compute_projection(upper_timestep, layer_name, feature_type)

        if P_lower is None or P_upper is None:
            # Can't interpolate without both matrices
            return self.get_projection_matrix(timestep, layer_name, feature_type)

        # Calculate interpolation weight
        alpha = (timestep - lower_timestep) / (upper_timestep - lower_timestep)

        # Interpolate projection matrices
        # Note: For projection matrices, we should interpolate in the tangent space
        # For simplicity, we use linear interpolation here
        P_interp = (1 - alpha) * P_lower + alpha * P_upper

        # Ensure the interpolated matrix maintains projection properties
        # P should be symmetric and idempotent (P² = P)
        # We can enforce symmetry
        P_interp = 0.5 * (P_interp + P_interp.T)

        if self.debug:
            print(f"Interpolated projection for timestep {timestep}:")
            print(f"  Between {lower_timestep} and {upper_timestep}")
            print(f"  Weight alpha = {alpha:.3f}")

        return P_interp

    def get_projection_matrix(
        self,
        timestep: int,
        layer_name: str,
        feature_type: str
    ) -> Optional[torch.Tensor]:
        """
        Get or compute projection matrix P = ŮŮᵀ for given timestep, layer, and feature type.

        Args:
            timestep: Target timestep
            layer_name: Name of the attention layer
            feature_type: Type of feature ('q', 'k', 'v', 'out')

        Returns:
            Projection matrix or None if not available
        """
        # If interpolation is enabled and timestep is not exact match, use interpolation
        if self.use_interpolation and timestep not in self.available_timesteps:
            return self.get_interpolated_projection_matrix(timestep, layer_name, feature_type)

        # Find closest available timestep
        closest_timestep = self.find_closest_timestep(timestep)

        # Create cache key
        cache_key = f"{closest_timestep}_{layer_name}_{feature_type}"

        # Check cache
        if cache_key in self.projection_cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.projection_cache[cache_key]

        # Load and compute projection matrix
        projection_matrix = self._load_and_compute_projection(
            closest_timestep, layer_name, feature_type
        )

        if projection_matrix is not None:
            # Cache management (LRU)
            cache_size = self.cache_size if self.cache_size is not None else 50
            if len(self.projection_cache) >= cache_size:
                # Remove least recently used item
                lru_key = min(self.access_count.keys(), key=self.access_count.get)
                del self.projection_cache[lru_key]
                del self.access_count[lru_key]

            # Add to cache
            self.projection_cache[cache_key] = projection_matrix
            self.access_count[cache_key] = 1

        return projection_matrix

    def _load_and_compute_projection(
        self,
        timestep: int,
        layer_name: str,
        feature_type: str
    ) -> Optional[torch.Tensor]:
        """
        Load covariance matrix and compute projection matrix following AlphaEdit method.

        NEW: Adapted for current file structure:
        - Old: cov_dir/timestep_key/module_layer_name/feature_covariance.npy
        - New: cov_dir/covariance_matrices/module_layer_name/timestep_key_feature_cov.npy
        """
        try:
            # Construct file path
            timestep_key = self.timestep_to_key[timestep]
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")

            # Try with module_ prefix first (for build_cov_parallel output)
            module_layer_name = f"module_{safe_layer_name}"

            # Check if we have the new structure
            covariance_dir = self.cov_matrices_dir / "covariance_matrices"

            if covariance_dir.exists():
                # NEW STRUCTURE: cov_dir/covariance_matrices/module_layer_name/timestep_key_feature_cov.npy
                cov_file = covariance_dir / module_layer_name / f"{timestep_key}_{feature_type}_cov.npy"

                # If not found with module_ prefix, try without
                if not cov_file.exists():
                    cov_file = covariance_dir / safe_layer_name / f"{timestep_key}_{feature_type}_cov.npy"

            else:
                # OLD STRUCTURE: cov_dir/timestep_key/module_layer_name/feature_covariance.npy
                cov_file = (
                    self.cov_matrices_dir /
                    timestep_key /
                    module_layer_name /
                    f"{feature_type}_covariance.npy"
                )

                # If not found, try without module_ prefix
                if not cov_file.exists():
                    cov_file = (
                        self.cov_matrices_dir /
                        timestep_key /
                        safe_layer_name /
                        f"{feature_type}_covariance.npy"
                    )

            if not cov_file.exists():
                if self.debug:
                    print(f"Warning: Covariance file not found: {cov_file}")
                return None

            # Load covariance matrix K₀K₀ᵀ
            cov_matrix = np.load(cov_file)
            cov_tensor = torch.from_numpy(cov_matrix).to(
                device=self.device, dtype=torch.float32
            )

            # Apply AlphaEdit projection computation
            # Step 1: SVD decomposition of K₀K₀ᵀ
            U, S, _ = torch.linalg.svd(cov_tensor, full_matrices=False)

            # Step 2: Find eigenvectors corresponding to small eigenvalues (null space)
            # Priority: threshold_percent > nullspace_threshold
            if self.threshold_percent is not None:
                # Use percentile-based threshold (more robust)
                threshold = torch.quantile(S, self.threshold_percent / 100.0)
                threshold_type = f"percentile({self.threshold_percent}%)"
            else:
                # Use absolute threshold (original method)
                threshold = float(self.nullspace_threshold)  # Ensure it's a float
                threshold_type = f"absolute({threshold:.2e})"

            small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]

            if len(small_singular_indices) > 0:
                # Step 3: Construct projection matrix P = ŮŮᵀ
                # where Ů contains eigenvectors corresponding to small eigenvalues
                U_null = U[:, small_singular_indices]
                projection_matrix = torch.mm(U_null, U_null.T)

                if self.debug:
                    print(f"Computed projection matrix for {timestep}_{layer_name}_{feature_type}: "
                          f"shape={projection_matrix.shape}, null_dim={len(small_singular_indices)}, "
                          f"threshold={threshold_type}, singular_range=[{S.min():.2e}, {S.max():.2e}]")
            else:
                # No null space found, use zero projection (no constraint)
                projection_matrix = torch.zeros_like(cov_tensor)
                if self.debug:
                    print(f"No null space found for {timestep}_{layer_name}_{feature_type}, "
                          f"using zero projection, threshold={threshold_type}")

            return projection_matrix

        except Exception as e:
            print(f"Error computing projection matrix for {timestep}_{layer_name}_{feature_type}: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return None


class NullBoothTrainer:
    """
    Main NullBooth training coordinator.
    Integrates covariance matrix management and optimizer-based weight update projection.
    Following exact AlphaEdit implementation.
    """

    def __init__(self, config, unet, device, optimizer=None):
        self.config = config
        self.unet = unet
        self.device = device
        self.enabled = False
        self.cov_manager = None
        self.optimizer_wrapper = None

        # Check if NullBooth is enabled
        if hasattr(config, 'nullbooth') and getattr(config.nullbooth, 'enable', False):
            self.enabled = True
            self._initialize_nullbooth()

    def _initialize_nullbooth(self):
        """Initialize NullBooth components."""
        print("\n" + "="*60)
        print("🎯 Initializing NullBooth (AlphaEdit) Training")
        print("="*60)

        # Get covariance matrices directory
        cov_dir = getattr(self.config.nullbooth, 'cov_matrices_output_dir', None)
        if not cov_dir:
            raise ValueError("NullBooth enabled but cov_matrices_output_dir not specified")

        # Initialize covariance manager
        self.cov_manager = CovarianceMatrixManager(
            cov_matrices_dir=cov_dir,
            device=self.device,
            nullspace_threshold=getattr(self.config.nullbooth, 'nullspace_threshold', 2e-2),
            threshold_percent=getattr(self.config.nullbooth, 'threshold_percent', None),
            cache_size=getattr(self.config.nullbooth, 'cache_size', 50),
            use_interpolation=getattr(self.config.nullbooth, 'use_interpolation', True),
            debug=getattr(self.config.nullbooth, 'debug', False)
        )

        print(f"✅ NullBooth initialized successfully")
        print(f"  Covariance dir: {cov_dir}")
        print(f"  Available timesteps: {len(self.cov_manager.available_timesteps)}")
        print("="*60 + "\n")

    def wrap_optimizer(self, optimizer):
        """Wrap the optimizer with AlphaEdit projection."""
        if not self.enabled:
            return optimizer

        self.optimizer_wrapper = AlphaEditOptimizer(
            optimizer=optimizer,
            cov_manager=self.cov_manager,
            unet=self.unet,
            enable_projection=True,
            debug=getattr(self.config.nullbooth, 'debug', False)
        )

        print("✅ Optimizer wrapped with AlphaEdit projection")
        return self.optimizer_wrapper

    def set_timestep(self, timestep: int):
        """Set current timestep for projection matrix selection."""
        if self.optimizer_wrapper:
            self.optimizer_wrapper.set_current_timestep(timestep)

    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up NullBooth trainer...")
        # Clear caches
        if self.cov_manager:
            self.cov_manager.projection_cache.clear()
            self.cov_manager.access_count.clear()