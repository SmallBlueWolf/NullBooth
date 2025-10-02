"""
Improved Modern Samplers with proper alpha/sigma value handling
For Karras and other modern samplers that use continuous noise values
"""

import torch
import numpy as np


class ImprovedModernSamplers:
    """Modern diffusion samplers with proper alpha/sigma value handling."""

    @staticmethod
    def get_sampler_timesteps_and_alphas(sampler_strategy: str, num_denoising_steps: int, total_timesteps: int = 1000):
        """Get timesteps and alpha values for different sampling strategies.

        Returns:
            timesteps: List of approximate integer timesteps (for compatibility)
            alphas: List of actual alpha/sigma values used by the sampler
            mapping: Dict mapping from timestep format to alpha format for file naming
        """
        if sampler_strategy is None:
            # Original uniform spacing - no special alpha values needed
            if num_denoising_steps >= total_timesteps:
                timesteps = list(range(total_timesteps-1, -1, -1))
            else:
                timesteps = torch.linspace(total_timesteps-1, 0, num_denoising_steps, dtype=torch.long).tolist()

            # For uniform sampling, no special alpha naming needed
            alphas = timesteps
            mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}
            return timesteps, alphas, mapping

        # Special case: early_timestep strategy
        if sampler_strategy == "early_timestep":
            # Select the earliest timesteps (highest noise levels)
            # Starting from timestep 999 (or total_timesteps-1), going down
            timesteps = list(range(total_timesteps-1, max(total_timesteps-num_denoising_steps-1, -1), -1))
            alphas = timesteps
            mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}
            print(f"Early timestep strategy: selecting {len(timesteps)} timesteps from {timesteps[0]} to {timesteps[-1]}")
            return timesteps, alphas, mapping

        sampler_map = {
            "DPM++ 2M": ImprovedModernSamplers.dpmpp_2m_timesteps_and_alphas,
            "DPM++ 2M Karras": ImprovedModernSamplers.dpmpp_2m_karras_timesteps_and_alphas,
            "DPM++ 3M": ImprovedModernSamplers.dpmpp_3m_timesteps_and_alphas,
            "DPM++ 3M SDE Karras": ImprovedModernSamplers.dpmpp_3m_sde_karras_timesteps_and_alphas,
            "Euler": ImprovedModernSamplers.euler_timesteps_and_alphas,
            "Euler a": ImprovedModernSamplers.euler_a_timesteps_and_alphas,
            "UniPC": ImprovedModernSamplers.unipc_timesteps_and_alphas,
        }

        if sampler_strategy not in sampler_map:
            raise ValueError(f"Unsupported sampler strategy: {sampler_strategy}")

        return sampler_map[sampler_strategy](num_denoising_steps, total_timesteps)

    @staticmethod
    def alpha_to_filename(alpha: float) -> str:
        """Convert alpha/sigma value to a filename-safe format.

        Uses scientific notation: alpha_MM_XXeYY
        where MM is the step index (00-99)
        XX is the mantissa (00-99)
        YY is the exponent
        """
        if alpha >= 100.0:
            # Very large values
            return f"alpha_99_1e2"
        elif alpha >= 10.0:
            # Large values: 10.0 - 99.9
            mantissa = int(alpha)
            return f"alpha_{mantissa:02d}_1e1"
        elif alpha >= 1.0:
            # Values 1.0 - 9.9
            mantissa = int(alpha * 10)
            exp = 0
            return f"alpha_{mantissa:02d}_{mantissa}e{exp}"
        elif alpha >= 0.01:
            # Values 0.01 - 0.99
            mantissa = int(alpha * 100)
            exp = -2
            return f"alpha_{mantissa:02d}_{mantissa}e{exp}"
        else:
            # Small values < 0.01
            exp = int(np.floor(np.log10(alpha)))
            mantissa = int(alpha / (10 ** exp))
            return f"alpha_{mantissa:02d}_{mantissa}e{exp}"

    @staticmethod
    def dpmpp_2m_karras_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """DPM++ 2M Karras timestep schedule with proper alpha values.

        Karras schedule uses continuous sigma values that don't map cleanly to integer timesteps.
        We preserve the actual sigma values for proper null-space projection.
        """
        # Karras schedule parameters
        sigma_min, sigma_max = 0.1, 10.0
        rho = 7.0

        # Generate sigma schedule
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # Flip to go from high noise to low noise
        sigmas = sigmas.flip(0)
        alphas = sigmas.tolist()

        # For compatibility, also compute approximate timesteps
        # But these are just approximations - the real values are in alphas
        timesteps_approx = (total_timesteps - 1) * (sigmas / sigma_max)
        timesteps_approx = torch.clamp(timesteps_approx, 0, total_timesteps - 1).long()
        timesteps = timesteps_approx.tolist()

        # Create mapping for file naming
        # Use both step index and alpha value for unique identification
        mapping = {}
        for i, (t, alpha) in enumerate(zip(timesteps, alphas)):
            # Use step index for ordering and alpha value for precision
            alpha_key = f"alpha_{i:02d}_{int(alpha*1e3):04d}"  # e.g., alpha_00_10000 for step 0, alpha=10.0
            mapping[f"timestep_{t:04d}"] = alpha_key

        return timesteps, alphas, mapping

    @staticmethod
    def dpmpp_2m_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """DPM++ 2M timestep schedule (non-Karras)."""
        # DPM++ 2M uses exponential spacing
        timesteps = torch.logspace(
            np.log10(total_timesteps - 1),
            np.log10(1),
            num_steps,
            dtype=torch.float32
        )
        timesteps = timesteps.flip(0).long().tolist()

        # For non-Karras DPM++, timesteps are the actual values
        alphas = timesteps
        mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}

        return timesteps, alphas, mapping

    @staticmethod
    def dpmpp_3m_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """DPM++ 3M timestep schedule."""
        # Similar to 2M but with different spacing for 3rd order
        timesteps = torch.logspace(
            np.log10(total_timesteps - 1),
            np.log10(0.5),
            num_steps,
            dtype=torch.float32
        )
        timesteps = timesteps.flip(0).long().tolist()

        alphas = timesteps
        mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}

        return timesteps, alphas, mapping

    @staticmethod
    def dpmpp_3m_sde_karras_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """DPM++ 3M SDE Karras timestep schedule with stochastic component."""
        # Karras schedule with SDE
        sigma_min, sigma_max = 0.1, 10.0
        rho = 7.0

        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # Note: In actual usage, SDE adds stochasticity during sampling
        # Here we just use the base Karras schedule
        sigmas = sigmas.flip(0)
        alphas = sigmas.tolist()

        # Approximate timesteps
        timesteps_approx = (total_timesteps - 1) * (sigmas / sigma_max)
        timesteps_approx = torch.clamp(timesteps_approx, 0, total_timesteps - 1).long()
        timesteps = timesteps_approx.tolist()

        # Create mapping
        mapping = {}
        for i, (t, alpha) in enumerate(zip(timesteps, alphas)):
            alpha_key = f"alpha_{i:02d}_{int(alpha*1e3):04d}"
            mapping[f"timestep_{t:04d}"] = alpha_key

        return timesteps, alphas, mapping

    @staticmethod
    def euler_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """Euler method timestep schedule."""
        # Simple linear spacing for Euler method
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        alphas = timesteps
        mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}
        return timesteps, alphas, mapping

    @staticmethod
    def euler_a_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """Euler Ancestral timestep schedule."""
        # Slightly modified Euler with ancestral sampling consideration
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps + 1, dtype=torch.long)[:-1].tolist()
        alphas = timesteps
        mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}
        return timesteps, alphas, mapping

    @staticmethod
    def unipc_timesteps_and_alphas(num_steps: int, total_timesteps: int = 1000):
        """UniPC (Unified Predictor-Corrector) timestep schedule."""
        # UniPC uses adaptive timestep selection
        base_timesteps = torch.logspace(
            np.log10(total_timesteps - 1),
            np.log10(1),
            num_steps,
            dtype=torch.float32
        ).long()

        timesteps = base_timesteps.flip(0).tolist()
        alphas = timesteps
        mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}

        return timesteps, alphas, mapping


def test_samplers():
    """Test the improved sampler implementation."""
    print("\n" + "="*60)
    print("Testing Improved Modern Samplers")
    print("="*60)

    # Test DPM++ 2M Karras
    sampler_strategy = "DPM++ 2M Karras"
    num_steps = 30

    timesteps, alphas, mapping = ImprovedModernSamplers.get_sampler_timesteps_and_alphas(
        sampler_strategy, num_steps
    )

    print(f"\nSampler: {sampler_strategy}")
    print(f"Number of steps: {num_steps}")
    print(f"\nFirst 5 entries:")
    for i in range(min(5, len(timesteps))):
        t = timesteps[i]
        a = alphas[i]
        key = f"timestep_{t:04d}"
        mapped = mapping[key]
        print(f"  Step {i}: timestep={t:4d}, alpha={a:.4f}, mapped={mapped}")

    print(f"\nLast 5 entries:")
    for i in range(max(0, len(timesteps)-5), len(timesteps)):
        t = timesteps[i]
        a = alphas[i]
        key = f"timestep_{t:04d}"
        mapped = mapping[key]
        print(f"  Step {i}: timestep={t:4d}, alpha={a:.4f}, mapped={mapped}")

    # Verify properties
    print(f"\n✓ Total steps: {len(timesteps)}")
    print(f"✓ Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]")
    print(f"✓ All mappings unique: {len(set(mapping.values())) == len(mapping)}")

    return timesteps, alphas, mapping


if __name__ == "__main__":
    test_samplers()