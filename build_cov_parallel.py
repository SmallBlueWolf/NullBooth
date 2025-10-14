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
import hashlib
import tempfile
from math import ceil
from concurrent.futures import ThreadPoolExecutor

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
from typing import Callable


class ModernSamplers:
    """Modern diffusion samplers implementation for timestep scheduling."""
    
    @staticmethod
    def get_sampler_timesteps(sampler_strategy: str, num_denoising_steps: int, total_timesteps: int = 1000) -> list:
        """Get timesteps for different sampling strategies."""
        if sampler_strategy is None or sampler_strategy == "uniform":
            # Original uniform spacing
            if num_denoising_steps >= total_timesteps:
                return list(range(total_timesteps-1, -1, -1))
            else:
                return torch.linspace(total_timesteps-1, 0, num_denoising_steps, dtype=torch.long).tolist()

        # Special case: early_timestep strategy
        if sampler_strategy == "early_timestep":
            # Select the earliest timesteps (highest noise levels)
            # Starting from timestep 999 (or total_timesteps-1), going down
            timesteps = list(range(total_timesteps-1, max(total_timesteps-num_denoising_steps-1, -1), -1))
            print(f"Early timestep strategy: selecting {len(timesteps)} timesteps from {timesteps[0]} to {timesteps[-1]}")
            return timesteps

        # LCM scheduler (special handling for few-step models)
        if sampler_strategy == "LCM":
            return ModernSamplers.lcm_timesteps(num_denoising_steps, total_timesteps)

        sampler_map = {
            "DPM++ 2M": ModernSamplers.dpmpp_2m_timesteps,
            "DPM++ 2M Karras": ModernSamplers.dpmpp_2m_karras_timesteps,
            "DPM++ 3M": ModernSamplers.dpmpp_3m_timesteps,
            "DPM++ 3M SDE Karras": ModernSamplers.dpmpp_3m_sde_karras_timesteps,
            "Euler": ModernSamplers.euler_timesteps,
            "Euler a": ModernSamplers.euler_a_timesteps,
            "UniPC": ModernSamplers.unipc_timesteps,
        }

        if sampler_strategy not in sampler_map:
            raise ValueError(f"Unsupported sampler strategy: {sampler_strategy}")

        return sampler_map[sampler_strategy](num_denoising_steps, total_timesteps)
    
    @staticmethod
    def dpmpp_2m_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """DPM++ 2M timestep schedule."""
        # DPM++ 2M uses exponential spacing with specific beta schedule
        timesteps = torch.logspace(
            torch.log10(torch.tensor(total_timesteps - 1)), 
            torch.log10(torch.tensor(1)), 
            num_steps, 
            dtype=torch.long
        ).flip(0)
        return timesteps.tolist()
    
    @staticmethod
    def dpmpp_2m_karras_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """DPM++ 2M Karras timestep schedule."""
        # Karras schedule: σ_i = σ_min^(1-i/(n-1)) * σ_max^(i/(n-1))
        sigma_min, sigma_max = 0.1, 10.0
        rho = 7.0
        
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # Convert sigmas to timesteps (approximate mapping)
        timesteps = (total_timesteps - 1) * (1 - sigmas / sigma_max)
        timesteps = torch.clamp(timesteps, 0, total_timesteps - 1).long()
        return timesteps.flip(0).tolist()
    
    @staticmethod
    def dpmpp_3m_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """DPM++ 3M timestep schedule."""
        # Similar to 2M but with different spacing for 3rd order
        timesteps = torch.logspace(
            torch.log10(torch.tensor(total_timesteps - 1)), 
            torch.log10(torch.tensor(0.5)), 
            num_steps, 
            dtype=torch.long
        ).flip(0)
        return timesteps.tolist()
    
    @staticmethod
    def dpmpp_3m_sde_karras_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """DPM++ 3M SDE Karras timestep schedule."""
        # Combination of 3M and Karras with SDE modifications
        # SDE 采样器应该保持随机性特征
        sigma_min, sigma_max = 0.1, 10.0
        rho = 7.0
        
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # SDE modification: add stochastic perturbation (randomness restored)
        # 这是 SDE 的核心特征 - 随机扰动
        perturbation = 0.02 * torch.randn_like(sigmas)  # 恢复随机性
        sigmas = sigmas * (1 + perturbation)
        
        timesteps = (total_timesteps - 1) * (1 - sigmas / sigma_max)
        timesteps = torch.clamp(timesteps, 0, total_timesteps - 1).long()
        return timesteps.flip(0).tolist()
    
    @staticmethod
    def euler_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """Euler method timestep schedule."""
        # Simple linear spacing for Euler method
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps, dtype=torch.long)
        return timesteps.tolist()
    
    @staticmethod
    def euler_a_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """Euler Ancestral timestep schedule."""
        # Slightly modified Euler with ancestral sampling consideration
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps + 1, dtype=torch.long)[:-1]
        return timesteps.tolist()
    
    @staticmethod
    def unipc_timesteps(num_steps: int, total_timesteps: int = 1000) -> list:
        """UniPC (Unified Predictor-Corrector) timestep schedule."""
        # UniPC uses adaptive timestep selection
        # Start with exponential spacing and refine
        base_timesteps = torch.logspace(
            torch.log10(torch.tensor(total_timesteps - 1)), 
            torch.log10(torch.tensor(1)), 
            num_steps, 
            dtype=torch.long
        )
        
        # UniPC adaptive refinement (simplified)
        refined_timesteps = []
        for i, t in enumerate(base_timesteps):
            if i > 0:
                # Add intermediate points for better accuracy
                prev_t = refined_timesteps[-1] if refined_timesteps else total_timesteps - 1
                if t - prev_t > 50:  # Add intermediate point for large gaps
                    mid_t = (t + prev_t) // 2
                    refined_timesteps.append(mid_t.item())
            refined_timesteps.append(t.item())
        
        # Ensure we have exactly num_steps
        if len(refined_timesteps) > num_steps:
            # Downsample uniformly
            indices = torch.linspace(0, len(refined_timesteps) - 1, num_steps, dtype=torch.long)
            refined_timesteps = [refined_timesteps[i] for i in indices]
        elif len(refined_timesteps) < num_steps:
            # Upsample by interpolation
            current_timesteps = torch.tensor(refined_timesteps, dtype=torch.float)
            new_indices = torch.linspace(0, len(refined_timesteps) - 1, num_steps)
            refined_timesteps = torch.nn.functional.interpolate(
                current_timesteps.unsqueeze(0).unsqueeze(0), 
                size=num_steps, 
                mode='linear', 
                align_corners=True
            ).squeeze().long().tolist()
        
        return list(reversed(refined_timesteps))

    @staticmethod
    def lcm_timesteps(num_steps: int, total_timesteps: int = 1000, original_inference_steps: int = 50) -> list:
        """
        LCM (Latent Consistency Model) timestep schedule.
        LCM uses very few inference steps (typically 1-8) but maps them to the
        original training timesteps (e.g., 50 steps).

        Args:
            num_steps: Number of inference steps (e.g., 1, 2, 4, 8)
            total_timesteps: Total training timesteps (1000)
            original_inference_steps: Original steps LCM was distilled from (typically 50)

        Returns:
            List of timesteps for LCM inference
        """
        # LCM timestep spacing follows the "leading" strategy
        # Map num_steps to the range [0, original_inference_steps]
        c = total_timesteps // original_inference_steps

        # Generate linearly spaced timestep indices
        lcm_origin_timesteps = np.asarray(list(range(1, original_inference_steps + 1))) * c - 1

        # Select subset based on num_steps
        # For num_steps < original_inference_steps, we skip timesteps
        skipping_step = len(lcm_origin_timesteps) // num_steps

        # Get evenly spaced indices
        indices = np.arange(0, len(lcm_origin_timesteps), skipping_step)[:num_steps]
        timesteps = lcm_origin_timesteps[indices]

        # LCM uses descending order (high noise to low noise)
        timesteps = timesteps[::-1].copy()

        print(f"LCM scheduler: {num_steps} steps from {len(lcm_origin_timesteps)} original timesteps")
        print(f"  Timesteps: {timesteps.tolist()}")

        return timesteps.tolist()


class DataPrefetcher:
    """数据预取器，实现CPU/GPU流水线并行"""

    def __init__(self, file_paths, safe_key, batch_size=2000, num_workers=16, device='cuda'):
        self.file_paths = file_paths
        self.safe_key = safe_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.current_idx = 0
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_future = None

    def _load_feature(self, file_info):
        """加载单个特征文件"""
        file_path, key = file_info
        try:
            # 使用mmap_mode='r'减少内存占用
            data = np.load(file_path, mmap_mode='r')
            if key in data.files:
                feature_data = np.array(data[key])  # 转换为常规数组
                return feature_data
            return None
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None

    def _load_batch(self, batch_paths):
        """并行加载一批数据"""
        from concurrent.futures import as_completed
        batch_features = []
        original_shape = None

        # 提交所有加载任务
        futures = {self.executor.submit(self._load_feature, fp): fp for fp in batch_paths}

        # 收集结果
        for future in as_completed(futures):
            feature_data = future.result()
            if feature_data is not None:
                if original_shape is None:
                    original_shape = feature_data.shape

                # Reshape to (n_samples, feature_dim)
                if feature_data.ndim > 2:
                    feature_data = feature_data.reshape(-1, feature_data.shape[-1])

                batch_features.append(feature_data)

        if batch_features:
            # 在CPU上合并数据
            batch_array = np.concatenate(batch_features, axis=0)
            return batch_array, original_shape
        return None, None

    def prefetch_next(self):
        """异步预取下一批数据"""
        if self.current_idx < len(self.file_paths):
            batch_end = min(self.current_idx + self.batch_size, len(self.file_paths))
            batch_paths = [(self.file_paths[i], self.safe_key) for i in range(self.current_idx, batch_end)]
            self.prefetch_future = self.executor.submit(self._load_batch, batch_paths)
            self.current_idx = batch_end

    def get_next_batch(self):
        """获取下一批数据（可能已经预取）"""
        if self.prefetch_future is None:
            self.prefetch_next()

        if self.prefetch_future:
            # 等待预取完成
            batch_data, original_shape = self.prefetch_future.result()

            # 立即开始预取下一批
            self.prefetch_next()

            if batch_data is not None:
                # 将数据转换为GPU张量
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
                return batch_tensor, original_shape

        return None, None

    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=False)


class AsyncWriter:

    def __init__(self, max_workers: int = 0):
        self.max_workers = max_workers if max_workers and max_workers > 0 else 0
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.max_workers > 0 else None
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        if self.executor:
            future = self.executor.submit(fn, *args, **kwargs)
            self.futures.append(future)
            return future
        fn(*args, **kwargs)
        return None

    def wait(self):
        if not self.executor:
            return
        for future in self.futures:
            future.result()
        self.futures.clear()

    def shutdown(self):
        if not self.executor:
            return
        self.wait()
        self.executor.shutdown(wait=True)


class TimestepAverager:
    """Streaming averager for timestep sequences across prompts."""
    
    def __init__(self):
        self.sum_timesteps = None
        self.count = 0
    
    def update(self, timesteps: list):
        """Update running average with new timestep sequence."""
        if not timesteps:
            raise ValueError("Empty timestep list provided to TimestepAverager.update()")
            
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float)
        
        if self.sum_timesteps is None:
            self.sum_timesteps = timesteps_tensor.clone()
        else:
            # Ensure same length by interpolation if needed
            if len(self.sum_timesteps) != len(timesteps_tensor):
                target_length = max(len(self.sum_timesteps), len(timesteps_tensor))
                # 直接报错，不使用 try-catch 回退机制
                self.sum_timesteps = torch.nn.functional.interpolate(
                    self.sum_timesteps.unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=True
                ).squeeze()
                timesteps_tensor = torch.nn.functional.interpolate(
                    timesteps_tensor.unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=True
                ).squeeze()
            
            self.sum_timesteps += timesteps_tensor
        
        self.count += 1
    
    def get_average_timesteps(self) -> list:
        """Get average timestep sequence."""
        if self.sum_timesteps is None or self.count == 0:
            raise RuntimeError("No timesteps have been added to TimestepAverager. Call update() first.")
        
        avg_timesteps = self.sum_timesteps / self.count
        return avg_timesteps.round().long().tolist()


def generate_alpha_filename(index: int, timestep_value: float) -> str:
    """Generate filename in format alpha_xx_xxxx where xx is index and xxxx is exact timestep value."""
    # Use exact timestep value to preserve precision (no exponential notation)
    timestep_int = int(timestep_value)
    return f"alpha_{index:02d}_{timestep_int:04d}"


class ProgressTracker:
    """Tracks computation progress with JSON-based persistence and parameter validation."""
    
    def __init__(self, run_dir: Path, config: Dict, accelerator: Accelerator, resume: bool = True):
        self.run_dir = Path(run_dir)
        self.config = config
        self.accelerator = accelerator
        self.resume = resume  # Store resume flag
        self.progress_file = self.run_dir / "computation_progress.json"
        self.parameters_hash = self._compute_parameters_hash()

        # Initialize progress tracking
        self.progress = {
            "version": "1.1",  # Updated version for new features
            "parameters_hash": self.parameters_hash,
            "parameters": self._extract_parameters(),
            "timestamps": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "phase1_features": {
                "status": "not_started",
                "completed_timesteps": {},
                "partial_timesteps": {},  # New: track partially completed timesteps
                "total_prompts": 0,
                "completed_prompts": 0
            },
            "phase2_covariance": {
                "status": "not_started",
                "completed_timesteps": [],
                "total_timesteps": 0,
                "completed_matrices": 0
            }
        }

        # Load existing progress if available AND resume is enabled
        if self.resume:
            self._load_existing_progress()
        else:
            if accelerator.is_main_process and self.progress_file.exists():
                print("⚠️  Found existing progress file but --resume not specified")
                print("   Starting fresh computation (use --resume to continue from checkpoint)")
                self._backup_old_progress()

        # Always scan cache for Phase 1 if incomplete, regardless of resume flag
        # This allows Phase 2 to detect completed Phase 1 results
        phase1_status = self.progress.get('phase1_features', {}).get('status', 'not_started')
        if phase1_status != 'completed':
            # Scan cache directory to update progress if needed
            self._scan_and_update_progress()
    
    def _compute_parameters_hash(self) -> str:
        """Compute hash of critical parameters for cache validation."""
        critical_params = {
            "model_path": getattr(self.config, 'model_path', ''),
            "prompts_file": getattr(self.config.nullbooth, 'prompts_file', ''),
            "num_denoising_steps": getattr(self.config.nullbooth, 'num_denoising_steps', 50),
            "sampler_strategy": getattr(self.config.nullbooth, 'sampler_strategy', None),
            "timestep_mode": getattr(self.config.nullbooth, 'timestep_mode', 'avg'),
            "resolution": getattr(self.config, 'resolution', 512),
            "nullspace_threshold": getattr(self.config.nullbooth, 'nullspace_threshold', 1e-4),
            "collect_q": getattr(self.config.nullbooth, 'collect_q', True),
            "collect_k": getattr(self.config.nullbooth, 'collect_k', True),
            "collect_v": getattr(self.config.nullbooth, 'collect_v', True),
            "collect_out": getattr(self.config.nullbooth, 'collect_out', True),
            "visual_attention_map": getattr(self.config.nullbooth, 'visual_attention_map', True),
            "only_covariance_matrix": getattr(self.config.nullbooth, 'only_covariance_matrix', False),
        }
        
        params_str = json.dumps(critical_params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]
    
    def _extract_parameters(self) -> Dict:
        """Extract all relevant parameters for documentation."""
        return {
            "model_path": getattr(self.config, 'model_path', ''),
            "prompts_file": getattr(self.config.nullbooth, 'prompts_file', ''),
            "num_denoising_steps": getattr(self.config.nullbooth, 'num_denoising_steps', 50),
            "sampler_strategy": getattr(self.config.nullbooth, 'sampler_strategy', None),
            "timestep_mode": getattr(self.config.nullbooth, 'timestep_mode', 'avg'),
            "resolution": getattr(self.config, 'resolution', 512),
            "nullspace_threshold": getattr(self.config.nullbooth, 'nullspace_threshold', 1e-4),
            "collect_q": getattr(self.config.nullbooth, 'collect_q', True),
            "collect_k": getattr(self.config.nullbooth, 'collect_k', True),
            "collect_v": getattr(self.config.nullbooth, 'collect_v', True),
            "collect_out": getattr(self.config.nullbooth, 'collect_out', True),
            "visual_attention_map": getattr(self.config.nullbooth, 'visual_attention_map', True),
            "covariance_batch_size": getattr(self.config.nullbooth, 'covariance_batch_size', 25),
            "feature_batch_size": getattr(self.config.nullbooth, 'feature_batch_size', 100),
            "only_covariance_matrix": getattr(self.config.nullbooth, 'only_covariance_matrix', False),
            "num_processes": self.accelerator.num_processes if self.accelerator else 1
        }
    
    def _load_existing_progress(self):
        """Load existing progress if compatible."""
        if not self.progress_file.exists():
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                existing_progress = json.load(f)
            
            # Validate parameters hash
            if existing_progress.get("parameters_hash") == self.parameters_hash:
                self.progress = existing_progress
                self.progress["timestamps"]["last_loaded"] = datetime.now().isoformat()

                # Only scan cache for Phase 1 if incomplete
                # Skip scanning for Phase 2 to avoid performance issues with 22k prompts
                phase1_status = self.progress.get('phase1_features', {}).get('status', 'not_started')
                if phase1_status != 'completed':
                    # Scan cache directory to update progress if needed
                    self._scan_and_update_progress()
                else:
                    if self.accelerator.is_main_process:
                        print(f"   Phase 1 already completed, skipping cache scan")

                if self.accelerator.is_main_process:
                    print(f"✅ Loaded compatible progress from {self.progress_file}")
                    print(f"   Phase 1: {self.progress['phase1_features']['completed_prompts']} prompts completed")
                    print(f"   Phase 2: {len(self.progress['phase2_covariance']['completed_timesteps'])} timesteps completed")
            else:
                if self.accelerator.is_main_process:
                    print(f"⚠️  Existing progress file has different parameters (hash mismatch)")
                    print(f"   Expected: {self.parameters_hash}")
                    print(f"   Found:    {existing_progress.get('parameters_hash', 'none')}")
                    print(f"   Starting fresh computation...")
                self._backup_old_progress()
                
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"⚠️  Could not load existing progress: {e}")
                print("   Starting fresh computation...")

    def _scan_and_update_progress(self):
        """Scan cache directory and update progress to match actual cached files."""
        if not self.accelerator.is_main_process:
            return

        print(f"DEBUG: Starting _scan_and_update_progress...")
        cache_dir = self.run_dir / "cache"
        if not cache_dir.exists():
            print(f"DEBUG: Cache dir does not exist: {cache_dir}")
            return

        print(f"DEBUG: Cache dir exists: {cache_dir}")
        # Scan for existing prompt directories and timestep files
        timestep_completion = {}
        total_prompts_found = 0

        try:
            print(f"DEBUG: Starting to scan prompt directories...")
            prompt_dirs_list = []
            for prompt_dir in cache_dir.iterdir():
                if prompt_dir.is_dir() and prompt_dir.name.startswith("prompt_"):
                    prompt_dirs_list.append(prompt_dir)

            total_prompts_found = len(prompt_dirs_list)
            print(f"DEBUG: Found {total_prompts_found} prompt directories to scan")

            for prompt_dir in prompt_dirs_list:
                # Check which timesteps exist for this prompt with integrity validation
                for timestep_file in prompt_dir.iterdir():
                    if (timestep_file.suffix == '.npz' and
                        'attention' not in timestep_file.name and
                        (timestep_file.stem.startswith('timestep_') or
                         timestep_file.stem.startswith('alpha_'))):

                            # Verify file integrity before counting with enhanced validation
                            try:
                                # Enhanced integrity check with size validation
                                if timestep_file.stat().st_size < 100:  # Files should be at least 100 bytes
                                    raise ValueError(f"File too small: {timestep_file.stat().st_size} bytes")

                                # Quick integrity check by trying to load file metadata
                                with np.load(timestep_file, allow_pickle=False) as data:
                                    if len(data.files) == 0:  # File has no valid data
                                        raise ValueError("File contains no data arrays")

                                    # Additional validation: check if at least one array has reasonable size
                                    has_valid_data = False
                                    for array_name in data.files[:3]:  # Check first 3 arrays
                                        try:
                                            array_data = data[array_name]
                                            if array_data.size > 0 and not np.isnan(array_data).all():
                                                has_valid_data = True
                                                break
                                        except:
                                            continue

                                    if not has_valid_data:
                                        raise ValueError("File contains no valid array data")

                                    timestep_key = timestep_file.stem
                                    if timestep_key not in timestep_completion:
                                        timestep_completion[timestep_key] = 0
                                    timestep_completion[timestep_key] += 1

                            except Exception as e:
                                # File is corrupted or invalid, attempt recovery
                                print(f"⚠️  Detected corrupted/invalid cache file: {timestep_file}")
                                print(f"    Error: {e}")

                                # Create backup before removal
                                backup_dir = timestep_file.parent / ".corrupted_backups"
                                backup_dir.mkdir(exist_ok=True)
                                backup_file = backup_dir / f"{timestep_file.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                                try:
                                    timestep_file.rename(backup_file)
                                    print(f"    Moved corrupted file to: {backup_file}")
                                except Exception as backup_error:
                                    print(f"    Failed to backup corrupted file: {backup_error}")
                                    try:
                                        timestep_file.unlink()
                                        print(f"    Removed corrupted file: {timestep_file}")
                                    except Exception as remove_error:
                                        print(f"    Failed to remove corrupted file: {remove_error}")
                                        pass  # File might be in use, skip removal

            # Update progress based on scan results
            if total_prompts_found > 0:
                # Find timesteps that are completed for all prompts
                completed_timesteps = {}
                for timestep_key, count in timestep_completion.items():
                    if count == total_prompts_found:  # All prompts have this timestep
                        completed_timesteps[timestep_key] = {
                            "completed_at": datetime.now().isoformat(),
                            "num_prompts": total_prompts_found
                        }

                # Update progress
                self.progress["phase1_features"]["completed_timesteps"] = completed_timesteps
                self.progress["phase1_features"]["total_prompts"] = total_prompts_found

                # If we have completed timesteps, mark phase 1 as completed
                if completed_timesteps:
                    self.progress["phase1_features"]["completed_prompts"] = total_prompts_found
                    self.progress["phase1_features"]["status"] = "completed"

                print(f"   Cache scan found: {total_prompts_found} prompts, {len(completed_timesteps)} completed timesteps")

        except Exception as e:
            # 直接抛出异常，不进行回退处理
            raise RuntimeError(f"Failed to scan cache directory: {e}") from e
    
    def _backup_old_progress(self):
        """Backup old progress file with timestamp."""
        if self.progress_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.progress_file.with_name(f"computation_progress_backup_{timestamp}.json")
            self.progress_file.rename(backup_file)
            if self.accelerator.is_main_process:
                print(f"   Backed up old progress to: {backup_file}")
    
    def save_progress(self):
        """Save current progress to JSON file with atomic write and error recovery."""
        if not self.accelerator.is_main_process:
            return  # Only main process should save progress

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self.progress["timestamps"]["last_updated"] = datetime.now().isoformat()

                # Atomic write: save to temporary file first, then rename
                temp_file = self.progress_file.with_suffix('.tmp')

                # Write to temp file with validation
                with open(temp_file, 'w') as f:
                    json.dump(self.progress, f, indent=2)

                # Verify temp file was written correctly
                if not temp_file.exists() or temp_file.stat().st_size == 0:
                    raise RuntimeError("Temporary progress file is empty or missing")

                # Validate JSON content
                with open(temp_file, 'r') as f:
                    json.load(f)  # This will raise exception if JSON is corrupted

                # Atomic rename - this is guaranteed to be atomic on POSIX systems
                temp_file.rename(self.progress_file)
                return  # Success, exit retry loop

            except Exception as e:
                # Clean up temp file if write failed
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass  # Ignore cleanup errors

                if attempt < max_retries - 1:
                    print(f"⚠️  Progress save attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to save progress after {max_retries} attempts: {e}")
                    print(f"   Computation will continue but progress may be lost if interrupted.")
                    # Don't raise exception to avoid stopping computation
    
    def is_timestep_completed_phase1(self, timestep: int, use_alpha_naming: bool = False, all_alpha_keys: List[str] = None) -> bool:
        """Check if a timestep's feature collection is completed."""
        if use_alpha_naming and all_alpha_keys:
            # For alpha naming, check if any alpha files for this timestep exist in completed list
            # Parse all alpha keys to find matching timestep values
            for alpha_key in all_alpha_keys:
                parts = alpha_key.split('_')
                if len(parts) >= 3:
                    # Parse exact timestep value from format alpha_xx_xxxx
                    timestep_value = int(parts[2])
                    if timestep_value == timestep and alpha_key in self.progress["phase1_features"]["completed_timesteps"]:
                        return True
            return False
        else:
            timestep_key = f"timestep_{timestep:04d}"
            return timestep_key in self.progress["phase1_features"]["completed_timesteps"]
    
    def mark_prompt_completed_phase1(self, timestep: int, prompt_idx: int, timestep_key: str = None):
        """Mark a specific prompt as completed for a timestep."""
        if not self.accelerator.is_main_process:
            return

        try:
            if timestep_key is None:
                timestep_key = f"timestep_{timestep:04d}"

            # Initialize partial tracking if needed
            if timestep_key not in self.progress["phase1_features"]["partial_timesteps"]:
                self.progress["phase1_features"]["partial_timesteps"][timestep_key] = {
                    "completed_prompts": [],
                    "last_updated": datetime.now().isoformat()
                }

            # Add prompt if not already recorded
            if prompt_idx not in self.progress["phase1_features"]["partial_timesteps"][timestep_key]["completed_prompts"]:
                self.progress["phase1_features"]["partial_timesteps"][timestep_key]["completed_prompts"].append(prompt_idx)
                self.progress["phase1_features"]["partial_timesteps"][timestep_key]["last_updated"] = datetime.now().isoformat()

        except Exception as e:
            print(f"⚠️  Warning: Failed to mark prompt {prompt_idx} for timestep {timestep}: {e}")

    def get_completed_prompts_for_timestep(self, timestep: int, timestep_key: str = None) -> List[int]:
        """Get list of completed prompt indices for a timestep."""
        if timestep_key is None:
            timestep_key = f"timestep_{timestep:04d}"

        partial_data = self.progress["phase1_features"].get("partial_timesteps", {}).get(timestep_key, {})
        return partial_data.get("completed_prompts", [])

    def mark_timestep_completed_phase1(self, timestep: int, num_prompts: int, timestep_key: str = None):
        """Mark a timestep as completed in phase 1 with error handling."""
        if not self.accelerator.is_main_process:
            return  # Only main process should update progress

        try:
            if timestep_key is None:
                timestep_key = f"timestep_{timestep:04d}"
            self.progress["phase1_features"]["completed_timesteps"][timestep_key] = {
                "completed_at": datetime.now().isoformat(),
                "num_prompts": num_prompts
            }
            # Clean up partial tracking when fully complete
            if timestep_key in self.progress["phase1_features"].get("partial_timesteps", {}):
                del self.progress["phase1_features"]["partial_timesteps"][timestep_key]
            self.save_progress()
        except Exception as e:
            print(f"⚠️  Warning: Failed to mark timestep {timestep} as completed: {e}")
            print(f"   Progress may not be saved correctly, but computation can continue.")
    
    def is_timestep_completed_phase2(self, timestep: int, use_alpha_naming: bool = False, all_alpha_keys: List[str] = None) -> bool:
        """Check if a timestep's covariance computation is completed."""
        if use_alpha_naming and all_alpha_keys:
            # For alpha naming, check if any alpha files for this timestep exist in completed list
            # Parse all alpha keys to find matching timestep values
            for alpha_key in all_alpha_keys:
                parts = alpha_key.split('_')
                if len(parts) >= 3:
                    # Parse exact timestep value from format alpha_xx_xxxx
                    timestep_value = int(parts[2])
                    if timestep_value == timestep and alpha_key in self.progress["phase2_covariance"]["completed_timesteps"]:
                        return True
            return False
        else:
            timestep_key = f"timestep_{timestep:04d}"
            return timestep_key in self.progress["phase2_covariance"]["completed_timesteps"]
    
    def mark_timestep_completed_phase2(self, timestep: int, timestep_key: str = None):
        """Mark a timestep as completed in phase 2 with error handling."""
        if not self.accelerator.is_main_process:
            return  # Only main process should update progress

        try:
            if timestep_key is None:
                timestep_key = f"timestep_{timestep:04d}"
            # Prevent duplicate entries
            if timestep_key not in self.progress["phase2_covariance"]["completed_timesteps"]:
                self.progress["phase2_covariance"]["completed_timesteps"].append(timestep_key)
                self.progress["phase2_covariance"]["completed_matrices"] += 1
            self.save_progress()
        except Exception as e:
            print(f"⚠️  Warning: Failed to mark timestep {timestep} as completed: {e}")
            print(f"   Progress may not be saved correctly, but computation can continue.")
    
    def update_phase1_status(self, status: str, completed_prompts: int = None, total_prompts: int = None):
        """Update phase 1 status."""
        self.progress["phase1_features"]["status"] = status
        if completed_prompts is not None:
            self.progress["phase1_features"]["completed_prompts"] = completed_prompts
        if total_prompts is not None:
            self.progress["phase1_features"]["total_prompts"] = total_prompts
        self.save_progress()
    
    def update_phase2_status(self, status: str, total_timesteps: int = None):
        """Update phase 2 status."""
        self.progress["phase2_covariance"]["status"] = status
        if total_timesteps is not None:
            self.progress["phase2_covariance"]["total_timesteps"] = total_timesteps
        self.save_progress()
    
    def get_remaining_timesteps_phase1(self, all_timesteps: List[int], use_alpha_naming: bool = False, all_alpha_keys: List[str] = None) -> List[int]:
        """Get timesteps that still need feature collection."""
        return [t for t in all_timesteps if not self.is_timestep_completed_phase1(t, use_alpha_naming, all_alpha_keys)]
    
    def get_remaining_timesteps_phase2(self, all_timesteps: List[int], use_alpha_naming: bool = False, all_alpha_keys: List[str] = None) -> List[int]:
        """Get timesteps that still need covariance computation."""
        return [t for t in all_timesteps if not self.is_timestep_completed_phase2(t, use_alpha_naming, all_alpha_keys)]


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
        
        # Setup global cache directory (process-agnostic structure)
        self.cache_dir = Path(cache_dir) / "cache" if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes
        
        # Feature collection flags - Fix config field access
        self.collect_q = getattr(config.nullbooth.collect_features, 'q_features', True)
        self.collect_k = getattr(config.nullbooth.collect_features, 'k_features', True)
        self.collect_v = getattr(config.nullbooth.collect_features, 'v_features', True)
        self.collect_out = getattr(config.nullbooth.collect_features, 'out_features', True)
        self.visual_attention_map = getattr(config.nullbooth, 'visual_attention_map', True)
        
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
        """Collect INPUT features to Q, K, V, and output components from cross-attention layers.

        IMPORTANT (AlphaEdit Implementation):
        - We collect INPUT features (not outputs) that go into to_q, to_k, to_v, to_out
        - These inputs are what get multiplied by the weight matrices we want to protect
        - Stored with keys 'q', 'k', 'v', 'out' (simple names for consistency)
        """
        if self.current_timestep is None:
            return

        # Only collect from cross-attention layers (attn2)
        if "attn2" not in layer_name:
            return

        # Get the hidden_states from input_data (INPUT to the attention module)
        if isinstance(input_data, tuple):
            hidden_states = input_data[0]  # INPUT to to_q
        else:
            hidden_states = input_data

        # Use stored encoder_hidden_states (INPUT to to_k and to_v)
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

        # Collect INPUT features (stored with simple names for consistency)
        with torch.no_grad():
            batch_size = hidden_states.shape[0] if hasattr(hidden_states, 'shape') and len(hidden_states.shape) >= 1 else 1
            # Input to to_q: hidden_states (from image latents)
            if self.collect_q:
                q_input = hidden_states.detach()
                # Flatten spatial dimensions if needed
                if len(q_input.shape) == 4:  # [batch, channels, height, width]
                    q_input = q_input.reshape(q_input.shape[0], q_input.shape[1], -1).permute(0, 2, 1)
                elif len(q_input.shape) == 3:  # [batch, seq_len, dim]
                    pass  # Already in correct format
                # Store with simple key 'q' (but it's the INPUT to to_q)
                layer_features['q'] = q_input.cpu().numpy()

            # Input to to_k: encoder_hidden_states (from text encoder)
            if self.collect_k:
                k_input = encoder_hidden_states.detach()
                # Store with simple key 'k' (but it's the INPUT to to_k)
                layer_features['k'] = k_input.cpu().numpy()

            # Input to to_v: encoder_hidden_states (from text encoder)
            if self.collect_v:
                v_input = encoder_hidden_states.detach()
                # Store with simple key 'v' (but it's the INPUT to to_v)
                layer_features['v'] = v_input.cpu().numpy()

            # For to_out, we need the input to the output projection
            if self.collect_out:
                # Compute Q, K, V outputs
                query = module.to_q(hidden_states)
                key = module.to_k(encoder_hidden_states)
                value = module.to_v(encoder_hidden_states)

                query = module.head_to_batch_dim(query)
                key = module.head_to_batch_dim(key)
                value = module.head_to_batch_dim(value)

                # Compute attention
                attention_scores = torch.matmul(query, key.transpose(-1, -2))
                attention_scores = attention_scores / (query.shape[-1] ** 0.5)
                attention_probs = F.softmax(attention_scores, dim=-1)

                # Apply attention to values - this is the INPUT to to_out
                attention_output = torch.matmul(attention_probs, value)
                attention_output = module.batch_to_head_dim(attention_output)

                # Store with simple key 'out' (but it's the INPUT to to_out)
                layer_features['out'] = attention_output.detach().cpu().numpy()

                # Store attention map for visualization if requested
                if self.visual_attention_map:
                    try:
                        attention_maps = attention_probs.detach().cpu().numpy()
                        attention_maps = attention_maps.reshape(batch_size, module.heads, attention_maps.shape[-2], attention_maps.shape[-1])
                    except Exception:
                        attention_maps = attention_probs.detach().cpu().numpy()
                    self.attention_maps_cache[timestep_key][layer_name] = attention_maps

        # Store features for this layer
        if layer_name not in self.feature_cache[timestep_key]:
            self.feature_cache[timestep_key][layer_name] = {}

        # Store INPUT features with simple keys 'q', 'k', 'v', 'out'
        for feature_type, feature_data in layer_features.items():
            if feature_type not in self.feature_cache[timestep_key][layer_name]:
                self.feature_cache[timestep_key][layer_name][feature_type] = []

            self.feature_cache[timestep_key][layer_name][feature_type].append(feature_data)
    
    def set_current_prompt(self, prompt_idx: int):
        """Set current prompt index for caching."""
        self.current_prompt_idx = prompt_idx

    def _atomic_write_npz(self, file_path: Path, data: Dict[str, np.ndarray], async_writer: Optional[AsyncWriter] = None):
        if not data:
            return

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        def write():
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=file_path.parent)
            os.close(tmp_fd)
            np.savez_compressed(tmp_path, **data)
            os.replace(tmp_path, file_path)

        if async_writer:
            async_writer.submit(write)
        else:
            write()

    def save_batch_features_to_disk(
        self,
        prompt_indices: List[int],
        timestep_mapping: Optional[dict] = None,
        async_writer: Optional[AsyncWriter] = None,
    ):
        """Save features for a batch of prompts to disk."""

        if not self.feature_cache:
            raise RuntimeError(
                "No features collected for current batch. This indicates hooks were not triggered or UNet forward pass failed."
            )

        num_samples = len(prompt_indices)

        for sample_pos, global_prompt_idx in enumerate(prompt_indices):
            self.current_prompt_idx = global_prompt_idx
            prompt_cache_dir = self.cache_dir / f"prompt_{global_prompt_idx:04d}"
            prompt_cache_dir.mkdir(parents=True, exist_ok=True)

            for timestep_key, timestep_features in self.feature_cache.items():
                filename_base = timestep_mapping.get(timestep_key, timestep_key) if timestep_mapping else timestep_key
                timestep_file = prompt_cache_dir / f"{filename_base}.npz"

                if timestep_file.exists():
                    continue

                save_dict = {}
                for layer_name, layer_features in timestep_features.items():
                    for feature_type, feature_data in layer_features.items():
                        if not feature_data:
                            continue

                        feature_array = feature_data[-1] if isinstance(feature_data, list) else feature_data
                        feature_array = np.asarray(feature_array)

                        if feature_array.ndim >= 1 and feature_array.shape[0] == num_samples:
                            sample_feature = np.array(feature_array[sample_pos], copy=True)
                        else:
                            sample_feature = np.array(feature_array, copy=True)

                        safe_key = f"{layer_name.replace('.', '_').replace('/', '_')}_{feature_type}"
                        save_dict[safe_key] = sample_feature

                if save_dict:
                    self._atomic_write_npz(timestep_file, save_dict, async_writer)
                else:
                    raise RuntimeError(
                        f"No valid features collected for {timestep_key} in prompt {global_prompt_idx}."
                    )

            if self.visual_attention_map and self.attention_maps_cache:
                for timestep_key, timestep_maps in self.attention_maps_cache.items():
                    filename_base = timestep_mapping.get(timestep_key, timestep_key) if timestep_mapping else timestep_key
                    attention_file = prompt_cache_dir / f"{filename_base}_attention.npz"

                    if attention_file.exists():
                        continue

                    attention_dict = {}
                    for layer_name, attention_data in timestep_maps.items():
                        if attention_data is None:
                            continue
                        attention_array = np.asarray(attention_data)
                        if attention_array.ndim >= 1 and attention_array.shape[0] == num_samples:
                            sample_attention = np.array(attention_array[sample_pos], copy=True)
                        else:
                            sample_attention = np.array(attention_array, copy=True)

                        safe_key = layer_name.replace('.', '_').replace('/', '_')
                        attention_dict[safe_key] = sample_attention

                    if attention_dict:
                        self._atomic_write_npz(attention_file, attention_dict, async_writer)

        self.feature_cache.clear()
        self.attention_maps_cache.clear()

    def save_prompt_features_to_disk(self, timestep_mapping: dict = None, async_writer: Optional[AsyncWriter] = None):
        """Backward compatible single-prompt save wrapper."""
        if self.current_prompt_idx is None:
            raise RuntimeError("Current prompt index not set before saving features.")
        self.save_batch_features_to_disk([self.current_prompt_idx], timestep_mapping, async_writer)
    
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
        # Use default threshold if not specified
        self.nullspace_threshold = float(getattr(config.nullbooth, 'nullspace_threshold', 1e-5) or 1e-5)
        # Get batch size from config with fallback to default
        self.covariance_batch_size = getattr(config.nullbooth, 'covariance_batch_size', 25)
        self.gpu_memory_batch_size = getattr(config.nullbooth, 'gpu_memory_batch_size', 100)  # New GPU batch size
        self.only_covariance_matrix = True  # Always only compute covariance matrices, no SVD/projection
        
    def gather_all_features_from_processes(self, run_dir: Path) -> Dict:
        """Gather features from unified cache directory."""
        if self.accelerator.is_main_process:
            print("DEBUG: Inside gather_all_features_from_processes")
            print("Analyzing unified cache directory...")

        # Wait for all processes to finish feature collection with error handling
        try:
            if self.accelerator.is_main_process:
                print("DEBUG: Waiting for all processes to sync...")
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print("DEBUG: All processes synced successfully")
        except Exception as sync_error:
            print(f"⚠️  Process {self.accelerator.process_index}: Feature collection sync failed: {sync_error}")
            print(f"   Continuing with cache analysis - some processes may be out of sync")

        # Use unified cache directory (process-agnostic)
        cache_dir = run_dir / "cache"

        if self.accelerator.is_main_process:
            print(f"DEBUG: Cache directory path: {cache_dir}")
            print(f"DEBUG: Cache directory exists: {cache_dir.exists()}")

        # ALL processes should analyze the cache directory
        # This ensures everyone has the same cache_structure for proper synchronization
        if cache_dir.exists():
            print(f"Process {self.accelerator.process_index}: Found unified cache directory: {cache_dir}")
            print(f"Process {self.accelerator.process_index}: Calling _merge_features_from_cache_dirs...")
            result = self._merge_features_from_cache_dirs([cache_dir])
            print(f"Process {self.accelerator.process_index}: _merge_features_from_cache_dirs returned with {len(result)} keys")
            return result
        else:
            print(f"❌ Process {self.accelerator.process_index}: Cache directory not found: {cache_dir}")
            return {}
    
    def _merge_features_from_cache_dirs(self, cache_dirs: List[Path]) -> Dict:
        """Return cache directory structure for streaming processing using optimized scanning."""
        all_timestep_keys = set()
        all_layer_feature_pairs = set()
        total_prompt_dirs = 0

        print(f"DEBUG: _merge_features_from_cache_dirs called with {len(cache_dirs)} cache dirs")

        # Optimized approach: analyze structure from first prompt directory only
        # Since all prompts have the same structure, we can infer the complete structure
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                print(f"DEBUG: Cache dir {cache_dir} does not exist, skipping")
                continue

            print(f"DEBUG: Scanning cache dir: {cache_dir}")

            # OPTIMIZATION: Only count directories, don't iterate through all
            # Use a faster method to count prompt directories
            prompt_count = 0
            first_prompt = None

            # Quick count using os.scandir which is faster than pathlib
            import os
            with os.scandir(cache_dir) as entries:
                for entry in entries:
                    if entry.is_dir() and entry.name.startswith("prompt_"):
                        prompt_count += 1
                        if first_prompt is None:
                            first_prompt = Path(entry.path)

            total_prompt_dirs = prompt_count
            print(f"DEBUG: Found {total_prompt_dirs} prompt directories (counted quickly)")

            if not first_prompt:
                print("DEBUG: No prompt directories found")
                continue

            # Use the first prompt directory to infer the complete structure
            print(f"Analyzing cache structure from {first_prompt.name} (assuming uniform structure)...")

            # First, collect all timestep keys from filenames
            print(f"DEBUG: Scanning timestep files in {first_prompt}...")
            for timestep_file in first_prompt.iterdir():
                # Skip temporary and attention files
                if timestep_file.suffix == '.tmp' or 'attention' in timestep_file.name:
                    continue
                if timestep_file.suffix == '.npz':
                    all_timestep_keys.add(timestep_file.stem)

            print(f"DEBUG: Found {len(all_timestep_keys)} timestep keys")

            # Then, analyze layer/feature combinations from just one file
            for timestep_file in first_prompt.iterdir():
                if timestep_file.suffix == '.npz' and 'attention' not in timestep_file.name:
                    try:
                        data = np.load(timestep_file)
                        for key in data.files:
                            parts = key.rsplit('_', 1)
                            if len(parts) == 2:
                                layer_name = parts[0].replace('_', '.')
                                feature_type = parts[1]
                                all_layer_feature_pairs.add((layer_name, feature_type))
                        data.close()
                        break  # Only need to analyze one file to get layer/feature structure
                    except Exception as e:
                        print(f"Warning: Failed to analyze {timestep_file}: {e}")
                        continue

        print(f"Found {total_prompt_dirs} prompt directories")
        print(f"Found {len(all_timestep_keys)} unique timesteps")
        print(f"Found {len(all_layer_feature_pairs)} unique layer/feature combinations")

        # Return structure for streaming processing
        return {
            'cache_dirs': cache_dirs,
            'timestep_keys': sorted(all_timestep_keys),
            'layer_feature_pairs': sorted(all_layer_feature_pairs)
        }
    
    def compute_covariance_matrices_from_merged_features(self, cache_structure: Dict) -> Dict:
        """Compute covariance matrices using optimized batch processing."""
        cov_matrices = {}

        cache_dirs = cache_structure['cache_dirs']
        timestep_keys = cache_structure['timestep_keys']
        layer_feature_pairs = cache_structure['layer_feature_pairs']

        print(f"Process {self.accelerator.process_index}: Computing covariance matrices with optimized batch processing...")
        print(f"Process {self.accelerator.process_index}: Processing {len(timestep_keys)} timesteps × {len(layer_feature_pairs)} layer/feature combinations")

        # Main progress bar for timesteps
        timestep_pbar = tqdm(
            timestep_keys,
            desc=f"Process {self.accelerator.process_index}: Computing timesteps",
            position=self.accelerator.process_index,
            disable=not self.accelerator.is_local_main_process,
            leave=True,
            colour='green' if self.accelerator.is_main_process else 'yellow'
        )

        # Process each timestep
        for timestep_key in timestep_pbar:
            cov_matrices[timestep_key] = {}

            # Update timestep description
            if self.accelerator.is_local_main_process:
                timestep_pbar.set_description(f"Process {self.accelerator.process_index}: {timestep_key}")

            # MEMORY OPTIMIZATION: Process one layer/feature at a time
            print(f"\n    Processing {len(layer_feature_pairs)} layer/feature combinations for {timestep_key}...")

            # Nested progress bar for layer/feature pairs
            layer_pbar = tqdm(
                layer_feature_pairs,
                desc=f"  └─ Layers for {timestep_key}",
                position=self.accelerator.process_index + self.accelerator.num_processes,
                disable=not self.accelerator.is_local_main_process,
                leave=False,
                colour='magenta'
            )

            # Process each layer/feature combination with streaming
            for layer_name, feature_type in layer_pbar:
                # Update layer progress description
                if self.accelerator.is_local_main_process:
                    layer_pbar.set_description(f"  └─ {layer_name}/{feature_type}")

                # Initialize nested dictionary structure
                if layer_name not in cov_matrices[timestep_key]:
                    cov_matrices[timestep_key][layer_name] = {}

                # Compute covariance using streaming (one layer at a time)
                cov_info = self._compute_covariance_streaming(
                    cache_dirs, timestep_key, layer_name, feature_type
                )

                if cov_info is not None:
                    cov_matrices[timestep_key][layer_name][feature_type] = cov_info
                    # Update postfix with matrix shape
                    if self.accelerator.is_local_main_process:
                        layer_pbar.set_postfix(shape=str(cov_info['covariance_matrix'].shape))

                # Clear GPU cache after each layer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Close layer progress bar
            layer_pbar.close()

            # Update timestep postfix with completion status
            if self.accelerator.is_local_main_process:
                num_computed = sum(len(layer_data) for layer_data in cov_matrices[timestep_key].values())
                timestep_pbar.set_postfix(matrices=num_computed)

        # Close timestep progress bar
        timestep_pbar.close()

        return cov_matrices

    def _load_all_features_for_timestep(self, cache_dirs: List[Path], timestep_key: str,
                                        layer_feature_pairs: List[Tuple[str, str]]) -> Dict:
        """Load all features for a timestep at once to avoid repeated file I/O."""
        all_features = {}
        batch_size = getattr(self.config.nullbooth, 'gpu_memory_batch_size', 1500)
        num_io_workers = getattr(self.config.nullbooth, 'async_io_workers', 4)

        # Collect all file paths once
        file_paths = []
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            prompt_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")]
            for prompt_dir in prompt_dirs:
                timestep_file = prompt_dir / f"{timestep_key}.npz"
                if timestep_file.exists():
                    file_paths.append(timestep_file)

        if not file_paths:
            return {}

        print(f"        Loading {len(file_paths)} files for timestep {timestep_key}...")

        # Load files in batches with parallel I/O
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def load_all_features_from_file(file_path):
            """Load all features from a single file."""
            features_dict = {}
            try:
                data = np.load(file_path)
                for layer_name, feature_type in layer_feature_pairs:
                    safe_key = f"{layer_name.replace('.', '_')}_{feature_type}"
                    if safe_key in data.files:
                        features_dict[(layer_name, feature_type)] = np.array(data[safe_key])
                data.close()
                return features_dict
            except Exception as e:
                print(f"        WARNING: Failed to load {file_path}: {e}")
                return {}

        # Process files in batches with progress bar
        file_pbar = tqdm(
            range(0, len(file_paths), batch_size),
            desc=f"        Loading batches",
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            colour='yellow'
        )

        # Initialize feature collectors
        for layer_name, feature_type in layer_feature_pairs:
            all_features[(layer_name, feature_type)] = []

        for i in file_pbar:
            batch_paths = file_paths[i:i+batch_size]

            # Update progress bar description
            if self.accelerator.is_local_main_process:
                file_pbar.set_description(f"        Batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

            # Parallel load current batch
            with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
                futures = [executor.submit(load_all_features_from_file, fp) for fp in batch_paths]

                for future in as_completed(futures):
                    try:
                        features_dict = future.result(timeout=30)
                        # Accumulate features
                        for key, feature_data in features_dict.items():
                            if key in all_features:
                                all_features[key].append(feature_data)
                    except Exception as e:
                        continue

        file_pbar.close()

        # Convert lists to numpy arrays
        print(f"        Concatenating features...")
        final_features = {}
        for key, feature_list in all_features.items():
            if feature_list:
                # Store both concatenated features and original shape
                concatenated = np.concatenate([
                    f.reshape(-1, f.shape[-1]) if f.ndim > 2 else f
                    for f in feature_list
                ], axis=0)
                final_features[key] = {
                    'features': concatenated,
                    'original_shape': feature_list[0].shape if feature_list else None,
                    'n_samples': len(feature_list)
                }

        return final_features

    def _compute_covariance_from_features(self, feature_data: Dict) -> Dict:
        """Compute covariance from pre-loaded features."""
        if not feature_data or 'features' not in feature_data:
            return None

        features = feature_data['features']
        original_shape = feature_data['original_shape']
        n_samples = feature_data['n_samples']

        # Move to GPU for computation
        device = self.accelerator.device
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)

        # Compute covariance
        feature_dim = features_tensor.shape[1]
        mean_x = torch.mean(features_tensor, dim=0)
        centered = features_tensor - mean_x
        cov_matrix = torch.matmul(centered.T, centered) / max(features_tensor.shape[0] - 1, 1)

        # Store result
        cov_info = {
            'covariance_matrix': cov_matrix.cpu().numpy(),
            'original_shape': original_shape,
            'n_samples': n_samples,
            'feature_dim': feature_dim,
            'projection_matrix': None,  # Skip SVD computation
            'null_space_dim': 0,
            'singular_values': None
        }

        # Clear GPU memory
        del features_tensor, cov_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cov_info

    def _compute_covariance_streaming(self, cache_dirs: List[Path], timestep_key: str,
                                    layer_name: str, feature_type: str) -> Dict:
        """Compute covariance for a single timestep/layer/feature using streaming with GPU acceleration."""
        # 获取批处理大小配置
        batch_size = getattr(self.config.nullbooth, 'gpu_memory_batch_size', 1500)
        device = self.accelerator.device
        num_io_workers = getattr(self.config.nullbooth, 'async_io_workers', 4)

        # Safe key for file lookup
        safe_key = f"{layer_name.replace('.', '_')}_{feature_type}"

        # First pass: collect all file paths WITHOUT opening files
        if self.accelerator.is_local_main_process:
            # Add progress bar for file path collection
            print(f"        Collecting file paths for {layer_name}/{feature_type}...")

        file_paths = []
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            # Use list comprehension for faster collection
            prompt_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")]
            for prompt_dir in prompt_dirs:
                timestep_file = prompt_dir / f"{timestep_key}.npz"
                if timestep_file.exists():
                    file_paths.append(timestep_file)

        if not file_paths:
            return None

        if self.accelerator.is_local_main_process:
            print(f"        Found {len(file_paths)} files to process")

        # 快速验证第一个文件是否包含所需的key
        try:
            data = np.load(file_paths[0])
            if safe_key not in data.files:
                data.close()
                return None
            data.close()
        except:
            return None

        # Silent processing - no print needed as progress is shown in tqdm

        # Initialize accumulators for streaming computation
        sum_x = None
        sum_xx = None
        n_samples = 0
        feature_dim = None
        original_shape = None

        # Simple parallel loading function
        def load_feature(file_path):
            """Load a single feature file."""
            try:
                data = np.load(file_path)
                if safe_key in data.files:
                    feature_data = np.array(data[safe_key])
                    data.close()
                    return feature_data
                data.close()
                return None
            except Exception as e:
                return None

        # Process data in batches with simple parallel I/O
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 添加文件处理进度条
        file_pbar = tqdm(
            range(0, len(file_paths), batch_size),
            desc=f"      Processing batches",
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            colour='yellow'
        )

        for i in file_pbar:
            batch_paths = file_paths[i:i+batch_size]
            batch_features = []

            # 更新进度条描述
            if self.accelerator.is_local_main_process:
                file_pbar.set_description(f"      Batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

            # Parallel load current batch
            with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
                futures = [executor.submit(load_feature, fp) for fp in batch_paths]

                for future in as_completed(futures):
                    try:
                        feature_data = future.result(timeout=30)  # 30 second timeout
                        if feature_data is not None:
                            # Store original shape from first sample
                            if original_shape is None:
                                original_shape = feature_data.shape

                            # Reshape to (n_samples, feature_dim)
                            if feature_data.ndim > 2:
                                feature_data = feature_data.reshape(-1, feature_data.shape[-1])

                            batch_features.append(feature_data)
                    except Exception:
                        continue

            if not batch_features:
                continue

            # Concatenate current batch and move to GPU
            batch_tensor = torch.tensor(np.concatenate(batch_features, axis=0),
                                      dtype=torch.float32, device=device)

            # Initialize accumulators on GPU on first batch
            if sum_x is None:
                feature_dim = batch_tensor.shape[1]
                sum_x = torch.zeros(feature_dim, device=device, dtype=torch.float32)
                sum_xx = torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float32)

            # Update statistics on GPU (use more efficient batch operations)
            n_samples += batch_tensor.shape[0]

            # Vectorized operations for better GPU utilization
            batch_sum = torch.sum(batch_tensor, dim=0)
            sum_x += batch_sum

            # More efficient covariance accumulation: X^T @ X
            sum_xx += torch.matmul(batch_tensor.T, batch_tensor)

            # Clear batch tensor to free GPU memory
            del batch_tensor

            # Periodically clear GPU cache
            if (i // batch_size) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if n_samples == 0 or sum_x is None:
            return None

        # Compute covariance matrix from accumulated statistics
        mean_x = sum_x / n_samples
        # Cov = E[XX^T] - E[X]E[X]^T = (sum_xx - n*mean*mean^T) / (n-1)
        cov_matrix = (sum_xx - n_samples * torch.outer(mean_x, mean_x)) / max(n_samples - 1, 1)

        # Store covariance matrix and compute null space projection
        cov_info = {
            'covariance_matrix': cov_matrix.cpu().numpy(),  # Move back to CPU for saving
            'original_shape': original_shape,
            'n_samples': n_samples,
            'feature_dim': feature_dim
        }

        # Always skip SVD computation - only compute covariance matrices K₀K₀ᵀ
        # Projection matrices will be computed during training when needed
        cov_info['projection_matrix'] = None
        cov_info['null_space_dim'] = 0
        cov_info['singular_values'] = None

        # Clear GPU memory
        del cov_matrix, sum_x, sum_xx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return cov_info


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


def save_single_timestep_covariance(cov_matrices: Dict, output_dir: Path, only_covariance: bool = True):
    """Save covariance matrices for a single timestep directly to disk (used by parallel processes).

    This function is designed to be called by each process independently to avoid
    gather_object() failures with large matrices.
    """
    covariance_dir = output_dir / "covariance_matrices"
    covariance_dir.mkdir(parents=True, exist_ok=True)

    for timestep, timestep_data in cov_matrices.items():
        for layer_name, layer_data in timestep_data.items():
            # Create safe filename from layer name
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            layer_dir = covariance_dir / safe_layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            for feature_type, cov_info in layer_data.items():
                # Save covariance matrix with timestep in filename
                cov_file = layer_dir / f"{timestep}_{feature_type}_cov.npy"
                np.save(cov_file, cov_info['covariance_matrix'])

                # Save metadata
                metadata = {
                    'original_shape': cov_info['original_shape'],
                    'n_samples': int(cov_info['n_samples']),
                    'feature_dim': int(cov_info['feature_dim']),
                    'only_covariance_matrix': only_covariance,
                }

                # Extract actual timestep value based on naming format
                if 'alpha_' in timestep:
                    parts = timestep.split('_')
                    if len(parts) >= 3:
                        actual_timestep = int(parts[2])
                    else:
                        actual_timestep = None
                elif 'timestep_' in timestep:
                    actual_timestep = int(timestep.split('_')[1])
                else:
                    actual_timestep = None

                metadata['actual_timestep'] = actual_timestep

                # Only save covariance matrix (projection computed during training)
                metadata.update({
                    'null_space_dim': None,
                    'singular_values': None,
                    'nullspace_threshold': None,
                })

                metadata_file = layer_dir / f"{timestep}_{feature_type}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

def save_covariance_matrices(cov_matrices: Dict, output_dir: Path, only_covariance: bool = True):
    """Save covariance matrices to disk in organized structure (only K₀K₀ᵀ, no projection)."""
    print(f"Saving {'covariance matrices only' if only_covariance else 'covariance and projection matrices'} to {output_dir}...")

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

                # Only save covariance matrix (projection will be computed during training)

                # Save metadata
                metadata = {
                    'original_shape': cov_info['original_shape'],
                    'n_samples': int(cov_info['n_samples']),
                    'feature_dim': int(cov_info['feature_dim']),
                    'only_covariance_matrix': only_covariance,
                }

                # Extract actual timestep value based on naming format
                if 'alpha_' in timestep:
                    # For alpha naming, extract timestep from filename: alpha_xx_xxxx -> xxxx value
                    parts = timestep.split('_')
                    if len(parts) >= 3:
                        # Parse exact timestep value from format alpha_xx_xxxx
                        actual_timestep = int(parts[2])
                    else:
                        actual_timestep = None
                elif 'timestep_' in timestep:
                    # For timestep naming, extract from timestep_xxxx format
                    actual_timestep = int(timestep.split('_')[1])
                else:
                    actual_timestep = None

                metadata['actual_timestep'] = actual_timestep

                # Add projection-related metadata only if computed
                if not only_covariance and cov_info.get('singular_values') is not None:
                    metadata.update({
                        'null_space_dim': int(cov_info['null_space_dim']),
                        'singular_values': cov_info['singular_values'].tolist(),
                        'nullspace_threshold': float(cov_info.get('nullspace_threshold', 1e-5)),
                    })
                else:
                    metadata.update({
                        'null_space_dim': None,
                        'singular_values': None,
                        'nullspace_threshold': None,
                    })

                metadata_file = layer_dir / f"{feature_type}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)


def distribute_prompts(prompts: List[str], accelerator: Accelerator) -> List[Tuple[int, str]]:
    """Distribute prompts across processes and return (global_index, prompt)."""
    total_prompts = len(prompts)
    prompts_per_process = ceil(total_prompts / accelerator.num_processes)
    
    start_idx = accelerator.process_index * prompts_per_process
    end_idx = min(start_idx + prompts_per_process, total_prompts)

    process_prompts = [(idx, prompts[idx]) for idx in range(start_idx, end_idx)]
    
    if accelerator.is_main_process:
        print(f"Distributing {total_prompts} prompts across {accelerator.num_processes} processes")
        for i in range(accelerator.num_processes):
            proc_start = i * prompts_per_process
            proc_end = min(proc_start + prompts_per_process, total_prompts)
            proc_count = max(0, proc_end - proc_start)
            print(f"  Process {i}: {proc_count} prompts (indices {proc_start}-{proc_end-1})")
    
    print(f"Process {accelerator.process_index}: Processing {len(process_prompts)} prompts")
    return process_prompts


def verify_system_configuration(config, accelerator):
    """Comprehensive system configuration verification for safe checkpoint resumption."""

    if accelerator.is_main_process:
        print(f"\\n{'='*60}")
        print(f"SYSTEM CONFIGURATION VERIFICATION")
        print(f"{'='*60}")

    issues_found = []
    warnings_found = []

    # 1. GPU Memory Verification
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if accelerator.is_main_process:
            print(f"🔍 GPU Memory Check: {gpu_count} GPUs detected")

        for i in range(gpu_count):
            try:
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                if total_memory < 40:  # RTX 5880 should have ~49GB
                    warnings_found.append(f"GPU {i} has only {total_memory:.1f}GB memory (expected ~49GB for RTX 5880)")
                elif accelerator.is_main_process:
                    print(f"  ✅ GPU {i}: {total_memory:.1f}GB memory available")
            except Exception as e:
                issues_found.append(f"Failed to check GPU {i} memory: {e}")

    # 2. Configuration Parameter Validation
    batch_size_gpu = getattr(config.nullbooth, 'gpu_memory_batch_size', 1500)
    batch_size_cov = getattr(config.nullbooth, 'covariance_batch_size', 70)
    num_processes = accelerator.num_processes

    if accelerator.is_main_process:
        print(f"🔍 Configuration Check:")
        print(f"  GPU batch size: {batch_size_gpu}")
        print(f"  Covariance batch size: {batch_size_cov}")
        print(f"  Number of processes: {num_processes}")

    # Conservative memory estimation (very rough)
    estimated_gpu_memory_per_process = (batch_size_gpu * 512 * 768 * 4) / (1024**3)  # Rough estimate in GB
    if estimated_gpu_memory_per_process > 40:
        warnings_found.append(f"GPU batch size may be too large: estimated {estimated_gpu_memory_per_process:.1f}GB per process")

    # 3. Output Directory Verification
    output_dir = Path(config.nullbooth.cov_matrices_output_dir)
    try:
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ Output directory writable: {output_dir}")
    except Exception as e:
        issues_found.append(f"Output directory not writable: {e}")

    # 4. Prompts File Verification
    prompts_file = getattr(config.nullbooth, 'original_knowledge_prompts', '')
    if prompts_file and accelerator.is_main_process:
        try:
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"✅ Prompts file valid: {len(prompts)} prompts loaded from {prompts_file}")
        except Exception as e:
            issues_found.append(f"Failed to read prompts file {prompts_file}: {e}")

    # 5. NCCL Configuration Check
    nccl_vars = ['NCCL_DEBUG', 'NCCL_SOCKET_IFNAME', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE']
    if accelerator.is_main_process and num_processes > 1:
        print(f"🔍 NCCL Configuration:")
        for var in nccl_vars:
            value = os.environ.get(var, 'Not set')
            print(f"  {var}: {value}")

    # 6. Disk Space Check
    if accelerator.is_main_process:
        try:
            import shutil
            free_space = shutil.disk_usage(output_dir).free / (1024**3)  # GB

            # Rough estimate: 1000 prompts × 50 timesteps × 4 feature types × ~10MB per file = ~2TB
            estimated_space_needed = 2000  # GB, conservative estimate

            if free_space < estimated_space_needed:
                warnings_found.append(f"Low disk space: {free_space:.1f}GB available, estimated need: {estimated_space_needed}GB")
            else:
                print(f"✅ Disk space sufficient: {free_space:.1f}GB available")
        except Exception as e:
            warnings_found.append(f"Could not check disk space: {e}")

    # Summary
    if accelerator.is_main_process:
        print(f"\\n{'='*60}")
        if issues_found:
            print(f"❌ CRITICAL ISSUES FOUND ({len(issues_found)}):")
            for issue in issues_found:
                print(f"  • {issue}")

        if warnings_found:
            print(f"⚠️  WARNINGS ({len(warnings_found)}):")
            for warning in warnings_found:
                print(f"  • {warning}")

        if not issues_found and not warnings_found:
            print(f"✅ All system checks passed!")

        print(f"{'='*60}\\n")

    return len(issues_found) == 0, issues_found, warnings_found


def main():
    """Main function with two-phase processing and JSON progress tracking."""
    # Initialize Accelerator with NCCL backend for better GPU performance
    kwargs = InitProcessGroupKwargs(backend='nccl', timeout=timedelta(hours=2))
    accelerator = Accelerator(
        split_batches=False,
        kwargs_handlers=[kwargs]
    )
    
    # Setup logging
    with log_script_execution("build_cov_parallel"):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Build covariance matrices for NullBooth in parallel")
        parser.add_argument("--config", "-c", default="configs/nullbooth.yaml", 
                           help="Path to configuration file")
        parser.add_argument("--phase", choices=["1", "2", "both"], default="both",
                           help="Run specific phase: 1=features, 2=covariance, both=sequential")
        parser.add_argument("--resume", action="store_true",
                           help="Resume from existing progress")
        args = parser.parse_args()
        
        # Load configuration
        if accelerator.is_main_process:
            print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        if accelerator.is_main_process:
            print(f"DEBUG: Configuration loaded successfully")
            print(f"DEBUG: args.phase = {args.phase}")
            print(f"DEBUG: args.resume = {args.resume}")

        # Verify system configuration before proceeding
        config_ok, critical_issues, warnings = verify_system_configuration(config, accelerator)

        if not config_ok:
            if accelerator.is_main_process:
                print("❌ Critical configuration issues detected. Please fix these issues before proceeding:")
                for issue in critical_issues:
                    print(f"  • {issue}")
            return

        if warnings and accelerator.is_main_process:
            print("⚠️  Configuration warnings detected. Consider reviewing these issues:")
            for warning in warnings:
                print(f"  • {warning}")
            print("Proceeding with computation...\\n")
        
        # Check if NullBooth is enabled
        if not config.nullbooth.enable:
            if accelerator.is_main_process:
                print("NullBooth mode is not enabled in config. Set nullbooth.enable=true to proceed.")
            return
        
        # Setup output directory
        output_dir = Path(config.nullbooth.cov_matrices_output_dir)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use cov_matrices root directory directly (no timestamped subdirectories)
        run_dir = output_dir
        if accelerator.is_main_process:
            print(f"Using output directory: {run_dir}")
        
        # Wait for main process to create directories with error handling
        max_sync_retries = 3
        for retry in range(max_sync_retries):
            try:
                accelerator.wait_for_everyone()
                break
            except Exception as sync_error:
                if retry < max_sync_retries - 1:
                    print(f"⚠️  Process {accelerator.process_index}: Directory sync retry {retry + 1}: {sync_error}")
                    import time
                    time.sleep(1.0)
                else:
                    print(f"❌ Process {accelerator.process_index}: Failed to sync after directory creation. Continuing anyway.")
                    break
        
        # Initialize progress tracker with resume flag
        progress_tracker = ProgressTracker(run_dir, config, accelerator, resume=args.resume)

        if accelerator.is_main_process:
            print(f"DEBUG: ProgressTracker initialized")
            print(f"DEBUG: Phase 1 status: {progress_tracker.progress['phase1_features']['status']}")
            print(f"DEBUG: Phase 1 completed timesteps: {list(progress_tracker.progress['phase1_features']['completed_timesteps'].keys())}")
        
        # Load prompts
        prompts = load_prompts_from_file(config.nullbooth.original_knowledge_prompts)
        if accelerator.is_main_process:
            print(f"Total prompts loaded: {len(prompts)}")
            print(f"Using {accelerator.num_processes} GPUs: {[f'cuda:{i}' for i in range(accelerator.num_processes)]}")
        
        # Get timesteps for processing with sampler strategy support
        total_timesteps = 1000  # Default diffusion timesteps
        num_sample_steps = config.nullbooth.num_denoising_steps
        sampler_strategy = getattr(config.nullbooth, 'sampler_strategy', None)
        timestep_mode = getattr(config.nullbooth, 'timestep_mode', 'avg')
        
        # Validate sampler strategy
        if sampler_strategy is not None:
            valid_samplers = ["uniform", "early_timestep", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 3M", "DPM++ 3M SDE Karras", "Euler", "Euler a", "UniPC", "LCM"]
            if sampler_strategy not in valid_samplers:
                if accelerator.is_main_process:
                    print(f"❌ Invalid sampler strategy: {sampler_strategy}")
                    print(f"   Valid options: {valid_samplers}")
                return

        # Validate timestep mode
        valid_timestep_modes = ['avg', 'first', 'all_steps']  # Added 'all_steps' for LCM
        if timestep_mode not in valid_timestep_modes:
            if accelerator.is_main_process:
                print(f"❌ Invalid timestep_mode: {timestep_mode}. Valid options: {valid_timestep_modes}")
            return

        if accelerator.is_main_process:
            print(f"Timestep generation strategy: {sampler_strategy if sampler_strategy else 'Uniform'}")
            print(f"Timestep mode: {timestep_mode}")

        use_alpha_naming = sampler_strategy is not None

        # Generate timesteps and mapping
        if args.phase == "2":
            # For Phase 2 only, get timesteps from completed Phase 1 cache
            if accelerator.is_main_process:
                print("Phase 2: Detecting timesteps from completed Phase 1 cache...")
                print(f"DEBUG: args.phase = {args.phase}")
                print(f"DEBUG: About to access progress_tracker.progress...")
                import time
                time.sleep(0.5)  # Small delay to ensure print buffers flush

            # Synchronize before accessing progress data
            accelerator.wait_for_everyone()

            # Get timesteps from progress tracker
            if accelerator.is_main_process:
                print(f"DEBUG: Accessing completed_timesteps_phase1...")
            completed_timesteps_phase1 = progress_tracker.progress["phase1_features"]["completed_timesteps"]
            if completed_timesteps_phase1:
                # Extract timestep values and maintain original alpha keys
                timesteps = []
                timestep_to_alpha_mapping = {}

                for timestep_key in completed_timesteps_phase1.keys():
                    if timestep_key.startswith('timestep_'):
                        timestep_val = int(timestep_key.split('_')[1])
                        timesteps.append(timestep_val)
                        timestep_to_alpha_mapping[f"timestep_{timestep_val:04d}"] = timestep_key
                    elif timestep_key.startswith('alpha_'):
                        # Parse alpha format: alpha_xx_xxxx -> xxxx
                        parts = timestep_key.split('_')
                        if len(parts) >= 3:
                            # Parse exact timestep value from format alpha_xx_xxxx
                            timestep_val = int(parts[2])
                            timesteps.append(timestep_val)
                            # Map timestep value to original alpha key
                            timestep_to_alpha_mapping[f"timestep_{timestep_val:04d}"] = timestep_key

                timesteps = sorted(timesteps, reverse=True)  # Sort in descending order

                if accelerator.is_main_process:
                    print(f"Phase 2: Found {len(timesteps)} completed timesteps from Phase 1")
                    print(f"Timesteps: {timesteps[:5]}{'...' if len(timesteps) > 5 else ''}")
                    print(f"DEBUG: Alpha mapping created:")
                    for ts_key, alpha_key in timestep_to_alpha_mapping.items():
                        print(f"  {ts_key} -> {alpha_key}")
            else:
                if accelerator.is_main_process:
                    print("❌ No completed timesteps found in Phase 1. Please run Phase 1 first.")
                return

        elif sampler_strategy is None:
            # Original uniform spacing
            if num_sample_steps >= total_timesteps:
                timesteps = list(range(total_timesteps-1, -1, -1))
            else:
                timesteps = torch.linspace(total_timesteps-1, 0, num_sample_steps, dtype=torch.long).tolist()
            timesteps = [int(t) for t in timesteps]
            timestep_to_alpha_mapping = {f"timestep_{t:04d}": f"timestep_{t:04d}" for t in timesteps}
        else:
            # Modern sampler strategy
            if accelerator.is_main_process:
                print(f"Computing timesteps using {sampler_strategy} sampler...")
                if timestep_mode == 'all_steps':
                    print("Using 'all_steps' mode - computing covariance for all inference timesteps (no sampling/averaging)")
                elif timestep_mode == 'avg':
                    print(f"Averaging timesteps across {len(prompts)} prompts...")
                else:
                    print("Using first prompt only for timestep calculation...")

            # LCM requires lcm_original_inference_steps parameter
            if sampler_strategy == "LCM":
                lcm_original_steps = getattr(config.nullbooth, 'lcm_original_inference_steps', 50)
                if accelerator.is_main_process:
                    print(f"LCM Configuration:")
                    print(f"  Inference steps: {num_sample_steps}")
                    print(f"  Original training steps: {lcm_original_steps}")

                # For LCM with 'all_steps' mode, generate timesteps directly
                timesteps = ModernSamplers.lcm_timesteps(
                    num_sample_steps, total_timesteps, lcm_original_steps
                )
                timesteps = [int(t) for t in timesteps]

            elif timestep_mode == 'all_steps':
                # For all_steps mode with other samplers: compute once, no averaging
                timesteps = ModernSamplers.get_sampler_timesteps(
                    sampler_strategy, num_sample_steps, total_timesteps
                )
                timesteps = [int(t) for t in timesteps]

            elif timestep_mode == 'avg':
                # Collect timesteps from all prompts using the specified sampler
                timestep_averager = TimestepAverager()
                for i, prompt in enumerate(prompts):
                    if accelerator.is_main_process and i % 100 == 0:
                        print(f"  Processing prompt {i+1}/{len(prompts)} for timestep calculation...")

                    prompt_timesteps = ModernSamplers.get_sampler_timesteps(
                        sampler_strategy, num_sample_steps, total_timesteps
                    )
                    timestep_averager.update(prompt_timesteps)

                # Get average timesteps
                avg_timesteps = timestep_averager.get_average_timesteps()
                timesteps = [int(t) for t in avg_timesteps]

                if accelerator.is_main_process:
                    print(f"  Completed timestep averaging across {len(prompts)} prompts")
            else:  # timestep_mode == 'first'
                # Use only the first prompt for timestep calculation
                timesteps = ModernSamplers.get_sampler_timesteps(
                    sampler_strategy, num_sample_steps, total_timesteps
                )
                timesteps = [int(t) for t in timesteps]

            # Create mapping from timestep_xxxx to alpha_xx_xxx format
            timestep_to_alpha_mapping = {}
            for i, t in enumerate(timesteps):
                alpha_name = generate_alpha_filename(i, float(t))
                timestep_to_alpha_mapping[f"timestep_{t:04d}"] = alpha_name

            if accelerator.is_main_process:
                print(f"Timesteps computed: {timesteps[:5]}{'...' if len(timesteps) > 5 else ''}")
                print(f"Using alpha naming scheme for {len(timesteps)} timesteps")

        timesteps = [int(t) for t in timesteps]

        try:
            # Phase 1: Feature Collection
            if args.phase in ["1", "both"]:
                phase1_success = run_phase1_feature_collection(
                    accelerator, config, run_dir, progress_tracker, prompts, timesteps,
                    timestep_to_alpha_mapping, use_alpha_naming
                )
                if not phase1_success:
                    if accelerator.is_main_process:
                        print("❌ Phase 1 failed. Aborting.")
                    return

            # Phase 2: Covariance Matrix Computation
            if args.phase in ["2", "both"]:
                phase2_success = run_phase2_covariance_computation(
                    accelerator, config, run_dir, progress_tracker, timesteps, use_alpha_naming
                )
                if not phase2_success:
                    if accelerator.is_main_process:
                        print("❌ Phase 2 failed. Aborting.")
                    return

            # Final summary
            if accelerator.is_main_process:
                print(f"\n✅ Parallel build_cov completed successfully!")
                print(f"Results saved to: {run_dir}")
                print(f"Total timesteps processed: {len(timesteps)}")
                print(f"Total prompts processed: {len(prompts)}")
                print(f"Speedup achieved: ~{accelerator.num_processes}x with {accelerator.num_processes} GPUs")

        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Error during execution: {e}")
                import traceback
                traceback.print_exc()
        finally:
            # Proper cleanup of distributed process group with robust error handling
            cleanup_success = False
            max_cleanup_retries = 3

            for cleanup_retry in range(max_cleanup_retries):
                try:
                    # Final synchronization with timeout
                    if accelerator.is_main_process:
                        print(f"🧹 Final cleanup synchronization (attempt {cleanup_retry + 1}/{max_cleanup_retries})...")

                    accelerator.wait_for_everyone()
                    cleanup_success = True

                    if accelerator.is_main_process:
                        print("🧹 Cleaning up distributed process group...")

                    accelerator.free_memory()
                    break

                except Exception as cleanup_sync_error:
                    print(f"⚠️  Process {accelerator.process_index}: Cleanup sync attempt {cleanup_retry + 1} failed: {cleanup_sync_error}")
                    if cleanup_retry < max_cleanup_retries - 1:
                        import time
                        time.sleep(2.0)
                    else:
                        print(f"❌ Process {accelerator.process_index}: Failed final sync, proceeding with individual cleanup")

            # Individual cleanup regardless of sync success
            try:
                # Clean up torch distributed if it was initialized
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as dist_cleanup_error:
                print(f"⚠️  Process {accelerator.process_index}: Distributed cleanup warning: {dist_cleanup_error}")

            # Clear CUDA cache
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as cuda_cleanup_error:
                print(f"⚠️  Process {accelerator.process_index}: CUDA cleanup warning: {cuda_cleanup_error}")

            if accelerator.is_main_process:
                if cleanup_success:
                    print("✅ Cleanup completed successfully")
                else:
                    print("⚠️  Cleanup completed with warnings")


def run_phase1_feature_collection(accelerator, config, run_dir, progress_tracker, prompts, timesteps, timestep_to_alpha_mapping=None, use_alpha_naming=False):
    """Phase 1: Parallel feature collection with timestep-level progress tracking."""

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"PHASE 1: FEATURE COLLECTION")
        print(f"{'='*60}")

    # Update progress
    progress_tracker.update_phase1_status("running", total_prompts=len(prompts))

    # Check which timesteps need processing
    # For Phase 1, we get alpha keys from existing progress if available
    all_alpha_keys = None
    if use_alpha_naming:
        # Get existing alpha keys from progress tracker
        existing_completed = progress_tracker.progress.get("phase1_features", {}).get("completed_timesteps", {})
        all_alpha_keys = [key for key in existing_completed.keys() if key.startswith('alpha_')]

    remaining_timesteps = progress_tracker.get_remaining_timesteps_phase1(timesteps, use_alpha_naming, all_alpha_keys)

    if not remaining_timesteps:
        if accelerator.is_main_process:
            print("✅ All timesteps already completed in Phase 1. Skipping feature collection.")
        progress_tracker.update_phase1_status("completed")
        return True

    if accelerator.is_main_process:
        print(f"Processing {len(remaining_timesteps)}/{len(timesteps)} remaining timesteps")
        print(f"Skipping {len(timesteps) - len(remaining_timesteps)} completed timesteps")

    # Load models
    if accelerator.is_main_process:
        print("Loading diffusion models...")

    tokenizer, text_encoder, unet, pipeline = load_diffusion_models(config, accelerator)

    # Initialize feature collector
    collector = ParallelAttentionFeatureCollector(config, accelerator, cache_dir=run_dir)
    collector.register_hooks(unet)

    if accelerator.is_main_process:
        print(f"Total {len(collector.hooks)} cross-attention hooks registered.")

    # Distribute prompts across processes
    process_prompts = distribute_prompts(prompts, accelerator)
    prompt_batch_size = max(1, getattr(config.nullbooth, 'prompt_batch_size', 1))
    async_writer = AsyncWriter(getattr(config.nullbooth, 'async_io_workers', 0))
    prompt_embedding_cache: Dict[str, torch.Tensor] = {}
    unet_module = unet.module if hasattr(unet, 'module') else unet
    text_encoder_module = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
    sample_param = next(iter(unet_module.parameters()), None)
    unet_dtype = sample_param.dtype if sample_param is not None else torch.float32
    latent_channels = unet_module.config.in_channels
    latent_height = config.resolution // 8
    latent_width = config.resolution // 8

    try:
        if accelerator.is_main_process:
            print("\n📊 Starting parallel feature collection...")
            print(f"   Total timesteps: {len(remaining_timesteps)}")
            print(f"   Prompts per process: {len(process_prompts)}")
            print(f"   Batch size: {prompt_batch_size}\n")

        with torch.no_grad():
            # Create main progress bar for timesteps (only on main process)
            timestep_pbar = tqdm(
                remaining_timesteps,
                desc=f"Process {accelerator.process_index}: Phase 1 Timesteps",
                position=accelerator.process_index,
                disable=not accelerator.is_local_main_process,
                leave=True,
                colour='green' if accelerator.is_main_process else 'blue'
            )

            for timestep_idx, target_timestep in enumerate(timestep_pbar):
                # Update timestep progress description
                if accelerator.is_local_main_process:
                    timestep_pbar.set_description(f"Process {accelerator.process_index}: Timestep {target_timestep}")

                # Check for partially completed timestep
                timestep_key_check = timestep_to_alpha_mapping.get(f"timestep_{target_timestep:04d}") if use_alpha_naming and timestep_to_alpha_mapping else f"timestep_{target_timestep:04d}"
                completed_prompts_for_timestep = progress_tracker.get_completed_prompts_for_timestep(target_timestep, timestep_key_check)

                collector.set_current_timestep(target_timestep)
                timestep_success = True
                timestep_error_count = 0
                max_errors_per_timestep = max(1, min(10, len(process_prompts) // 4))

                # Filter out already completed prompts for this timestep
                remaining_prompts = [(idx, prompt) for idx, prompt in process_prompts
                                   if idx not in completed_prompts_for_timestep]

                if not remaining_prompts:
                    if accelerator.is_local_main_process:
                        timestep_pbar.set_postfix_str(f"✅ Already completed")
                    continue

                total_prompts = len(remaining_prompts)
                batch_start = 0
                batch_step = 0

                # Create nested progress bar for prompts within this timestep
                prompt_pbar = tqdm(
                    total=total_prompts,
                    desc=f"  └─ Prompts for timestep {target_timestep} (resuming {len(completed_prompts_for_timestep)} completed)",
                    position=accelerator.process_index + accelerator.num_processes,
                    disable=not accelerator.is_local_main_process,
                    leave=False,
                    colour='cyan'
                )

                while batch_start < total_prompts:
                    current_batch_size = min(prompt_batch_size, total_prompts - batch_start)
                    batch_processed = False

                    while current_batch_size >= 1 and not batch_processed:
                        batch_entries = remaining_prompts[batch_start:batch_start + current_batch_size]
                        batch_indices = [idx for idx, _ in batch_entries]
                        batch_prompts = [prompt for _, prompt in batch_entries]

                        try:
                            uncached_prompts = [p for p in batch_prompts if p not in prompt_embedding_cache]
                            if uncached_prompts:
                                text_inputs = tokenizer(
                                    uncached_prompts,
                                    padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                                input_ids = text_inputs.input_ids.to(accelerator.device)
                                new_embeddings = text_encoder_module(input_ids)[0]
                                for prompt_text, embedding in zip(uncached_prompts, new_embeddings):
                                    prompt_embedding_cache[prompt_text] = embedding.detach().cpu().to(torch.float16)

                            embedding_tensors = [
                                prompt_embedding_cache[prompt_text].to(device=accelerator.device, dtype=unet_dtype)
                                for prompt_text in batch_prompts
                            ]
                            batch_embeddings = torch.stack(embedding_tensors, dim=0)
                            collector.set_current_encoder_hidden_states(batch_embeddings)

                            base_latents = torch.randn(
                                (len(batch_entries), latent_channels, latent_height, latent_width),
                                device=accelerator.device,
                                dtype=unet_dtype,
                            )
                            noise = torch.randn_like(base_latents)
                            t = torch.full((len(batch_entries),), target_timestep, dtype=torch.long, device=accelerator.device)
                            noisy_latents = pipeline.scheduler.add_noise(base_latents.to(torch.float32), noise.to(torch.float32), t)
                            latent_model_input = pipeline.scheduler.scale_model_input(noisy_latents, t).to(unet_dtype)

                            _ = unet_module(
                                latent_model_input,
                                t,
                                encoder_hidden_states=batch_embeddings,
                                return_dict=False,
                            )[0]

                            collector.save_batch_features_to_disk(batch_indices, timestep_to_alpha_mapping, async_writer)

                            # Mark individual prompts as completed after successful save
                            if accelerator.is_main_process:
                                for prompt_idx in batch_indices:
                                    progress_tracker.mark_prompt_completed_phase1(target_timestep, prompt_idx, timestep_key_check)

                            if (batch_step + 1) % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Update prompt progress bar
                            prompt_pbar.update(current_batch_size)

                            batch_processed = True
                        except (RuntimeError, MemoryError) as e:
                            message = str(e).lower()
                            is_oom = any(token in message for token in ["out of memory", "cublas_status_alloc_failed", "cuda error"])
                            if is_oom and current_batch_size > 1:
                                if accelerator.is_local_main_process:
                                    timestep_pbar.set_postfix_str(f"OOM! Reducing batch: {current_batch_size}→{current_batch_size//2}")
                                current_batch_size = max(1, current_batch_size // 2)
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                async_writer.wait()
                                collector.feature_cache.clear()
                                collector.attention_maps_cache.clear()
                                continue

                            timestep_error_count += 1
                            if accelerator.is_local_main_process:
                                timestep_pbar.set_postfix_str(f"Errors: {timestep_error_count}")

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            collector.feature_cache.clear()
                            collector.attention_maps_cache.clear()

                            if timestep_error_count > max_errors_per_timestep:
                                timestep_success = False
                                if accelerator.is_local_main_process:
                                    timestep_pbar.set_postfix_str(f"Failed! Too many errors: {timestep_error_count}")
                                current_batch_size = 0
                                break

                            batch_processed = True

                    if not batch_processed:
                        break

                    batch_start += current_batch_size if current_batch_size else prompt_batch_size
                    batch_step += 1
                    async_writer.wait()

                # Close prompt progress bar
                prompt_pbar.close()

                # Synchronize across all processes before marking completion
                try:
                    accelerator.wait_for_everyone()
                except Exception as sync_error:
                    if accelerator.is_local_main_process:
                        timestep_pbar.set_postfix_str(f"Sync error!")
                    timestep_success = False

                # Only mark as completed if this process succeeded and we're main process
                if accelerator.is_main_process and timestep_success:
                    # Use appropriate timestep key based on naming strategy
                    if use_alpha_naming and timestep_to_alpha_mapping:
                        timestep_key = timestep_to_alpha_mapping.get(f"timestep_{target_timestep:04d}")
                    else:
                        timestep_key = f"timestep_{target_timestep:04d}"
                    progress_tracker.mark_timestep_completed_phase1(target_timestep, len(prompts), timestep_key)
                    timestep_pbar.set_postfix_str(f"✅ Completed")
                elif accelerator.is_main_process:
                    timestep_pbar.set_postfix_str(f"❌ Failed")

            # Close main progress bar
            timestep_pbar.close()

        # Final status update
        progress_tracker.update_phase1_status("completed", completed_prompts=len(prompts))

        if accelerator.is_main_process:
            print(f"\n✅ Phase 1 completed: All {len(remaining_timesteps)} timesteps processed")
            print(f"   Features saved to: {run_dir / 'cache'}")

        return True
        
    except Exception as e:
        progress_tracker.update_phase1_status("failed")
        if accelerator.is_main_process:
            print(f"❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
        return False
    finally:
        async_writer.shutdown()
        # Clean up hooks
        collector.remove_hooks()


def run_phase2_covariance_computation(accelerator, config, run_dir, progress_tracker, timesteps, use_alpha_naming=False):
    """Phase 2: Parallel streaming covariance matrix computation with timestep-level progress tracking."""

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"PHASE 2: PARALLEL COVARIANCE MATRIX COMPUTATION")
        print(f"{'='*60}")
        print(f"DEBUG: Starting Phase 2 with {len(timesteps)} timesteps")
        print(f"DEBUG: Use alpha naming: {use_alpha_naming}")

    # Update progress
    progress_tracker.update_phase2_status("running", total_timesteps=len(timesteps))

    # Create covariance computer and get cache structure first (needed for timestep resolution)
    if accelerator.is_main_process:
        print("DEBUG: Creating ParallelCovarianceMatrixComputer...")
    computer = ParallelCovarianceMatrixComputer(config, accelerator)

    # Get cache structure (all processes need this)
    if accelerator.is_main_process:
        print("DEBUG: Gathering cache structure from all processes...")
        print(f"DEBUG: Run directory: {run_dir}")
    cache_structure = computer.gather_all_features_from_processes(run_dir)

    if accelerator.is_main_process:
        print(f"DEBUG: Cache structure gathered successfully")

    # Properly gather cache_structure from all processes
    # All processes must participate in gather_object with the same operation
    all_cache_structures = gather_object([cache_structure])

    # Only the main process has the actual cache_structure, others have empty dict
    # Use the first non-empty cache_structure (from main process)
    if accelerator.is_main_process:
        # Main process should have the actual data in its own structure
        final_cache_structure = cache_structure if cache_structure else {}
    else:
        # Non-main processes get the structure from the gathered data
        final_cache_structure = {}
        for cs in all_cache_structures:
            if cs and cs.get('cache_dirs'):
                final_cache_structure = cs
                break

    cache_structure = final_cache_structure

    # Check which timesteps need processing
    all_alpha_keys = cache_structure.get('timestep_keys', []) if use_alpha_naming else None
    remaining_timesteps = progress_tracker.get_remaining_timesteps_phase2(timesteps, use_alpha_naming, all_alpha_keys)

    if not remaining_timesteps:
        if accelerator.is_main_process:
            print("✅ All timesteps already completed in Phase 2. Skipping covariance computation.")
        progress_tracker.update_phase2_status("completed")
        return True

    if accelerator.is_main_process:
        print(f"\n📊 Starting parallel covariance computation...")
        print(f"   Total timesteps: {len(remaining_timesteps)}/{len(timesteps)}")
        print(f"   Skipping: {len(timesteps) - len(remaining_timesteps)} completed")
        print(f"   Using: {accelerator.num_processes} GPUs\n")

    # Distribute timesteps evenly across all processes
    # Use simple round-robin distribution to ensure all timesteps are processed
    timesteps_per_process = ceil(len(remaining_timesteps) / accelerator.num_processes)
    start_idx = accelerator.process_index * timesteps_per_process
    end_idx = min(start_idx + timesteps_per_process, len(remaining_timesteps))
    process_timesteps = remaining_timesteps[start_idx:end_idx]

    if accelerator.is_main_process:
        print("Timestep distribution across processes:")
        for i in range(accelerator.num_processes):
            proc_start = i * timesteps_per_process
            proc_end = min(proc_start + timesteps_per_process, len(remaining_timesteps))
            proc_timesteps = remaining_timesteps[proc_start:proc_end] if proc_end > proc_start else []
            print(f"  Process {i}: {len(proc_timesteps)} timesteps {proc_timesteps[:3]}{'...' if len(proc_timesteps) > 3 else ''}")

    print(f"Process {accelerator.process_index}: Processing {len(process_timesteps)} timesteps")

    try:

        if not cache_structure or not cache_structure.get('cache_dirs'):
            if accelerator.is_main_process:
                print("❌ No cache directories found. Phase 1 may not have completed successfully.")
            return False

        if accelerator.is_main_process:
            print(f"Found {len(cache_structure['timestep_keys'])} timesteps in cache")
            print(f"Found {len(cache_structure['layer_feature_pairs'])} layer/feature combinations\n")

        # Each process computes covariance matrices for its assigned timesteps
        process_cov_matrices = {}
        process_failed_timesteps = []

        # Create progress bar for this process's timesteps
        process_pbar = tqdm(
            process_timesteps,
            desc=f"GPU {accelerator.process_index}: Phase 2 Timesteps",
            position=accelerator.process_index,
            disable=not accelerator.is_local_main_process,
            leave=True,
            colour='green' if accelerator.is_main_process else 'cyan'
        )

        for timestep in process_pbar:
            # Update progress description
            if accelerator.is_local_main_process:
                process_pbar.set_description(f"GPU {accelerator.process_index}: Timestep {timestep}")

            # Use correct timestep key based on naming strategy
            if use_alpha_naming:
                # Find all alpha keys that match this timestep value
                matching_alpha_keys = []
                for alpha_key in cache_structure['timestep_keys']:
                    if alpha_key.startswith('alpha_'):
                        parts = alpha_key.split('_')
                        if len(parts) >= 3:
                            # Parse exact timestep value from format alpha_xx_xxxx
                            timestep_value = int(parts[2])
                            if timestep_value == timestep:
                                matching_alpha_keys.append(alpha_key)

                if matching_alpha_keys:
                    # Use the first matching alpha key for this timestep
                    timestep_key = matching_alpha_keys[0]
                else:
                    timestep_key = f"timestep_{timestep:04d}"
            else:
                timestep_key = f"timestep_{timestep:04d}"

            if timestep_key not in cache_structure['timestep_keys']:
                if accelerator.is_local_main_process:
                    process_pbar.set_postfix_str(f"⚠️ Not in cache")
                process_failed_timesteps.append(timestep)
                continue

            try:
                # Create timestep-specific structure
                timestep_structure = {
                    'cache_dirs': cache_structure['cache_dirs'],
                    'timestep_keys': [timestep_key],
                    'layer_feature_pairs': cache_structure['layer_feature_pairs']
                }

                # Compute covariance matrices for this timestep with memory management
                cov_matrices = computer.compute_covariance_matrices_from_merged_features(timestep_structure)

                if cov_matrices and timestep_key in cov_matrices:
                    # Instead of storing in memory, save directly to disk
                    # This avoids gather_object failures with large matrices
                    only_cov_flag = getattr(config.nullbooth, 'only_covariance_matrix', False)
                    save_single_timestep_covariance(
                        {timestep_key: cov_matrices[timestep_key]},
                        run_dir,
                        only_covariance=only_cov_flag
                    )

                    # Store only metadata for gathering
                    process_cov_matrices[timestep_key] = {
                        'timestep': timestep,
                        'timestep_key': timestep_key,
                        'saved': True,
                        'num_layers': len(cov_matrices[timestep_key])
                    }

                    if accelerator.is_local_main_process:
                        process_pbar.set_postfix_str(f"✅ Saved {len(cov_matrices[timestep_key])} layers")

                    # Clear memory after each timestep to prevent accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                else:
                    if accelerator.is_local_main_process:
                        process_pbar.set_postfix_str(f"❌ Failed")
                    process_failed_timesteps.append(timestep)

            except Exception as e:
                if accelerator.is_local_main_process:
                    process_pbar.set_postfix_str(f"❌ Error: {str(e)[:20]}")
                process_failed_timesteps.append(timestep)

                # Clear memory and try to recover from OOM or other errors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection to free memory
                import gc
                gc.collect()

                continue

        # Close progress bar
        process_pbar.close()

        # Wait for all processes to finish their computations with robust error handling
        sync_success = True
        max_sync_retries = 3
        sync_retry_delay = 5.0

        for retry in range(max_sync_retries):
            try:
                if accelerator.is_main_process:
                    print(f"Synchronizing all processes (attempt {retry + 1}/{max_sync_retries})...")

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    print("✅ All processes synchronized successfully")
                break

            except Exception as sync_error:
                print(f"⚠️  Process {accelerator.process_index}: Synchronization attempt {retry + 1} failed: {sync_error}")

                if retry < max_sync_retries - 1:
                    print(f"   Retrying synchronization in {sync_retry_delay} seconds...")
                    import time
                    time.sleep(sync_retry_delay)
                    sync_retry_delay *= 1.5  # Exponential backoff
                else:
                    print(f"❌ Process {accelerator.process_index}: Failed to synchronize after {max_sync_retries} attempts")
                    print(f"   Proceeding without synchronization - results may be incomplete")
                    sync_success = False

        # Report failed timesteps for visibility
        if process_failed_timesteps:
            print(f"Process {accelerator.process_index}: Failed timesteps: {process_failed_timesteps}")

        # Gather metadata only (not the large matrices, as they're already saved to disk)
        if accelerator.is_main_process:
            print("\nGathering metadata from all processes...")

        try:
            # Only gather lightweight metadata, not the actual matrices
            # Use accelerator's gather method which is more robust
            all_metadata = gather_object([process_cov_matrices])
            all_failed_timesteps = gather_object([process_failed_timesteps])

            # Debug output
            if accelerator.is_main_process:
                print(f"DEBUG: Gathered metadata type: {type(all_metadata)}, length: {len(all_metadata) if isinstance(all_metadata, list) else 'N/A'}")
                print(f"DEBUG: First metadata entry: {all_metadata[0] if isinstance(all_metadata, list) and len(all_metadata) > 0 else 'None'}")

        except Exception as gather_error:
            print(f"❌ Process {accelerator.process_index}: Failed to gather metadata: {gather_error}")
            if accelerator.is_main_process:
                print("   Attempting fallback result collection...")
                # Fallback: use only main process results
                all_metadata = [process_cov_matrices] if process_cov_matrices else [{}]
                all_failed_timesteps = [process_failed_timesteps] if process_failed_timesteps else [[]]

        # Update progress tracking (only main process)
        if accelerator.is_main_process:
            # Count completed timesteps from metadata
            completed_timesteps = {}
            total_computed = 0

            # all_metadata should be a list with one entry per process
            if isinstance(all_metadata, list):
                # Multi-process case - merge metadata from all processes
                for process_metadata in all_metadata:
                    if isinstance(process_metadata, dict) and process_metadata:
                        for key, value in process_metadata.items():
                            if key not in completed_timesteps:  # Avoid duplicates
                                completed_timesteps[key] = value
                        total_computed += len(process_metadata)

                print(f"DEBUG: Merged metadata from {len(all_metadata)} processes")
                print(f"DEBUG: Total completed timesteps: {len(completed_timesteps)}")
                print(f"DEBUG: Completed timestep keys: {list(completed_timesteps.keys())}")
            else:
                print(f"⚠️  Unexpected metadata format: {type(all_metadata)}")
                # Try to use it anyway if it's a dict
                if isinstance(all_metadata, dict):
                    completed_timesteps = all_metadata
                    total_computed = len(all_metadata)

            # Mark completed timesteps in progress tracker
            for timestep_key, metadata in completed_timesteps.items():
                if metadata.get('saved', False):
                    if use_alpha_naming and timestep_key.startswith('alpha_'):
                        # For alpha naming, extract timestep from key
                        parts = timestep_key.split('_')
                        if len(parts) >= 3:
                            timestep = int(parts[2])
                        else:
                            timestep = 0
                        progress_tracker.mark_timestep_completed_phase2(timestep, timestep_key)
                    else:
                        # For timestep naming, extract from key
                        timestep = int(timestep_key.split('_')[1])
                        progress_tracker.mark_timestep_completed_phase2(timestep)

            print(f"✅ Successfully saved {total_computed} timesteps to disk")
            print(f"   Covariance matrices location: {run_dir / 'covariance_matrices'}")

            # Final status update
            progress_tracker.update_phase2_status("completed")

            print(f"✅ Phase 2 completed: {total_computed} timesteps processed across {accelerator.num_processes} GPUs")
            print(f"   Speedup: ~{accelerator.num_processes}x for covariance computation")
        
        return True
        
    except Exception as e:
        progress_tracker.update_phase2_status("failed")
        if accelerator.is_main_process:
            print(f"❌ Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
        return False


def load_diffusion_models(config, accelerator):
    """Load and prepare diffusion models for parallel processing (SD or LCM)."""
    # Check if we're loading an LCM model
    model_type = getattr(config, 'model_type', 'SD')  # Default to Stable Diffusion

    if model_type == 'LCM':
        # Load LCM-specific pipeline
        from diffusers import LatentConsistencyModelPipeline, LCMScheduler

        if accelerator.is_main_process:
            print(f"Loading LCM model from: {config.pretrained_model_name_or_path}")

        # Load LCM pipeline directly
        pipeline = LatentConsistencyModelPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16 if getattr(config, 'use_fp16', True) else torch.float32
        )

        # Extract components
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        unet = pipeline.unet
        vae = pipeline.vae

        # Move models to device using accelerator
        text_encoder = accelerator.prepare(text_encoder)
        unet = accelerator.prepare(unet)
        vae = accelerator.prepare(vae)

        # Update pipeline components
        pipeline.text_encoder = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
        pipeline.unet = unet.module if hasattr(unet, 'module') else unet
        pipeline.vae = vae.module if hasattr(vae, 'module') else vae

        return tokenizer, text_encoder, unet, pipeline

    else:
        # Original Stable Diffusion loading
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
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32
        )

        return tokenizer, text_encoder, unet, pipeline


if __name__ == "__main__":
    main()
