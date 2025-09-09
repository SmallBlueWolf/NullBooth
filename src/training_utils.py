import gc
import logging
import math
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .dataset import DreamBoothDataset, PromptDataset, collate_fn

logger = get_logger(__name__)


def b2mb(x):
    """Converting Bytes to Megabytes"""
    return int(x / 2**20)


class TorchTracemalloc:
    """Context manager to track peak memory usage of the process"""
    def __enter__(self):
        gc.collect()
        self.device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        self.device_module = getattr(torch, self.device_type, torch.cuda)
        self.device_module.empty_cache()
        self.device_module.reset_peak_memory_stats()  # reset the peak gauge to zero
        self.begin = self.device_module.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        self.device_module.empty_cache()
        self.end = self.device_module.memory_allocated()
        self.peak = self.device_module.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


def setup_logging(accelerator):
    """Setup logging configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        import datasets
        import transformers
        import diffusers
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        import datasets
        import transformers
        import diffusers
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def setup_accelerator(config):
    """Setup accelerator for training"""
    logging_dir = Path(config.output_dir, config.logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_dir=logging_dir,
    )
    
    if config.report_to == "wandb":
        import wandb
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.wandb_project_name)
    
    return accelerator


def validate_config(config):
    """Validate configuration parameters"""
    if config.train_text_encoder and config.gradient_accumulation_steps > 1:
        # This check is from the original code
        warnings.warn(
            "Gradient accumulation is not fully supported when training the text encoder. "
            "This feature will be supported in the future."
        )

    if config.with_prior_preservation:
        if config.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if config.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if config.class_data_dir is not None:
            warnings.warn("You need not use class_data_dir without with_prior_preservation.")
        if config.class_prompt is not None:
            warnings.warn("You need not use class_prompt without with_prior_preservation.")


def load_tokenizer(config):
    """Load tokenizer"""
    if config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.revision, use_fast=False)
    elif config.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.revision,
            use_fast=False,
        )
    else:
        raise ValueError("Must specify either tokenizer_name or pretrained_model_name_or_path")
    
    return tokenizer


def create_datasets_and_dataloaders(config, tokenizer):
    """Create training dataset and dataloader"""
    train_dataset = DreamBoothDataset(
        instance_data_root=config.instance_data_dir,
        instance_prompt=config.instance_prompt,
        class_data_root=config.class_data_dir if config.with_prior_preservation else None,
        class_prompt=config.class_prompt,
        tokenizer=tokenizer,
        size=config.resolution,
        center_crop=config.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, config.with_prior_preservation),
        num_workers=config.num_dataloader_workers,
    )
    
    return train_dataset, train_dataloader


def setup_training_params(config, train_dataloader, accelerator):
    """Setup training parameters and learning rate scheduler"""
    # Calculate training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )
    
    return overrode_max_train_steps, num_update_steps_per_epoch


def setup_lr_scheduler(config, optimizer):
    """Setup learning rate scheduler"""
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    return lr_scheduler


def setup_hub_and_output_dir(config, accelerator):
    """Setup hub and output directory"""
    if accelerator.is_main_process:
        if config.push_to_hub:
            api = HfApi(token=config.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = config.hub_model_id
            if repo_name is None:
                repo_name = Path(config.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(config.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore.read():
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore.read():
                    gitignore.write("epoch_*\n")
            
            return api, repo_id
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
    
    return None, None


def enable_optimizations(unet, text_encoder, config, accelerator):
    """Enable various training optimizations"""
    if config.enable_xformers_memory_efficient_attention:
        if accelerator.device.type == "xpu":
            logger.warn("XPU hasn't support xformers yet, ignore it.")
        elif is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train_text_encoder and not config.lora.use_lora:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True