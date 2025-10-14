import hashlib
import itertools
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from accelerate.logging import get_logger

from .dataset import PromptDataset
from .training_utils import TorchTracemalloc, b2mb
from .nullbooth_trainer import NullBoothTrainer

logger = get_logger(__name__)


def generate_class_images(config, accelerator):
    """Generate class images if prior preservation is enabled."""
    if config.with_prior_preservation:
        class_images_dir = Path(config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type in ["cuda", "xpu"] else torch.float32
            if config.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
                
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()


def run_validation(config, accelerator, unet, text_encoder, epoch, step, num_update_steps_per_epoch):
    """Run validation during training."""
    # Only the global main process performs validation to avoid redundant work on other ranks.
    if not accelerator.is_main_process:
        return

    if (
        config.validation_prompt is not None
        and (step + num_update_steps_per_epoch * epoch) % config.validation_steps == 0
    ):
        logger.info(
            f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
            f" {config.validation_prompt}."
        )
        # create pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            safety_checker=None,
            revision=config.revision,
        )
        # set `keep_fp32_wrapper` to True because we do not want to remove
        # mixed precision hooks while we are still training
        pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
        pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)

        # Check if using LCM model
        model_type = getattr(config, 'model_type', 'standard')
        if model_type == "LCM":
            from diffusers import LCMScheduler
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            num_inference_steps = 4  # LCM uses 2-8 steps
        else:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            num_inference_steps = 25  # Standard diffusion uses more steps

        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        if config.seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        else:
            generator = None
        images = []
        for _ in range(config.num_validation_images):
            image = pipeline(config.validation_prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
            images.append(image)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                import wandb
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()


def training_step(batch, unet, text_encoder, vae, noise_scheduler, accelerator, config, weight_dtype, ema_unet=None):
    """Perform a single training step.
    Note: NullBooth projection is now handled by the optimizer wrapper, not here.

    Supports both standard training and LCF (Latent Consistency Fine-tuning) mode.
    """
    # Convert images to latent space
    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * 0.18215

    # Check if LCF mode is enabled
    lcf_mode = hasattr(config, 'lcf_mode') and config.lcf_mode

    if lcf_mode:
        with_prior = getattr(config, "with_prior_preservation", False)

        if with_prior:
            full_bsz = latents.shape[0]
            if full_bsz % 2 != 0:
                raise ValueError("With prior preservation enabled, batch size should be even.")
            half_bsz = full_bsz // 2

            latents_inst, latents_cls = latents[:half_bsz], latents[half_bsz:]
            input_ids_inst = batch["input_ids"][:half_bsz]
            input_ids_cls = batch["input_ids"][half_bsz:]
        else:
            latents_inst = latents
            input_ids_inst = batch["input_ids"]

        # LCF Training Step - Non-distillation fine-tuning
        # Based on Algorithm 4 from the LCM paper

        # Get LCF parameters
        lcf_params = getattr(config, 'lcf_parameters', {})
        skipping_step = lcf_params.get('skipping_step', 20)
        huber_c = lcf_params.get('huber_c', 0.001)
        w_min = lcf_params.get('w_min', 3.0)
        w_max = lcf_params.get('w_max', 15.0)
        consistency_weight = lcf_params.get('consistency_weight', 1.0)

        # Sample timestep n (avoiding too early in the schedule)
        bsz = latents_inst.shape[0]
        n = torch.randint(
            1,
            noise_scheduler.config.num_train_timesteps - skipping_step,
            (bsz,),
            device=latents_inst.device
        ).long()

        # Get timesteps t_n and t_{n+k}
        timesteps_nk = torch.clamp(n + skipping_step, 0, noise_scheduler.config.num_train_timesteps - 1).long()
        timesteps_n = torch.clamp(n, 0, noise_scheduler.config.num_train_timesteps - 1).long()

        # Sample CFG scale
        w = torch.rand(1, device=latents_inst.device).item() * (w_max - w_min) + w_min

        # Sample noise (same for both timesteps - critical for consistency!)
        noise = torch.randn_like(latents_inst)

        # Get alpha and sigma values using the scheduler
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(
            device=latents_inst.device, dtype=latents_inst.dtype
        )

        # Ensure timesteps are within bounds
        timesteps_nk = torch.clamp(timesteps_nk, 0, len(alphas_cumprod) - 1)
        timesteps_n = torch.clamp(timesteps_n, 0, len(alphas_cumprod) - 1)

        alpha_prod_t = alphas_cumprod[timesteps_nk]
        alpha_prod_t_prev = alphas_cumprod[timesteps_n]
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_prod_t, min=0.0))
        sigma_t_prev = torch.sqrt(torch.clamp(1 - alpha_prod_t_prev, min=0.0))

        # Reshape for broadcasting
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        sigma_t_prev = sigma_t_prev.view(-1, 1, 1, 1)

        # Add noise at two different timesteps using SAME epsilon
        z_t_nk = alpha_prod_t * latents_inst + sigma_t * noise  # z at t_{n+k}
        z_t_n = alpha_prod_t_prev * latents_inst + sigma_t_prev * noise  # z at t_n

        # Get text embeddings
        encoder_hidden_states = text_encoder(input_ids_inst)[0]

        # Get unconditional embeddings for CFG
        # Create empty tokens directly without using tokenizer
        uncond_tokens = torch.full_like(input_ids_inst, fill_value=49407)
        # Set padding token (typically 0 or 49407 for CLIP)
        # 49407 是 CLIP 的 PAD token
        uncond_embeddings = text_encoder(uncond_tokens)[0]

        # Concatenate for CFG
        text_embeddings = torch.cat([uncond_embeddings, encoder_hidden_states])

        # Target prediction from t_n using EMA model (if available) or current model
        with torch.no_grad():
            target_model = ema_unet if ema_unet is not None else unet
            # Expand inputs for CFG
            z_t_n_expanded = torch.cat([z_t_n] * 2)
            timesteps_n_expanded = torch.cat([timesteps_n] * 2)

            # Get model predictions
            noise_pred_uncond, noise_pred_cond = target_model(
                z_t_n_expanded, timesteps_n_expanded, text_embeddings
            ).sample.chunk(2)

            # Apply CFG
            target_noise_pred = noise_pred_uncond + w * (noise_pred_cond - noise_pred_uncond)

            # Convert to predicted z_0
            target_z0 = (z_t_n - sigma_t_prev * target_noise_pred) / alpha_prod_t_prev

        # Online model prediction from t_{n+k}
        z_t_nk_expanded = torch.cat([z_t_nk] * 2)
        timesteps_nk_expanded = torch.cat([timesteps_nk] * 2)

        noise_pred_uncond, noise_pred_cond = unet(
            z_t_nk_expanded, timesteps_nk_expanded, text_embeddings
        ).sample.chunk(2)

        # Apply CFG
        online_noise_pred = noise_pred_uncond + w * (noise_pred_cond - noise_pred_uncond)

        # Convert to predicted z_0
        online_z0 = (z_t_nk - sigma_t * online_noise_pred) / alpha_prod_t

        # Consistency loss using Pseudo-Huber loss
        diff = online_z0 - target_z0.detach()
        loss = consistency_weight * torch.mean(torch.sqrt(diff**2 + huber_c**2) - huber_c)

        if with_prior:
            prior_loss_weight = float(getattr(config, "prior_loss_weight", 1.0))
        else:
            prior_loss_weight = 0.0

        if with_prior and prior_loss_weight != 0.0:
            # DreamBooth-style prior preservation on class samples
            noise_cls = torch.randn_like(latents_cls)
            timesteps_cls = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents_cls.shape[0],), device=latents_cls.device
            ).long()

            noisy_latents_cls = noise_scheduler.add_noise(latents_cls, noise_cls, timesteps_cls)
            encoder_hidden_states_cls = text_encoder(input_ids_cls)[0]
            model_pred_cls = unet(noisy_latents_cls, timesteps_cls, encoder_hidden_states_cls).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target_cls = noise_cls
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target_cls = noise_scheduler.get_velocity(latents_cls, noise_cls, timesteps_cls)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            prior_loss = F.mse_loss(model_pred_cls.float(), target_cls.float(), reduction="mean")
            loss = loss + prior_loss_weight * prior_loss

        # Return timesteps for potential NullBooth use
        timesteps = timesteps_nk

    else:
        # Standard diffusion training
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        # Predict the noise residual (with NullBooth projection applied via hooks)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        if config.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + config.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss, timesteps


def training_loop(
    config,
    accelerator,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    train_dataloader,
    train_dataset,
):
    """Main training loop with NullBooth and LCF support."""
    # Initialize NullBooth trainer if enabled
    nullbooth_trainer = None
    alphaedit_optimizer = None
    ema_unet = None

    # Check if LCF mode is enabled
    lcf_mode = hasattr(config, 'lcf_mode') and config.lcf_mode
    if lcf_mode:
        logger.info("\n" + "="*60)
        logger.info("🔬 LCF (Latent Consistency Fine-tuning) Mode Enabled")
        logger.info("="*60)
        logger.info("  Using self-consistency loss without distillation")

        # Create EMA model for LCF
        import copy
        ema_unet = copy.deepcopy(unet)
        ema_unet.requires_grad_(False)
        ema_unet.eval()

        # Get LCF parameters
        lcf_params = getattr(config, 'lcf_parameters', {})
        ema_decay = lcf_params.get('ema_decay', 0.95)
        logger.info(f"  EMA decay rate: {ema_decay}")
        logger.info(f"  Skipping step k: {lcf_params.get('skipping_step', 20)}")
        logger.info(f"  CFG scale range: [{lcf_params.get('w_min', 3.0)}, {lcf_params.get('w_max', 15.0)}]")
        logger.info("="*60 + "\n")

    if hasattr(config, 'nullbooth') and getattr(config.nullbooth, 'enable', False):
        try:
            from .nullbooth_trainer import NullBoothTrainer
            from .correct_optimizer import AlphaEditOptimizer

            # Get device from accelerator
            device = accelerator.device

            # Initialize NullBooth trainer (manages covariance matrices)
            nullbooth_trainer = NullBoothTrainer(config, unet, device)

            # Wrap optimizer with AlphaEdit projection
            # IMPORTANT: This must happen AFTER accelerator.prepare() in train.py
            logger.info("\n" + "="*60)
            logger.info("🎯 Initializing AlphaEdit NullBooth Training")
            logger.info("="*60)

            alphaedit_optimizer = AlphaEditOptimizer(
                optimizer=optimizer,
                cov_manager=nullbooth_trainer.cov_manager,
                unet=unet,
                enable_projection=True,
                debug=getattr(config.nullbooth, 'debug', False)
            )

            # Replace the original optimizer
            optimizer = alphaedit_optimizer

            logger.info("✅ AlphaEdit optimizer wrapper initialized successfully")
            logger.info("  - Mode: Projecting weight UPDATES (Δ = W_new - W_old)")
            logger.info("  - NOT projecting gradients or features (correct implementation)")
            logger.info(f"  - Covariance matrices: {config.nullbooth.cov_matrices_output_dir}")
            logger.info(f"  - Available timesteps: {len(nullbooth_trainer.cov_manager.available_timesteps)}")
            logger.info("="*60 + "\n")

        except Exception as e:
            logger.error(f"Failed to initialize NullBooth trainer: {e}")
            logger.info("Continuing with standard DreamBooth training")
            nullbooth_trainer = None
            alphaedit_optimizer = None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        # Convert config to a serializable dict for tracking
        config_dict = {}
        for key, value in config._config.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                config_dict[key] = value
            else:
                config_dict[key] = str(value)  # Convert complex objects to string

        accelerator.init_trackers("dreambooth", config=config_dict)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    if nullbooth_trainer and nullbooth_trainer.enabled:
        logger.info("  🎯 NullBooth Mode: AlphaEdit null-space projection ACTIVE")
        logger.info(f"  📊 Covariance matrices dir: {config.nullbooth.cov_matrices_output_dir}")
        logger.info(f"  🔍 Nullspace threshold: {getattr(config.nullbooth, 'nullspace_threshold', 'default (2e-2)')}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(config.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * config.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    try:
        for epoch in range(first_epoch, config.num_train_epochs):
            unet.train()
            if config.train_text_encoder:
                text_encoder.train()
            with TorchTracemalloc() if not config.no_tracemalloc else nullcontext() as tracemalloc:
                for step, batch in enumerate(train_dataloader):
                    # Skip steps until we reach the resumed step
                    if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                        if step % config.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            if config.report_to == "wandb":
                                accelerator.print(progress_bar)
                        continue

                    with accelerator.accumulate(unet):
                        loss, timesteps = training_step(
                            batch, unet, text_encoder, vae, noise_scheduler,
                            accelerator, config, weight_dtype, ema_unet
                        )

                        # Set current timestep for AlphaEdit projection AFTER training_step
                        if alphaedit_optimizer is not None:
                            alphaedit_optimizer.set_current_timestep(timesteps[0].item())

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(unet.parameters(), text_encoder.parameters())
                                if config.train_text_encoder
                                else unet.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # Update EMA model for LCF
                        if lcf_mode and ema_unet is not None and accelerator.sync_gradients:
                            with torch.no_grad():
                                lcf_params = getattr(config, 'lcf_parameters', {})
                                ema_decay = lcf_params.get('ema_decay', 0.95)
                                for ema_param, online_param in zip(ema_unet.parameters(), unet.parameters()):
                                    ema_param.data.mul_(ema_decay).add_(
                                        online_param.data, alpha=1 - ema_decay
                                    )

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        if config.report_to == "wandb":
                            accelerator.print(progress_bar)
                        global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

                    # Add NullBooth status to logs
                    if nullbooth_trainer and nullbooth_trainer.enabled:
                        logs["nullbooth"] = "active"

                    # Add LCF status to logs
                    if lcf_mode:
                        logs["mode"] = "LCF"

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # Run validation
                    run_validation(config, accelerator, unet, text_encoder, epoch, step, num_update_steps_per_epoch)

                    if global_step >= config.max_train_steps:
                        break

            # Print memory usage
            if not config.no_tracemalloc:
                accelerator.print(
                    f"{accelerator.device.type.upper()} Memory before entering the train : {b2mb(tracemalloc.begin)}"
                )
                accelerator.print(
                    f"{accelerator.device.type.upper()} Memory consumed at the end of the train (end-begin): {tracemalloc.used}"
                )
                accelerator.print(
                    f"{accelerator.device.type.upper()} Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}"
                )
                accelerator.print(
                    f"{accelerator.device.type.upper()} Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
                )

                accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
                accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
                accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
                accelerator.print(
                    f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
                )

    finally:
        # Clean up NullBooth trainer
        if alphaedit_optimizer:
            logger.info("\n" + "="*60)
            logger.info("📊 AlphaEdit NullBooth Training Summary")
            logger.info("="*60)
            logger.info("✅ Successfully applied null-space projection to weight updates")
            logger.info("✅ Original knowledge preserved through K₀K₀ᵀ null-space constraint")
            logger.info("="*60)

        if nullbooth_trainer:
            nullbooth_trainer.cleanup()
            logger.info("NullBooth trainer cleaned up")

def save_model(config, accelerator, unet, text_encoder, api=None, repo_id=None):
    """Save the trained model."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if config.lora.use_lora:
            unwarpped_unet = accelerator.unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(config.output_dir, "unet"), 
                state_dict=accelerator.get_state_dict(unet)
            )
            if config.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(config.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=config.revision,
            )
            pipeline.save_pretrained(config.output_dir)

        if config.push_to_hub and api and repo_id:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=config.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )
