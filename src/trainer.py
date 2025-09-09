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
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        if config.seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        else:
            generator = None
        images = []
        for _ in range(config.num_validation_images):
            image = pipeline(config.validation_prompt, num_inference_steps=25, generator=generator).images[0]
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


def training_step(batch, unet, text_encoder, vae, noise_scheduler, accelerator, config, weight_dtype):
    """Perform a single training step."""
    # Convert images to latent space
    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * 0.18215

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

    # Predict the noise residual
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

    return loss


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
    """Main training loop."""
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
                    loss = training_step(
                        batch, unet, text_encoder, vae, noise_scheduler, accelerator, config, weight_dtype
                    )

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

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if config.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
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