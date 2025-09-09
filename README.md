# DreamBooth LoRA Training - Modular Implementation

A modular implementation of DreamBooth training with LoRA (Low-Rank Adaptation) support, refactored from the original monolithic script into a clean, maintainable codebase.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for config, dataset, models, training utilities, and training logic
- **Configuration-driven**: YAML-based configuration instead of command-line arguments
- **LoRA Support**: Efficient fine-tuning using Parameter Efficient Fine-Tuning (PEFT) with LoRA
- **Memory Efficient**: Built-in memory tracking and optimization features
- **Easy Inference**: Simple inference script with interactive mode
- **Hub Integration**: Support for Hugging Face Hub model sharing

## Project Structure

```
NullBooth/
├── configs/
│   └── config.yaml          # Training and inference configuration
├── src/
│   ├── __init__.py         # Module exports
│   ├── config.py           # Configuration loading and validation
│   ├── dataset.py          # Dataset and data loading utilities
│   ├── models.py           # Model loading and LoRA setup
│   ├── training_utils.py   # Training utilities and setup functions
│   └── trainer.py          # Core training logic
├── train.py                # Main training script
├── inference.py            # Inference script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install additional dependencies if needed:
```bash
# For 8-bit training
pip install bitsandbytes

# For memory efficient attention
pip install xformers
```

## Configuration

Edit `configs/config.yaml` to configure your training. Key parameters to set:

```yaml
# Required parameters
pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"
instance_data_dir: "/path/to/your/instance/images"
instance_prompt: "a photo of sks person"

# LoRA settings
use_lora: true
lora_r: 8
lora_alpha: 32

# Training settings
train_batch_size: 4
num_train_epochs: 1
learning_rate: 5e-6
```

## Training

Run training with the default configuration:
```bash
python train.py
```

Or specify a custom config file:
```bash
python train.py configs/my_config.yaml
```

For distributed training with Accelerate:
```bash
accelerate launch train.py
```

## Inference

### Basic Inference
Run inference with settings from config:
```bash
python inference.py
```

### Interactive Mode
Run in interactive mode to generate images with custom prompts:
```bash
python inference.py --interactive
```

### Custom Config
Use a different config file:
```bash
python inference.py --config configs/my_config.yaml
```

## Configuration Options

### Model Configuration
- `pretrained_model_name_or_path`: Base model to fine-tune
- `revision`: Model revision to use
- `tokenizer_name`: Custom tokenizer (optional)

### Data Configuration
- `instance_data_dir`: Directory containing your training images
- `instance_prompt`: Prompt describing your concept (e.g., "a photo of sks person")
- `class_data_dir`: Directory for class/regularization images (optional)
- `class_prompt`: Prompt for class images (optional)

### LoRA Configuration
- `use_lora`: Enable LoRA training (recommended: true)
- `lora_r`: LoRA rank (higher = more parameters)
- `lora_alpha`: LoRA scaling parameter
- `lora_dropout`: LoRA dropout rate

### Training Configuration
- `train_batch_size`: Batch size per device
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `resolution`: Image resolution for training
- `gradient_checkpointing`: Enable to save memory

### Inference Configuration
- `prompt`: Default prompt for inference
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: CFG scale
- `height`/`width`: Output image dimensions

## Memory Optimization

For training on limited GPU memory:

1. Enable gradient checkpointing:
```yaml
gradient_checkpointing: true
```

2. Use mixed precision:
```yaml
mixed_precision: "fp16"  # or "bf16" for newer GPUs
```

3. Use 8-bit Adam optimizer:
```yaml
use_8bit_adam: true
```

4. Enable xformers memory efficient attention:
```yaml
enable_xformers_memory_efficient_attention: true
```

## Prior Preservation

To maintain the model's ability to generate the original class:

```yaml
with_prior_preservation: true
class_data_dir: "/path/to/class/images"
class_prompt: "a photo of person"
num_class_images: 100
prior_loss_weight: 1.0
```

## Validation

Enable validation during training:

```yaml
validation_prompt: "a photo of sks person in a garden"
num_validation_images: 4
validation_steps: 100
```

## Hub Integration

To push your trained model to Hugging Face Hub:

```yaml
push_to_hub: true
hub_model_id: "your-username/your-model-name"
hub_token: "your-hf-token"
```

## Example Training Workflow

1. Prepare your training images (3-10 high-quality images work well)
2. Update the config with your paths and prompts
3. Start training:
   ```bash
   accelerate launch train.py
   ```
4. Monitor training with TensorBoard:
   ```bash
   tensorboard --logdir dreambooth-lora-model/logs
   ```
5. Test your model:
   ```bash
   python inference.py --interactive
   ```

## Tips

- Use unique identifier tokens (like "sks", "xyz123") in your instance prompt
- Keep learning rate relatively low (1e-6 to 5e-6) for stable training
- For faces, 3-10 high-quality images usually work well
- Monitor the loss curves and validation images during training
- LoRA models are much smaller and faster to train than full fine-tuning

## Troubleshooting

### CUDA Out of Memory
- Reduce `train_batch_size`
- Enable `gradient_checkpointing`
- Use mixed precision training
- Enable xformers if available

### Poor Quality Results
- Increase `num_train_epochs` or `max_train_steps`
- Try different `learning_rate` values
- Add more diverse training images
- Use prior preservation with class images

### Model Not Learning the Concept
- Make sure instance prompt uses a unique identifier
- Check that training images are high quality and consistent
- Increase LoRA rank (`lora_r`) for more capacity
- Increase learning rate slightly

## License

This implementation is based on the original DreamBooth training script from the Diffusers library and follows the same licensing terms.