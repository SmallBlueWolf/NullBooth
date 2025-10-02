# Inference Feature Collection Modes

## Overview
The InferenceFeatureCollector supports two modes for collecting features that represent the model's original knowledge. Both modes perform proper step-by-step denoising without simplification, using the scheduler's actual denoising steps.

## Modes

### 1. `partial_denoise` Mode (Default)
**Process: Noise → Target Timestep**

- **Start**: Pure random noise (timestep 999)
- **End**: Target timestep
- **Steps**: Step-by-step denoising from max noise to target

**How it works:**
1. Start with pure random noise latents
2. Use scheduler to set all timesteps (typically 1000)
3. Perform step-by-step denoising from timestep 999 down to target timestep
4. Each step uses `scheduler.step()` for proper denoising
5. Collect features at the target timestep

**Example:**
- Target timestep 500: Denoise from 999→998→997...→501→500
- Target timestep 100: Denoise from 999→998→997...→101→100

**Use case**: Captures how the model progressively constructs concepts from noise

### 2. `full_denoise` Mode
**Process: Clean Image → Add Noise → Target Timestep**

- **Step 1**: Generate a complete clean image (full denoising)
- **Step 2**: Add noise to reach target timestep
- **Step 3**: Run UNet at target timestep to collect features

**How it works:**
1. Start with random noise
2. Perform complete denoising (50 steps) to generate clean image
3. Add calibrated noise to the clean latents to reach target timestep
4. Run UNet at target timestep to collect features

**Example:**
- Target timestep 500: Generate clean image, then add noise to timestep 500
- Target timestep 100: Generate clean image, then add noise to timestep 100

**Use case**: Analyzes how clean, well-formed concepts appear at different noise levels

## Key Differences

| Aspect | partial_denoise | full_denoise |
|--------|----------------|--------------|
| Starting point | Pure noise | Clean generated image |
| Process | Progressive denoising | Generate then corrupt |
| Computational cost | Variable (depends on target) | High (always full generation) |
| Feature interpretation | Construction process | Noise robustness |
| Best for | Understanding denoising dynamics | Understanding concept encoding |

## Configuration

In `configs/nullbooth.yaml`:
```yaml
inference_mode: "partial_denoise"  # or "full_denoise"
```

## Technical Details

### Scheduler Integration
Both modes use the diffusion pipeline's scheduler properly:
- `scheduler.set_timesteps()`: Initialize timestep schedule
- `scheduler.scale_model_input()`: Scale latents appropriately
- `scheduler.step()`: Perform denoising step (partial_denoise)
- `scheduler.add_noise()`: Add calibrated noise (full_denoise)

### Alpha Value Handling
For non-integer timesteps (e.g., from Karras samplers):
- Float values are converted to integers: `int(alpha)`
- The same denoising logic applies
- Compatible with all sampling strategies

### No Simplification
Unlike simplified approaches, both modes:
- Use the actual scheduler for each denoising step
- Maintain proper noise schedules
- Preserve the diffusion model's dynamics
- No arbitrary step sizes or shortcuts

## Performance Considerations

- **partial_denoise**: Faster for high-noise targets (e.g., 900+), slower for low-noise targets
- **full_denoise**: Consistent time (always generates full image), but higher overall cost
- **Memory**: Both modes clear cache periodically to manage GPU memory
- **Quality**: full_denoise may produce cleaner features but at higher computational cost