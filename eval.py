"""
Evaluation script for assessing continual learning in diffusion models.
Compares original and finetuned models on specified prompts using multiple metrics.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from src.feature_extractors import get_feature_extractor
from src.metrics import compute_all_metrics, interpret_metrics
from src.report_generator import create_markdown_report, save_json_report, create_visualization


class DiffusionModelEvaluator:
    """Evaluator for continual learning in diffusion models."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.output_dir = Path(config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "generated_images"
        self.features_dir = self.output_dir / "features"
        if config['output'].get('save_images', True):
            self.images_dir.mkdir(exist_ok=True)
        if config['output'].get('save_features', True):
            self.features_dir.mkdir(exist_ok=True)

        # Load models
        print("=" * 80)
        print("Loading models...")
        print("=" * 80)
        self.original_pipeline = self._load_pipeline(
            config['original_model_path'],
            "Original Model"
        )
        self.finetuned_pipeline = self._load_pipeline(
            config['finetuned_model_path'],
            "Finetuned Model"
        )

        # Initialize feature extractors
        print("\n" + "=" * 80)
        print("Loading feature extractors...")
        print("=" * 80)
        self.extractors = {}
        for extractor_name in config['feature_extractors']:
            try:
                self.extractors[extractor_name] = get_feature_extractor(
                    extractor_name,
                    device=self.device
                )
                print(f"✓ Loaded {extractor_name.upper()} feature extractor")
            except Exception as e:
                print(f"✗ Failed to load {extractor_name}: {e}")

        if not self.extractors:
            raise RuntimeError("No feature extractors could be loaded!")

        print("=" * 80)

    def _load_pipeline(self, model_path: str, model_name: str) -> StableDiffusionPipeline:
        """Load a Stable Diffusion pipeline."""
        print(f"\nLoading {model_name} from: {model_path}")

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.config.get('mixed_precision', True) else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
        except Exception as e:
            print(f"Failed to load from pretrained: {e}")
            print("Trying to load from single file...")
            pipeline = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if self.config.get('mixed_precision', True) else torch.float32,
                safety_checker=None,
            )

        # Use faster scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        # Enable optimizations
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print(f"✓ Enabled xformers for {model_name}")
            except:
                pass

        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()

        print(f"✓ {model_name} loaded successfully")
        return pipeline

    def generate_images(self,
                       pipeline: StableDiffusionPipeline,
                       prompt: str,
                       num_samples: int,
                       seed: int = None) -> List[Image.Image]:
        """Generate images from a pipeline."""
        images = []
        batch_size = self.config['generation']['batch_size']
        num_inference_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']

        # Set up generator
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating images"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            with torch.no_grad():
                output = pipeline(
                    prompt=[prompt] * current_batch_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                images.extend(output.images)

        return images[:num_samples]

    def save_images(self, images: List[Image.Image], prompt: str, model_type: str):
        """Save generated images to disk."""
        prompt_dir = self.images_dir / f"{model_type}" / prompt[:50].replace('/', '_')
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            img.save(prompt_dir / f"image_{idx:04d}.png")

    def extract_features(self,
                        images: List[Image.Image],
                        extractor_name: str) -> np.ndarray:
        """Extract features from images."""
        extractor = self.extractors[extractor_name]
        return extractor.extract(images)

    def evaluate_prompt(self, prompt: str) -> Dict:
        """Evaluate a single prompt."""
        print(f"\n{'=' * 80}")
        print(f"Evaluating prompt: \"{prompt}\"")
        print(f"{'=' * 80}")

        num_samples = self.config['generation']['num_samples_per_prompt']
        seed = self.config['generation'].get('seed')

        # Generate images from both models
        print("\n[1/4] Generating images from original model...")
        original_images = self.generate_images(
            self.original_pipeline,
            prompt,
            num_samples,
            seed
        )

        print("\n[2/4] Generating images from finetuned model...")
        finetuned_images = self.generate_images(
            self.finetuned_pipeline,
            prompt,
            num_samples,
            seed + 1000 if seed is not None else None  # Different seed
        )

        # Save images if requested
        if self.config['output'].get('save_images', True):
            print("\nSaving generated images...")
            self.save_images(original_images, prompt, "original")
            self.save_images(finetuned_images, prompt, "finetuned")

        # Extract features and compute metrics
        print("\n[3/4] Extracting features and computing metrics...")
        result = {
            'prompt': prompt,
            'num_samples': num_samples,
            'metrics': {},
            'interpretation': {}
        }

        for extractor_name in self.extractors.keys():
            print(f"\n  Using {extractor_name.upper()} extractor:")

            # Extract features
            features_orig = self.extract_features(original_images, extractor_name)
            features_fine = self.extract_features(finetuned_images, extractor_name)

            # Save features if requested
            if self.config['output'].get('save_features', True):
                feature_dir = self.features_dir / prompt[:50].replace('/', '_')
                feature_dir.mkdir(parents=True, exist_ok=True)
                np.save(feature_dir / f"{extractor_name}_original.npy", features_orig)
                np.save(feature_dir / f"{extractor_name}_finetuned.npy", features_fine)

            # Compute metrics
            metrics = compute_all_metrics(
                features_orig,
                features_fine,
                self.config['metrics'],
                self.config.get('metric_settings', {})
            )

            # Interpret metrics
            interpretation = interpret_metrics(metrics, extractor_name)

            result['metrics'][extractor_name] = metrics
            result['interpretation'][extractor_name] = interpretation

            # Print results
            print(f"    Metrics:")
            for metric_name, value in metrics.items():
                print(f"      {metric_name.upper()}: {value:.4f}")
            print(f"    Overall Quality: {interpretation.get('overall', 'N/A')}")

        return result

    def run_evaluation(self) -> Dict:
        """Run full evaluation on all prompts."""
        print("\n" + "=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)

        results = {
            'config': self.config,
            'per_prompt_results': []
        }

        # Evaluate each prompt
        for prompt in self.config['prompts']:
            try:
                result = self.evaluate_prompt(prompt)
                results['per_prompt_results'].append(result)
            except Exception as e:
                print(f"\n✗ Error evaluating prompt '{prompt}': {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)

        return results

    def generate_reports(self, results: Dict):
        """Generate evaluation reports."""
        print("\n" + "=" * 80)
        print("Generating reports...")
        print("=" * 80)

        report_format = self.config['output'].get('report_format', 'markdown')

        if report_format in ['markdown', 'both']:
            md_path = self.output_dir / "evaluation_report.md"
            create_markdown_report(results, self.config, str(md_path))

        if report_format in ['json', 'both']:
            json_path = self.output_dir / "evaluation_report.json"
            save_json_report(results, str(json_path))

        # Create visualizations
        if self.config['output'].get('create_visualizations', True):
            print("\nCreating visualizations...")
            try:
                create_visualization(results, str(self.output_dir), self.config)
            except Exception as e:
                print(f"Failed to create visualizations: {e}")

        print("\n" + "=" * 80)
        print(f"All reports saved to: {self.output_dir}")
        print("=" * 80)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continual learning in diffusion models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--original_model',
        type=str,
        default=None,
        help='Override original model path'
    )
    parser.add_argument(
        '--finetuned_model',
        type=str,
        default=None,
        help='Override finetuned model path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        default=None,
        help='Override test prompts'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override with command line arguments
    if args.original_model:
        config['original_model_path'] = args.original_model
    if args.finetuned_model:
        config['finetuned_model_path'] = args.finetuned_model
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.prompts:
        config['prompts'] = args.prompts

    # Validate required fields
    if not config.get('original_model_path'):
        raise ValueError("original_model_path must be specified in config or via --original_model")
    if not config.get('finetuned_model_path'):
        raise ValueError("finetuned_model_path must be specified in config or via --finetuned_model")

    # Run evaluation
    evaluator = DiffusionModelEvaluator(config)
    results = evaluator.run_evaluation()
    evaluator.generate_reports(results)

    print("\n✓ Evaluation complete!")
    print(f"  Results saved to: {evaluator.output_dir}")
    print(f"  Main report: {evaluator.output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
