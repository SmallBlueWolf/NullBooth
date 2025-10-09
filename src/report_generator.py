"""
Report generation utilities for evaluation results.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def create_markdown_report(results: Dict,
                          config: Dict,
                          output_path: str) -> None:
    """
    Create a comprehensive markdown report.

    Args:
        results: Evaluation results dictionary
        config: Configuration dictionary
        output_path: Path to save the markdown file
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("# Continual Learning Evaluation Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Model Information
        f.write("## Model Information\n\n")
        f.write(f"- **Original Model:** `{config['original_model_path']}`\n")
        f.write(f"- **Finetuned Model:** `{config['finetuned_model_path']}`\n")
        f.write(f"- **Samples per Prompt:** {config['generation']['num_samples_per_prompt']}\n")
        f.write(f"- **Inference Steps:** {config['generation']['num_inference_steps']}\n")
        f.write(f"- **Guidance Scale:** {config['generation']['guidance_scale']}\n\n")

        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write(_create_summary_table(results))
        f.write("\n\n")

        # Per-Prompt Results
        f.write("## Detailed Results by Prompt\n\n")

        for prompt_result in results['per_prompt_results']:
            prompt = prompt_result['prompt']
            f.write(f"### Prompt: \"{prompt}\"\n\n")

            # Create table for each feature extractor
            for extractor in config['feature_extractors']:
                if extractor not in prompt_result['metrics']:
                    continue

                f.write(f"#### Feature Extractor: {extractor.upper()}\n\n")
                f.write(_create_metrics_table(prompt_result['metrics'][extractor]))
                f.write("\n\n")

                # Interpretation
                if 'interpretation' in prompt_result:
                    interp = prompt_result['interpretation'].get(extractor, {})
                    if interp:
                        f.write("**Quality Assessment:**\n\n")
                        for metric, quality in interp.items():
                            if metric != 'overall':
                                emoji = _get_quality_emoji(quality)
                                f.write(f"- {metric.upper()}: {quality} {emoji}\n")
                        f.write(f"\n**Overall Quality:** {interp.get('overall', 'N/A')} "
                               f"{_get_quality_emoji(interp.get('overall', 'N/A'))}\n\n")

        # Aggregated Results Across All Prompts
        f.write("## Aggregated Results (Average Across All Prompts)\n\n")
        f.write(_create_aggregated_table(results))
        f.write("\n\n")

        # Metric Explanations
        f.write("## Metric Explanations\n\n")
        f.write(_create_metric_explanations())
        f.write("\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(_create_recommendations(results))
        f.write("\n")

    print(f"Markdown report saved to: {output_path}")


def _create_summary_table(results: Dict) -> str:
    """Create summary statistics table."""
    table = "| Metric | Description |\n"
    table += "|--------|-------------|\n"
    table += f"| Total Prompts Tested | {len(results['per_prompt_results'])} |\n"

    # Count by quality
    if results['per_prompt_results']:
        first_result = results['per_prompt_results'][0]
        extractors = list(first_result['metrics'].keys())
        table += f"| Feature Extractors | {', '.join([e.upper() for e in extractors])} |\n"

        # Calculate average overall quality
        quality_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        total_assessments = 0

        for prompt_result in results['per_prompt_results']:
            if 'interpretation' in prompt_result:
                for extractor, interp in prompt_result['interpretation'].items():
                    overall = interp.get('overall', 'N/A')
                    if overall in quality_counts:
                        quality_counts[overall] += 1
                        total_assessments += 1

        if total_assessments > 0:
            table += f"| Excellent Results | {quality_counts['excellent']} / {total_assessments} |\n"
            table += f"| Good Results | {quality_counts['good']} / {total_assessments} |\n"
            table += f"| Fair Results | {quality_counts['fair']} / {total_assessments} |\n"
            table += f"| Poor Results | {quality_counts['poor']} / {total_assessments} |\n"

    return table


def _create_metrics_table(metrics: Dict) -> str:
    """Create metrics table with interpretation."""
    table = "| Metric | Value | Interpretation |\n"
    table += "|--------|-------|----------------|\n"

    metric_info = {
        "mmd": ("MMD", "↓", lambda x: f"{x:.4f}"),
        "coverage": ("Coverage", "↑", lambda x: f"{x:.2f}%"),
        "1-nna": ("1-NNA", "~50%", lambda x: f"{x:.2f}%"),
        "fid": ("FID", "↓", lambda x: f"{x:.2f}"),
        "jsd": ("JSD", "↓", lambda x: f"{x:.4f}"),
    }

    for metric_key, value in metrics.items():
        if metric_key in metric_info:
            name, direction, formatter = metric_info[metric_key]
            formatted_value = formatter(value)
            table += f"| {name} | {formatted_value} | {direction} |\n"

    return table


def _create_aggregated_table(results: Dict) -> str:
    """Create aggregated results table."""
    # Aggregate metrics across all prompts
    aggregated = {}

    for prompt_result in results['per_prompt_results']:
        for extractor, metrics in prompt_result['metrics'].items():
            if extractor not in aggregated:
                aggregated[extractor] = {}

            for metric_name, value in metrics.items():
                if metric_name not in aggregated[extractor]:
                    aggregated[extractor][metric_name] = []
                aggregated[extractor][metric_name].append(value)

    # Create table
    table = ""
    for extractor, metrics_dict in aggregated.items():
        table += f"### {extractor.upper()}\n\n"
        table += "| Metric | Mean | Std | Min | Max |\n"
        table += "|--------|------|-----|-----|-----|\n"

        for metric_name, values in metrics_dict.items():
            values = np.array(values)
            table += f"| {metric_name.upper()} | "
            table += f"{values.mean():.4f} | "
            table += f"{values.std():.4f} | "
            table += f"{values.min():.4f} | "
            table += f"{values.max():.4f} |\n"

        table += "\n"

    return table


def _create_metric_explanations() -> str:
    """Create metric explanation section."""
    explanations = """
### Maximum Mean Discrepancy (MMD)
- **Range:** [0, ∞), lower is better
- **Interpretation:** Measures the distance between two feature distributions
- **Thresholds:**
  - < 0.01: Excellent (distributions very similar)
  - 0.01-0.05: Good
  - 0.05-0.1: Fair
  - \\> 0.1: Poor (distributions quite different)

### Coverage (COV)
- **Range:** [0, 100%], higher is better
- **Interpretation:** Percentage of finetuned features used to cover original features
- **Thresholds:**
  - \\> 85%: Excellent (good diversity preservation)
  - 70-85%: Good
  - 50-70%: Fair
  - < 50%: Poor (loss of diversity)

### 1-Nearest Neighbor Accuracy (1-NNA)
- **Range:** [0, 100%], ~50% is ideal
- **Interpretation:** How well a classifier can distinguish between original and finetuned samples
- **Thresholds:**
  - 50-55%: Excellent (indistinguishable)
  - 45-65%: Good
  - 40-75%: Fair
  - Otherwise: Poor (easily distinguishable)

### Fréchet Inception Distance (FID)
- **Range:** [0, ∞), lower is better
- **Interpretation:** Measures distance between feature distributions (mean + covariance)
- **Thresholds:**
  - < 10: Excellent
  - 10-30: Good
  - 30-50: Fair
  - \\> 50: Poor

### Jensen-Shannon Divergence (JSD)
- **Range:** [0, 1], lower is better
- **Interpretation:** Symmetric measure of similarity between two probability distributions
- **Thresholds:**
  - < 0.05: Excellent (very similar)
  - 0.05-0.15: Good
  - 0.15-0.3: Fair
  - \\> 0.3: Poor (very different)
"""
    return explanations


def _create_recommendations(results: Dict) -> str:
    """Create recommendations based on results."""
    recs = []

    # Analyze results
    poor_prompts = []
    good_prompts = []

    for prompt_result in results['per_prompt_results']:
        if 'interpretation' in prompt_result:
            for extractor, interp in prompt_result['interpretation'].items():
                overall = interp.get('overall', 'N/A')
                if overall == 'poor':
                    poor_prompts.append((prompt_result['prompt'], extractor))
                elif overall in ['excellent', 'good']:
                    good_prompts.append((prompt_result['prompt'], extractor))

    if poor_prompts:
        recs.append("### ⚠️ Areas of Concern\n")
        recs.append("The following prompts show significant semantic drift:\n")
        for prompt, extractor in poor_prompts[:5]:  # Show top 5
            recs.append(f"- \"{prompt}\" ({extractor})\n")
        recs.append("\n**Suggestion:** Consider increasing regularization or adjusting learning rate for these concepts.\n\n")

    if good_prompts:
        recs.append("### ✅ Well-Preserved Concepts\n")
        recs.append("The following prompts show good semantic preservation:\n")
        for prompt, extractor in good_prompts[:5]:  # Show top 5
            recs.append(f"- \"{prompt}\" ({extractor})\n")
        recs.append("\n")

    if not recs:
        recs.append("No specific recommendations at this time.\n")

    return "".join(recs)


def _get_quality_emoji(quality: str) -> str:
    """Get emoji for quality level."""
    emoji_map = {
        "excellent": "🟢",
        "good": "🟡",
        "fair": "🟠",
        "poor": "🔴",
        "N/A": "⚪"
    }
    return emoji_map.get(quality, "⚪")


def save_json_report(results: Dict, output_path: str) -> None:
    """Save results as JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved to: {output_path}")


def create_visualization(results: Dict,
                        output_dir: str,
                        config: Dict) -> None:
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping visualizations")
        return

    # Aggregate metrics
    aggregated = {}
    for prompt_result in results['per_prompt_results']:
        for extractor, metrics in prompt_result['metrics'].items():
            if extractor not in aggregated:
                aggregated[extractor] = {m: [] for m in metrics.keys()}
            for metric_name, value in metrics.items():
                aggregated[extractor][metric_name].append(value)

    # Create plots for each extractor
    for extractor, metrics_dict in aggregated.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Metrics Distribution - {extractor.upper()}', fontsize=16)

        metric_list = list(metrics_dict.keys())
        for idx, metric_name in enumerate(metric_list[:6]):  # Max 6 plots
            ax = axes[idx // 3, idx % 3]
            values = metrics_dict[metric_name]

            # Box plot
            ax.boxplot([values], labels=[metric_name.upper()])
            ax.set_ylabel('Value')
            ax.set_title(f'{metric_name.upper()}')
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(metric_list), 6):
            fig.delaxes(axes[idx // 3, idx % 3])

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'metrics_distribution_{extractor}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {plot_path}")

    # Create comparison plot across extractors
    if len(aggregated) > 1:
        _create_comparison_plot(aggregated, output_dir)


def _create_comparison_plot(aggregated: Dict, output_dir: str) -> None:
    """Create comparison plot across feature extractors."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get common metrics
    all_metrics = set()
    for metrics_dict in aggregated.values():
        all_metrics.update(metrics_dict.keys())

    common_metrics = list(all_metrics)[:5]  # Top 5 metrics

    fig, axes = plt.subplots(1, len(common_metrics), figsize=(5*len(common_metrics), 5))
    if len(common_metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(common_metrics):
        ax = axes[idx]
        data = []
        labels = []

        for extractor, metrics_dict in aggregated.items():
            if metric in metrics_dict:
                data.append(metrics_dict[metric])
                labels.append(extractor.upper())

        ax.boxplot(data, labels=labels)
        ax.set_title(metric.upper())
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'extractor_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_path}")
