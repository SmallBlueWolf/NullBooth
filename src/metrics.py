"""
Metric implementations for evaluating continual learning in diffusion models.
Adapted from MeshDiffusion metrics for image feature space.
"""

import numpy as np
import torch
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from typing import Optional, Tuple


def compute_mmd(features_a: np.ndarray,
                features_b: np.ndarray,
                kernel: str = "gaussian",
                sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two feature distributions.

    Args:
        features_a: Features from original model (N, D)
        features_b: Features from finetuned model (M, D)
        kernel: Kernel type ("gaussian" or "linear")
        sigma: Bandwidth for gaussian kernel

    Returns:
        MMD value (lower is better, means distributions are more similar)
    """
    features_a = torch.from_numpy(features_a).float()
    features_b = torch.from_numpy(features_b).float()

    if kernel == "gaussian":
        def gaussian_kernel(x, y, sigma):
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))

        K_aa = gaussian_kernel(features_a, features_a, sigma).mean()
        K_bb = gaussian_kernel(features_b, features_b, sigma).mean()
        K_ab = gaussian_kernel(features_a, features_b, sigma).mean()

    elif kernel == "linear":
        K_aa = (features_a @ features_a.T).mean()
        K_bb = (features_b @ features_b.T).mean()
        K_ab = (features_a @ features_b.T).mean()
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    mmd_squared = K_aa + K_bb - 2 * K_ab
    mmd = torch.sqrt(torch.clamp(mmd_squared, min=0.0))

    return float(mmd.item())


def compute_coverage(features_orig: np.ndarray,
                    features_fine: np.ndarray,
                    k: int = 5) -> float:
    """
    Compute Coverage metric.
    Measures what percentage of finetuned features are used to cover original features.

    Args:
        features_orig: Features from original model (N, D)
        features_fine: Features from finetuned model (M, D)
        k: Number of nearest neighbors

    Returns:
        Coverage percentage (higher is better, means good diversity preservation)
    """
    if len(features_fine) == 0:
        return 0.0

    # Find k nearest neighbors in finetuned features for each original feature
    nbrs = NearestNeighbors(n_neighbors=min(k, len(features_fine)),
                           metric='euclidean').fit(features_fine)
    _, indices = nbrs.kneighbors(features_orig)

    # Count unique finetuned features used
    unique_matches = len(set(indices.flatten()))
    coverage_pct = (unique_matches / len(features_fine)) * 100

    return coverage_pct


def compute_1nna(features_orig: np.ndarray,
                features_fine: np.ndarray) -> float:
    """
    Compute 1-Nearest Neighbor Accuracy (1-NNA).
    Leave-one-out accuracy of 1-NN classifier.

    Args:
        features_orig: Features from original model (N, D)
        features_fine: Features from finetuned model (M, D)

    Returns:
        Accuracy percentage (closer to 50% is better, means harder to distinguish)
    """
    # Combine features and create labels
    X = np.vstack([features_orig, features_fine])
    y = np.array([0] * len(features_orig) + [1] * len(features_fine))

    # Leave-one-out evaluation
    correct = 0
    for i in range(len(X)):
        # Remove current sample
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        # Train 1-NN classifier
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(X_train, y_train)

        # Predict
        pred = knn.predict(X[i:i+1])
        if pred[0] == y[i]:
            correct += 1

    accuracy = (correct / len(X)) * 100
    return accuracy


def compute_fid(features_orig: np.ndarray,
               features_fine: np.ndarray,
               eps: float = 1e-6) -> float:
    """
    Compute Frechet Inception Distance (FID).

    Args:
        features_orig: Features from original model (N, D)
        features_fine: Features from finetuned model (M, D)
        eps: Small value for numerical stability

    Returns:
        FID value (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = features_orig.mean(axis=0), np.cov(features_orig, rowvar=False)
    mu2, sigma2 = features_fine.mean(axis=0), np.cov(features_fine, rowvar=False)

    # Calculate Frechet distance
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


def compute_jsd(features_orig: np.ndarray,
               features_fine: np.ndarray,
               n_bins: int = 50) -> float:
    """
    Compute Jensen-Shannon Divergence (JSD).

    Args:
        features_orig: Features from original model (N, D)
        features_fine: Features from finetuned model (M, D)
        n_bins: Number of bins for histogram estimation

    Returns:
        JSD value (lower is better, range [0, 1])
    """
    from scipy.spatial.distance import jensenshannon

    # Flatten features to 1D for histogram
    # Alternative: compute JSD for each dimension and average
    jsd_per_dim = []

    for dim in range(features_orig.shape[1]):
        # Get values for this dimension
        vals_orig = features_orig[:, dim]
        vals_fine = features_fine[:, dim]

        # Create common bins
        all_vals = np.concatenate([vals_orig, vals_fine])
        bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

        # Compute histograms
        hist_orig, _ = np.histogram(vals_orig, bins=bins, density=True)
        hist_fine, _ = np.histogram(vals_fine, bins=bins, density=True)

        # Normalize to probability distributions
        hist_orig = hist_orig / (hist_orig.sum() + 1e-10)
        hist_fine = hist_fine / (hist_fine.sum() + 1e-10)

        # Compute JSD for this dimension
        jsd = jensenshannon(hist_orig, hist_fine, base=2)
        jsd_per_dim.append(jsd)

    # Average across dimensions
    return float(np.mean(jsd_per_dim))


def compute_all_metrics(features_orig: np.ndarray,
                       features_fine: np.ndarray,
                       metric_names: list,
                       metric_settings: dict) -> dict:
    """
    Compute all requested metrics.

    Args:
        features_orig: Features from original model
        features_fine: Features from finetuned model
        metric_names: List of metric names to compute
        metric_settings: Dictionary of metric-specific settings

    Returns:
        Dictionary of metric values
    """
    results = {}

    if "mmd" in metric_names:
        mmd_settings = metric_settings.get("mmd", {})
        results["mmd"] = compute_mmd(
            features_orig, features_fine,
            kernel=mmd_settings.get("kernel", "gaussian"),
            sigma=mmd_settings.get("sigma", 1.0)
        )

    if "coverage" in metric_names:
        cov_settings = metric_settings.get("coverage", {})
        results["coverage"] = compute_coverage(
            features_orig, features_fine,
            k=cov_settings.get("k", 5)
        )

    if "1-nna" in metric_names:
        results["1-nna"] = compute_1nna(features_orig, features_fine)

    if "fid" in metric_names:
        results["fid"] = compute_fid(features_orig, features_fine)

    if "jsd" in metric_names:
        jsd_settings = metric_settings.get("jsd", {})
        results["jsd"] = compute_jsd(
            features_orig, features_fine,
            n_bins=jsd_settings.get("n_bins", 50)
        )

    return results


def interpret_metrics(metrics: dict, feature_extractor: str) -> dict:
    """
    Interpret metric values and provide quality assessment.

    Args:
        metrics: Dictionary of computed metrics
        feature_extractor: Name of feature extractor used

    Returns:
        Dictionary with interpretation
    """
    interpretation = {}

    # MMD thresholds
    if "mmd" in metrics:
        mmd = metrics["mmd"]
        if mmd < 0.01:
            interpretation["mmd"] = "excellent"
        elif mmd < 0.05:
            interpretation["mmd"] = "good"
        elif mmd < 0.1:
            interpretation["mmd"] = "fair"
        else:
            interpretation["mmd"] = "poor"

    # Coverage thresholds
    if "coverage" in metrics:
        cov = metrics["coverage"]
        if cov > 85:
            interpretation["coverage"] = "excellent"
        elif cov > 70:
            interpretation["coverage"] = "good"
        elif cov > 50:
            interpretation["coverage"] = "fair"
        else:
            interpretation["coverage"] = "poor"

    # 1-NNA thresholds (50% is ideal - completely indistinguishable)
    if "1-nna" in metrics:
        nna = metrics["1-nna"]
        if 50 <= nna <= 55:
            interpretation["1-nna"] = "excellent"
        elif 45 <= nna <= 65:
            interpretation["1-nna"] = "good"
        elif 40 <= nna <= 75:
            interpretation["1-nna"] = "fair"
        else:
            interpretation["1-nna"] = "poor"

    # FID thresholds
    if "fid" in metrics:
        fid = metrics["fid"]
        if fid < 10:
            interpretation["fid"] = "excellent"
        elif fid < 30:
            interpretation["fid"] = "good"
        elif fid < 50:
            interpretation["fid"] = "fair"
        else:
            interpretation["fid"] = "poor"

    # JSD thresholds (0 is perfect, 1 is completely different)
    if "jsd" in metrics:
        jsd = metrics["jsd"]
        if jsd < 0.05:
            interpretation["jsd"] = "excellent"
        elif jsd < 0.15:
            interpretation["jsd"] = "good"
        elif jsd < 0.3:
            interpretation["jsd"] = "fair"
        else:
            interpretation["jsd"] = "poor"

    # Overall assessment
    quality_scores = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
    if interpretation:
        avg_score = np.mean([quality_scores[v] for v in interpretation.values()])
        if avg_score >= 3.5:
            interpretation["overall"] = "excellent"
        elif avg_score >= 2.5:
            interpretation["overall"] = "good"
        elif avg_score >= 1.5:
            interpretation["overall"] = "fair"
        else:
            interpretation["overall"] = "poor"

    return interpretation
