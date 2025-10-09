"""
Feature extraction utilities for evaluation.
Supports CLIP, DINOv2, and InceptionV3.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Union
from torchvision import transforms
from tqdm import tqdm


class FeatureExtractor:
    """Base class for feature extractors."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None

    def extract(self, images: Union[List[Image.Image], torch.Tensor]) -> np.ndarray:
        """Extract features from images."""
        raise NotImplementedError

    def _preprocess_images(self, images: Union[List[Image.Image], torch.Tensor],
                          batch_size: int = 32) -> torch.Tensor:
        """Preprocess images for the model."""
        if isinstance(images, torch.Tensor):
            return images.to(self.device)

        # Convert PIL images to tensors
        processed = []
        for img in images:
            if self.preprocess is not None:
                processed.append(self.preprocess(img))
            else:
                processed.append(transforms.ToTensor()(img))

        return torch.stack(processed).to(self.device)


class CLIPFeatureExtractor(FeatureExtractor):
    """CLIP ViT-L/14 feature extractor."""

    def __init__(self, device: str = "cuda", model_name: str = "ViT-L/14"):
        super().__init__(device)
        try:
            import clip
        except ImportError:
            raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.feature_dim = 768 if "ViT-L" in model_name else 512

    @torch.no_grad()
    def extract(self, images: Union[List[Image.Image], torch.Tensor],
               batch_size: int = 32) -> np.ndarray:
        """Extract CLIP image features."""
        if isinstance(images, list):
            # Process in batches
            all_features = []
            for i in tqdm(range(0, len(images), batch_size),
                         desc="Extracting CLIP features"):
                batch = images[i:i + batch_size]
                batch_tensors = torch.stack([self.preprocess(img) for img in batch])
                batch_tensors = batch_tensors.to(self.device)

                features = self.model.encode_image(batch_tensors)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                all_features.append(features.cpu().numpy())

            return np.vstack(all_features)
        else:
            # Single batch tensor
            images = images.to(self.device)
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()


class DINOv2FeatureExtractor(FeatureExtractor):
    """DINOv2 ViT-L/14 feature extractor."""

    def __init__(self, device: str = "cuda", model_name: str = "dinov2_vitl14"):
        super().__init__(device)

        print(f"Loading DINOv2 model: {model_name}")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception as e:
            print(f"Failed to load from torch.hub: {e}")
            print("Trying alternative loading method...")
            # Fallback to local loading if available
            raise ImportError("Please ensure DINOv2 is accessible via torch.hub")

        self.model = self.model.to(device)
        self.model.eval()

        # DINOv2 preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        self.feature_dim = 1024  # for vitl14

    @torch.no_grad()
    def extract(self, images: Union[List[Image.Image], torch.Tensor],
               batch_size: int = 32) -> np.ndarray:
        """Extract DINOv2 features."""
        if isinstance(images, list):
            all_features = []
            for i in tqdm(range(0, len(images), batch_size),
                         desc="Extracting DINOv2 features"):
                batch = images[i:i + batch_size]
                batch_tensors = torch.stack([self.preprocess(img) for img in batch])
                batch_tensors = batch_tensors.to(self.device)

                features = self.model(batch_tensors)
                all_features.append(features.cpu().numpy())

            return np.vstack(all_features)
        else:
            images = images.to(self.device)
            features = self.model(images)
            return features.cpu().numpy()


class InceptionV3FeatureExtractor(FeatureExtractor):
    """InceptionV3 feature extractor (for FID)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)

        print("Loading InceptionV3 model")
        from torchvision.models import inception_v3, Inception_V3_Weights

        # Load pretrained InceptionV3
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()  # Remove final classification layer
        self.model = inception.to(device)
        self.model.eval()

        # InceptionV3 preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        self.feature_dim = 2048

    @torch.no_grad()
    def extract(self, images: Union[List[Image.Image], torch.Tensor],
               batch_size: int = 32) -> np.ndarray:
        """Extract InceptionV3 features."""
        if isinstance(images, list):
            all_features = []
            for i in tqdm(range(0, len(images), batch_size),
                         desc="Extracting InceptionV3 features"):
                batch = images[i:i + batch_size]
                batch_tensors = torch.stack([self.preprocess(img) for img in batch])
                batch_tensors = batch_tensors.to(self.device)

                features = self.model(batch_tensors)
                if isinstance(features, tuple):  # Handle aux outputs
                    features = features[0]
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
                all_features.append(features.cpu().numpy())

            return np.vstack(all_features)
        else:
            images = images.to(self.device)
            features = self.model(images)
            if isinstance(features, tuple):
                features = features[0]
            features = features.squeeze(-1).squeeze(-1)
            return features.cpu().numpy()


def get_feature_extractor(name: str, device: str = "cuda") -> FeatureExtractor:
    """Factory function to get feature extractor by name."""
    name = name.lower()

    if name == "clip":
        return CLIPFeatureExtractor(device=device)
    elif name == "dinov2":
        return DINOv2FeatureExtractor(device=device)
    elif name == "inception":
        return InceptionV3FeatureExtractor(device=device)
    else:
        raise ValueError(f"Unknown feature extractor: {name}. "
                        f"Available: clip, dinov2, inception")
