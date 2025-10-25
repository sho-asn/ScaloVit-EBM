"""Model definitions for ScaloViT EBM."""

from .ablation import ConvHead_EBMWrapper, ImageBased_EBViTModelWrapper
from .ebm_vit import EBViTModelWrapper

__all__ = ["EBViTModelWrapper", "ImageBased_EBViTModelWrapper", "ConvHead_EBMWrapper"]
