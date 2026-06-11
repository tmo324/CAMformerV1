"""
CAMformer Pipeline Stages

Three-stage attention pipeline:
1. Association: Q*K^T similarity via BA-CAM binary search
2. Selection: Top-K + SoftMax for sparse attention weights
3. Contextualization: Attention × V via sparse BF16 MAC
"""

from .association import AssociationStage, AssociationConfig
from .selection import SelectionStage, SelectionConfig
from .contextualization import ContextualizationStage, ContextualizationConfig
from .camformer_pipeline import CAMformerPipeline, CAMformerConfig
from .attention_head import CAMformerAttentionHead, AttentionHeadConfig

__all__ = [
    # Pipeline stages
    'AssociationStage', 'AssociationConfig',
    'SelectionStage', 'SelectionConfig',
    'ContextualizationStage', 'ContextualizationConfig',
    # Full pipeline
    'CAMformerPipeline', 'CAMformerConfig',
    # High-level attention head
    'CAMformerAttentionHead', 'AttentionHeadConfig',
]
