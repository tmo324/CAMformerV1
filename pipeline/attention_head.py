"""
CAMformer Attention Head

High-level attention head abstraction with optional linear projections.
Provides a complete transformer attention head implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic

from .camformer_pipeline import CAMformerPipeline, CAMformerConfig


@dataclass
class AttentionHeadConfig:
    """Configuration for CAMformer Attention Head"""

    # Model dimensions
    hidden_dim: int = 512  # Model hidden dimension
    head_dim: int = 64     # Per-head dimension
    num_heads: int = 8     # Number of attention heads

    # Sparsity
    k_value: int = 32      # Top-k for sparse attention

    # Sequence
    max_seq_length: int = 512

    # Projection (for full transformer layer)
    include_projections: bool = False  # Include W_Q, W_K, W_V, W_O

    # Hardware
    pipeline_config: Optional[CAMformerConfig] = None

    def __post_init__(self):
        if self.pipeline_config is None:
            self.pipeline_config = CAMformerConfig(
                seq_length=self.max_seq_length,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                k_value=self.k_value,
            )


class CAMformerAttentionHead(Component):
    """
    CAMformer Attention Head.

    Implements a complete multi-head attention layer with:
    - Optional Q/K/V projections
    - Sparse attention via CAMformer pipeline
    - Optional output projection

    For evaluating CAMformer hardware without projections,
    set include_projections=False (default).
    """

    def __init__(self, name: str = "attention_head",
                 config: Optional[AttentionHeadConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or AttentionHeadConfig()

        # Create CAMformer pipeline
        self._pipeline = CAMformerPipeline(
            f"{name}_pipeline",
            config=self.config.pipeline_config
        )

        # Projection weights (if enabled)
        if self.config.include_projections:
            self._init_projections()

        # Register subcomponents
        self.set_subcomponent("pipeline", self._pipeline)

        # Setup statistics
        self._setup_statistics()

    def _init_projections(self) -> None:
        """Initialize projection weights"""
        h = self.config.hidden_dim
        d = self.config.head_dim
        n_heads = self.config.num_heads

        # W_Q, W_K, W_V: (hidden_dim, num_heads * head_dim)
        self._W_Q = np.random.randn(h, n_heads * d).astype(np.float32) * 0.02
        self._W_K = np.random.randn(h, n_heads * d).astype(np.float32) * 0.02
        self._W_V = np.random.randn(h, n_heads * d).astype(np.float32) * 0.02

        # W_O: (num_heads * head_dim, hidden_dim)
        self._W_O = np.random.randn(n_heads * d, h).astype(np.float32) * 0.02

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("forward_passes", CounterStatistic(
            "forward_passes", "Total forward passes", "passes"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total cycles", "cycles"
        ))

        if self.config.include_projections:
            self.add_statistic("projection_cycles", AccumulatorStatistic(
                "projection_cycles", "Projection cycles", "cycles"
            ))
            self.add_statistic("projection_ops", AccumulatorStatistic(
                "projection_ops", "Projection MACs", "ops"
            ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"CAMformerAttentionHead initialized: "
                         f"hidden={self.config.hidden_dim}, "
                         f"heads={self.config.num_heads}, "
                         f"d={self.config.head_dim}, "
                         f"k={self.config.k_value}")

    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Project input to Q, K, V.

        Args:
            X: Input tensor (N, hidden_dim)

        Returns:
            Tuple of (Q, K, V, projection_cycles)
            Q, K, V shapes: (num_heads, N, head_dim)
        """
        N = X.shape[0]
        n_heads = self.config.num_heads
        d = self.config.head_dim

        # Matrix multiplications
        Q = X @ self._W_Q  # (N, n_heads * d)
        K = X @ self._W_K
        V = X @ self._W_V

        # Reshape to multi-head format
        Q = Q.reshape(N, n_heads, d).transpose(1, 0, 2)  # (n_heads, N, d)
        K = K.reshape(N, n_heads, d).transpose(1, 0, 2)
        V = V.reshape(N, n_heads, d).transpose(1, 0, 2)

        # Estimate cycles: N * hidden * head_dim * 3 projections
        ops = 3 * N * self.config.hidden_dim * (n_heads * d)
        cycles = ops // 64  # Assume 64 MACs per cycle

        return Q, K, V, cycles

    def _project_output(self, attention_output: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Project attention output back to hidden dimension.

        Args:
            attention_output: (num_heads, N, head_dim) or (N, num_heads * head_dim)

        Returns:
            Tuple of (output, projection_cycles)
        """
        n_heads = self.config.num_heads
        d = self.config.head_dim

        # Reshape from multi-head format
        if attention_output.ndim == 3:
            N = attention_output.shape[1]
            concat = attention_output.transpose(1, 0, 2).reshape(N, n_heads * d)
        else:
            N = attention_output.shape[0]
            concat = attention_output

        # Output projection
        output = concat @ self._W_O

        # Estimate cycles
        ops = N * (n_heads * d) * self.config.hidden_dim
        cycles = ops // 64

        return output, cycles

    def forward(self, X: np.ndarray,
                Q: Optional[np.ndarray] = None,
                K: Optional[np.ndarray] = None,
                V: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through attention head.

        Args:
            X: Input tensor (N, hidden_dim) - used if include_projections=True
            Q: Optional pre-computed queries (num_heads, N, d_k)
            K: Optional pre-computed keys (num_heads, N, d_k)
            V: Optional pre-computed values (num_heads, N, d_v)

        Returns:
            Tuple of (output, stats)
        """
        stats = {'latency': {}}
        total_cycles = 0

        # Project if needed
        if self.config.include_projections and X is not None:
            Q, K, V, proj_cycles = self._project_qkv(X)
            stats['latency']['qkv_projection'] = proj_cycles
            total_cycles += proj_cycles
            self.get_statistic("projection_cycles").add_data(proj_cycles)
        elif Q is None or K is None or V is None:
            raise ValueError("Must provide either X (with projections) or Q, K, V")

        # CAMformer attention
        attention_output, pipeline_stats = self._pipeline.forward(Q, K, V)
        stats['latency']['attention'] = pipeline_stats['latency']['total']
        stats['pipeline'] = pipeline_stats
        total_cycles += pipeline_stats['latency']['total']

        # Output projection if needed
        if self.config.include_projections:
            output, out_proj_cycles = self._project_output(attention_output)
            stats['latency']['output_projection'] = out_proj_cycles
            total_cycles += out_proj_cycles
            self.get_statistic("projection_cycles").add_data(out_proj_cycles)
        else:
            output = attention_output

        stats['latency']['total'] = total_cycles

        # Update statistics
        self.get_statistic("forward_passes").add_data(1)
        self.get_statistic("total_cycles").add_data(total_cycles)

        return output, stats

    def forward_attention_only(self, Q: np.ndarray, K: np.ndarray,
                               V: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass with pre-projected Q, K, V (no projections).

        This is the primary interface for hardware evaluation.

        Args:
            Q: Query tensor (num_heads, N, d_k) or (N, d_k) for single head
            K: Key tensor
            V: Value tensor

        Returns:
            Tuple of (output, stats)
        """
        # Direct pipeline forward
        output, stats = self._pipeline.forward(Q, K, V)

        self.get_statistic("forward_passes").add_data(1)
        self.get_statistic("total_cycles").add_data(stats['latency']['total'])

        return output, stats

    def get_area_mm2(self) -> float:
        """Get total area"""
        return self._pipeline.get_area_mm2()

    def get_head_stats(self) -> Dict[str, Any]:
        """Get comprehensive head statistics"""
        stats = {
            'config': {
                'hidden_dim': self.config.hidden_dim,
                'head_dim': self.config.head_dim,
                'num_heads': self.config.num_heads,
                'k_value': self.config.k_value,
                'include_projections': self.config.include_projections,
            },
            'area_mm2': self.get_area_mm2(),
            'pipeline': self._pipeline.get_pipeline_stats(),
        }

        if self.config.include_projections:
            # Projection weight sizes
            stats['projection_params'] = {
                'W_Q_shape': self._W_Q.shape,
                'W_K_shape': self._W_K.shape,
                'W_V_shape': self._W_V.shape,
                'W_O_shape': self._W_O.shape,
                'total_params': sum(w.size for w in
                                   [self._W_Q, self._W_K, self._W_V, self._W_O]),
            }

        return stats

    def compare_with_dense(self) -> Dict[str, Any]:
        """Compare with dense attention"""
        return self._pipeline.compare_with_dense(
            self.config.max_seq_length,
            self.config.head_dim
        )

    def clear(self) -> None:
        """Clear state"""
        self._pipeline.clear()

    def __str__(self) -> str:
        return (f"CAMformerAttentionHead({self.name}, "
                f"h={self.config.num_heads}, "
                f"d={self.config.head_dim}, "
                f"k={self.config.k_value})")
