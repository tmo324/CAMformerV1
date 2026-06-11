"""
CAMformer Pipeline

Full attention pipeline integrating all three stages:
1. Association: Q*K^T similarity using BA-CAM
2. Selection: Top-K + SoftMax for sparse attention
3. Contextualization: Attention × V weighted sum
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic, AverageStatistic
from camformer.components.memory_controller import MemoryController, MemoryControllerConfig

from .association import AssociationStage, AssociationConfig
from .selection import SelectionStage, SelectionConfig
from .contextualization import ContextualizationStage, ContextualizationConfig


@dataclass
class CAMformerConfig:
    """Configuration for CAMformer pipeline"""

    # Model parameters
    seq_length: int = 512
    head_dim: int = 64
    num_heads: int = 8
    k_value: int = 32  # Top-k sparsity

    # Stage configurations
    association_config: Optional[AssociationConfig] = None
    selection_config: Optional[SelectionConfig] = None
    contextualization_config: Optional[ContextualizationConfig] = None

    # Memory configuration
    memory_config: Optional[MemoryControllerConfig] = None

    # Pipeline parameters
    enable_pipelining: bool = True
    prefetch_tiles: bool = True

    def __post_init__(self):
        """Initialize stage configs with consistent parameters"""
        if self.association_config is None:
            self.association_config = AssociationConfig(
                max_seq_length=self.seq_length,
                head_dim=self.head_dim,
            )

        if self.selection_config is None:
            self.selection_config = SelectionConfig(
                k_value=self.k_value,
                max_seq_length=self.seq_length,
            )

        if self.contextualization_config is None:
            self.contextualization_config = ContextualizationConfig(
                head_dim=self.head_dim,
                k_value=self.k_value,
                max_seq_length=self.seq_length,
            )

        if self.memory_config is None:
            self.memory_config = MemoryControllerConfig()


class CAMformerPipeline(Component):
    """
    CAMformer Attention Pipeline.

    Implements the full CAMformer attention mechanism:

    1. Association Stage (BA-CAM):
       - Store binarized K in CAM
       - For each Q, compute Hamming distances via parallel search
       - Single-cycle O(1) similarity computation per query

    2. Selection Stage (Top-K + SoftMax):
       - Select top-k most similar keys
       - Apply SoftMax to get sparse attention weights
       - Reduces O(N²) to O(N*k) for value aggregation

    3. Contextualization Stage (Sparse MatMul):
       - For each query, access only k value vectors
       - Compute weighted sum: output = Σ attention[j] * V[j]

    Key benefits:
    - O(N*d) CAM search vs O(N²*d) matrix multiply
    - O(N*k*d) value aggregation vs O(N²*d)
    - Significant memory traffic reduction
    """

    def __init__(self, name: str = "camformer",
                 config: Optional[CAMformerConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or CAMformerConfig()

        # Create pipeline stages
        self._association = AssociationStage(
            f"{name}_assoc",
            config=self.config.association_config
        )

        self._selection = SelectionStage(
            f"{name}_select",
            config=self.config.selection_config
        )

        self._contextualization = ContextualizationStage(
            f"{name}_context",
            config=self.config.contextualization_config
        )

        # Memory controller
        self._memory = MemoryController(
            f"{name}_memory",
            config=self.config.memory_config
        )

        # Register subcomponents
        self.set_subcomponent("association", self._association)
        self.set_subcomponent("selection", self._selection)
        self.set_subcomponent("contextualization", self._contextualization)
        self.set_subcomponent("memory", self._memory)

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup pipeline statistics"""
        self.add_statistic("attention_ops", CounterStatistic(
            "attention_ops", "Total attention operations", "ops"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total pipeline cycles", "cycles"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total energy consumption", "pJ"
        ))
        self.add_statistic("throughput_tokens_per_cycle", AverageStatistic(
            "throughput_tokens_per_cycle", "Throughput", "tokens/cycle"
        ))

    def _setup_impl(self) -> None:
        """Pipeline setup"""
        self._output.info(f"CAMformer Pipeline initialized: "
                         f"seq_len={self.config.seq_length}, "
                         f"head_dim={self.config.head_dim}, "
                         f"num_heads={self.config.num_heads}, "
                         f"k={self.config.k_value}")

    def forward_single_head(self, Q: np.ndarray, K: np.ndarray,
                            V: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass for single attention head.

        Args:
            Q: Query matrix (N, d_k)
            K: Key matrix (N, d_k)
            V: Value matrix (N, d_v)

        Returns:
            Tuple of (output, stats_dict)
        """
        N = Q.shape[0]
        stats = {
            'latency': {},
            'energy': {},
        }

        # Stage 1: Association
        # Load keys into BA-CAM
        key_load_latency = self._association.load_keys(K)
        stats['latency']['key_load'] = key_load_latency

        # Process all queries
        _, similarity_scores, assoc_latency = self._association.process_all_queries(Q)
        stats['latency']['association'] = assoc_latency

        # Stage 2: Selection
        # Select top-k and compute softmax
        _, indices, attention_weights, select_latency = self._selection.process_all_rows(
            similarity_scores
        )
        stats['latency']['selection'] = select_latency

        # Stage 3: Contextualization
        # Load values into V-SRAM
        value_load_latency = self._contextualization.load_values(V)
        stats['latency']['value_load'] = value_load_latency

        # Compute outputs
        outputs, context_latency = self._contextualization.compute_all_outputs(
            attention_weights, indices
        )
        stats['latency']['contextualization'] = context_latency

        # Calculate totals
        total_latency = (key_load_latency + assoc_latency + select_latency +
                        value_load_latency + context_latency)
        stats['latency']['total'] = total_latency

        # Memory traffic
        memory_traffic = self._memory.estimate_memory_traffic(
            N, self.config.head_dim, 1, self.config.k_value
        )
        stats['memory_traffic'] = memory_traffic

        # Update pipeline statistics
        self.get_statistic("attention_ops").add_data(1)
        self.get_statistic("total_cycles").add_data(total_latency)
        self.get_statistic("throughput_tokens_per_cycle").add_data(N / total_latency)

        return outputs, stats

    def forward(self, Q: np.ndarray, K: np.ndarray,
                V: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass for multi-head attention.

        Args:
            Q: Query tensor (N, num_heads, d_k) or (num_heads, N, d_k)
            K: Key tensor (N, num_heads, d_k) or (num_heads, N, d_k)
            V: Value tensor (N, num_heads, d_v) or (num_heads, N, d_v)

        Returns:
            Tuple of (output, aggregate_stats)
        """
        # Handle input format
        if Q.ndim == 2:
            # Single head, expand
            Q = Q[np.newaxis, :, :]
            K = K[np.newaxis, :, :]
            V = V[np.newaxis, :, :]
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure (num_heads, N, d) format
        if Q.shape[0] != self.config.num_heads:
            Q = Q.transpose(1, 0, 2)
            K = K.transpose(1, 0, 2)
            V = V.transpose(1, 0, 2)

        num_heads = Q.shape[0]
        N = Q.shape[1]
        d_v = V.shape[2]

        outputs = np.zeros((num_heads, N, d_v), dtype=np.float32)
        all_stats = []

        for h in range(num_heads):
            output, stats = self.forward_single_head(Q[h], K[h], V[h])
            outputs[h] = output
            all_stats.append(stats)

            # Clear state between heads
            self._association.clear()
            self._contextualization.clear()

        # Aggregate statistics
        agg_stats = self._aggregate_stats(all_stats)

        if squeeze_output:
            outputs = outputs.squeeze(0)

        return outputs, agg_stats

    def _aggregate_stats(self, stats_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate statistics across heads"""
        agg = {
            'latency': {},
            'energy': {},
            'memory_traffic': {},
            'num_heads': len(stats_list),
        }

        # Sum latencies
        for key in stats_list[0]['latency']:
            agg['latency'][key] = sum(s['latency'][key] for s in stats_list)

        # Sum memory traffic
        for key in stats_list[0]['memory_traffic']:
            agg['memory_traffic'][key] = sum(
                s['memory_traffic'][key] for s in stats_list
            )

        return agg

    def compare_with_dense(self, N: int, d: int) -> Dict[str, Any]:
        """
        Compare CAMformer vs dense attention complexity.

        Args:
            N: Sequence length
            d: Dimension

        Returns:
            Comparison dictionary
        """
        k = self.config.k_value

        # Dense attention
        dense_compute = N * N * d  # Q*K^T + Softmax + A*V
        dense_memory = 2 * N * d + N * N + N * d  # Q, K, attention, output

        # CAMformer (per head)
        cam_search = N * d  # CAM search (O(1) per query, N queries)
        topk_select = N * N  # Top-k selection (can be optimized)
        sparse_matmul = N * k * d  # Sparse A*V

        camformer_compute = cam_search + topk_select + sparse_matmul
        camformer_memory = 2 * N * d + N * k + N * d  # Q, K/CAM, sparse indices, output

        return {
            'dense': {
                'compute': dense_compute,
                'memory': dense_memory,
            },
            'camformer': {
                'compute': camformer_compute,
                'memory': camformer_memory,
            },
            'speedup': {
                'compute': dense_compute / camformer_compute,
                'memory': dense_memory / camformer_memory,
            },
            'params': {
                'N': N,
                'd': d,
                'k': k,
            }
        }

    def get_area_mm2(self) -> float:
        """Get total pipeline area"""
        return (self._association.get_area_mm2() +
                self._selection.get_area_mm2() +
                self._contextualization.get_area_mm2() +
                self._memory.get_area_mm2())

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'config': {
                'seq_length': self.config.seq_length,
                'head_dim': self.config.head_dim,
                'num_heads': self.config.num_heads,
                'k_value': self.config.k_value,
            },
            'area': {
                'total_mm2': self.get_area_mm2(),
                'association_mm2': self._association.get_area_mm2(),
                'selection_mm2': self._selection.get_area_mm2(),
                'contextualization_mm2': self._contextualization.get_area_mm2(),
            },
            'stages': {
                'association': self._association.get_stage_stats(),
                'selection': self._selection.get_stage_stats(),
                'contextualization': self._contextualization.get_stage_stats(),
            },
        }

    def clear(self) -> None:
        """Clear all pipeline state"""
        self._association.clear()
        self._contextualization.clear()

    def __str__(self) -> str:
        return (f"CAMformerPipeline({self.name}, "
                f"N={self.config.seq_length}, "
                f"d={self.config.head_dim}, "
                f"h={self.config.num_heads}, "
                f"k={self.config.k_value})")
