"""
Selection Stage

Second stage of CAMformer pipeline.
Performs Top-K selection on similarity scores and applies SoftMax normalization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic
from components.topk_sorter import TopKSorter, TopKConfig
from components.softmax import SoftMaxUnit, SoftMaxConfig


@dataclass
class SelectionConfig:
    """Configuration for Selection stage"""

    # Top-k parameters
    k_value: int = 32  # Number of top entries to select
    max_seq_length: int = 512

    # TopK sorter configuration
    topk_config: TopKConfig = field(default_factory=lambda: TopKConfig(
        k_value=32,
        max_input_size=512,
    ))

    # SoftMax configuration
    softmax_config: SoftMaxConfig = field(default_factory=lambda: SoftMaxConfig(
        max_input_size=32,  # k values
    ))

    def __post_init__(self):
        # Sync k_value
        self.topk_config.k_value = self.k_value
        self.topk_config.max_input_size = self.max_seq_length
        self.softmax_config.max_input_size = self.k_value


class SelectionStage(Component):
    """
    Selection Stage for CAMformer.

    Performs sparse attention selection:
    1. Select top-k similarity scores from Association stage
    2. Apply SoftMax to get attention weights
    3. Output sparse attention weights with indices

    This reduces O(N²) attention to O(N*k) for contextualization.
    """

    def __init__(self, name: str, config: Optional[SelectionConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or SelectionConfig()

        # Create subcomponents
        self._topk = TopKSorter(
            f"{name}_topk",
            config=self.config.topk_config
        )

        self._softmax = SoftMaxUnit(
            f"{name}_softmax",
            config=self.config.softmax_config
        )

        # Register subcomponents
        self.set_subcomponent("topk", self._topk)
        self.set_subcomponent("softmax", self._softmax)

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("selections_performed", CounterStatistic(
            "selections_performed", "Total selection operations", "ops"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total stage cycles", "cycles"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total stage energy", "pJ"
        ))
        self.add_statistic("sparsity_ratio", AccumulatorStatistic(
            "sparsity_ratio", "Average sparsity ratio", "ratio"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"SelectionStage initialized: "
                         f"k={self.config.k_value}, "
                         f"max_seq={self.config.max_seq_length}")

    def process_row(self, similarity_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Process a single row of similarity scores.

        Args:
            similarity_scores: Similarity scores from Association (N,)

        Returns:
            Tuple of (top_k_values, top_k_indices, attention_weights, latency_cycles)
        """
        # Top-k selection (sort returns top-k indices and values)
        top_indices, top_values, topk_latency = self._topk.sort(
            similarity_scores
        )

        # SoftMax on top-k values
        attention_weights, softmax_latency = self._softmax.compute(top_values)

        total_latency = topk_latency + softmax_latency

        # Update statistics
        self.get_statistic("selections_performed").add_data(1)
        self.get_statistic("total_cycles").add_data(total_latency)

        sparsity = self.config.k_value / len(similarity_scores)
        self.get_statistic("sparsity_ratio").add_data(sparsity)

        return top_values, top_indices, attention_weights, total_latency

    def process_all_rows(self, similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Process all rows of similarity matrix.

        Args:
            similarity_matrix: Full similarity matrix (N, N)

        Returns:
            Tuple of (all_top_values, all_top_indices, all_attention_weights, total_latency)
        """
        num_rows = similarity_matrix.shape[0]
        k = self.config.k_value

        all_values = np.zeros((num_rows, k), dtype=np.float32)
        all_indices = np.zeros((num_rows, k), dtype=np.int32)
        all_weights = np.zeros((num_rows, k), dtype=np.float32)
        total_latency = 0

        for i in range(num_rows):
            values, indices, weights, latency = self.process_row(
                similarity_matrix[i]
            )
            all_values[i] = values
            all_indices[i] = indices
            all_weights[i] = weights
            total_latency += latency

        self._output.debug(f"Processed {num_rows} rows, "
                          f"total_latency={total_latency} cycles")

        return all_values, all_indices, all_weights, total_latency

    def get_area_mm2(self) -> float:
        """Get total stage area"""
        return self._topk.get_area_mm2() + self._softmax.get_area_mm2()

    def get_stage_stats(self) -> Dict[str, Any]:
        """Get stage statistics"""
        return {
            'area_mm2': self.get_area_mm2(),
            'k_value': self.config.k_value,
            'topk_stats': self._topk.get_hardware_stats(),
            'softmax_stats': self._softmax.get_hardware_stats(),
        }

    def __str__(self) -> str:
        return f"SelectionStage({self.name}, k={self.config.k_value})"
