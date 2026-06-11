"""
Top-K Sorter Component

Selects top-k most similar entries from BA-CAM search results.
Critical for sparsified attention in CAMformer.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic, AverageStatistic


@dataclass
class TopKConfig:
    """Configuration for Top-K sorter"""

    # Sorter parameters
    max_input_size: int = 512    # Maximum number of inputs to sort
    k_value: int = 64            # Number of top entries to select

    # Implementation: bitonic sort or tree-based comparison
    sorter_type: str = "bitonic"  # "bitonic" or "tree"

    # Timing (cycles at 500 MHz)
    # Bitonic sort: O(log²N) stages, pipelined
    comparison_latency_cycles: int = 1
    pipeline_stages: int = 10  # For 512 inputs: ceil(log2(512)^2 / 2) ≈ 40, but pipelined

    # Power parameters (28nm)
    comparison_energy_pj: float = 0.05  # Per comparison
    register_energy_pj: float = 0.02    # Per register access

    # Area
    comparator_area_um2: float = 50.0   # Per comparator
    register_area_um2: float = 5.0      # Per register bit

    def get_num_comparators(self) -> int:
        """Get number of parallel comparators"""
        # Bitonic sort uses N/2 comparators per stage
        return self.max_input_size // 2

    def get_area_mm2(self) -> float:
        """Get sorter area"""
        # Comparators
        comp_area = self.get_num_comparators() * self.comparator_area_um2
        # Registers for storing intermediate results
        reg_bits = self.max_input_size * 32  # 32-bit indices + values
        reg_area = reg_bits * self.register_area_um2
        total_um2 = comp_area + reg_area
        return total_um2 * 1e-6


class TopKSorter(Component):
    """
    Top-K Sorter for selecting highest similarity entries.

    Uses bitonic sorting network for efficient hardware implementation.
    Returns indices and scores of top-k entries for sparse attention.
    """

    def __init__(self, name: str, config: Optional[TopKConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or TopKConfig()

        # Current k value (can be adjusted per layer)
        self._current_k = self.config.k_value

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("sort_ops", CounterStatistic(
            "sort_ops", "Total sort operations", "ops"
        ))
        self.add_statistic("comparisons", CounterStatistic(
            "comparisons", "Total comparisons", "ops"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total sorter energy", "pJ"
        ))
        self.add_statistic("cycles", AccumulatorStatistic(
            "cycles", "Total sorter cycles", "cycles"
        ))
        self.add_statistic("avg_input_size", AverageStatistic(
            "avg_input_size", "Average input size", "elements"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"TopKSorter initialized: max_input={self.config.max_input_size}, "
                         f"k={self.config.k_value}")

    def set_k(self, k: int) -> None:
        """
        Set k value for top-k selection.

        Args:
            k: Number of top entries to select
        """
        self._current_k = min(k, self.config.max_input_size)

    def sort(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Perform top-k selection on input scores.

        Args:
            scores: Array of similarity scores (higher = more similar)

        Returns:
            Tuple of (top_k_indices, top_k_scores, latency_cycles)
        """
        n = len(scores)
        k = min(self._current_k, n)

        # Use numpy's argpartition for efficient top-k
        # In hardware, this would be a bitonic sorting network
        if k >= n:
            # All elements selected
            sorted_indices = np.argsort(scores)[::-1]
            top_k_indices = sorted_indices
            top_k_scores = scores[sorted_indices]
        else:
            # Partial sort for top-k
            partition_indices = np.argpartition(scores, -k)[-k:]
            # Sort the top-k by score
            sorted_order = np.argsort(scores[partition_indices])[::-1]
            top_k_indices = partition_indices[sorted_order]
            top_k_scores = scores[top_k_indices]

        # Calculate hardware cost
        # Bitonic sort: O(log²N) comparisons
        num_stages = int(np.ceil(np.log2(max(n, 2)) ** 2 / 2))
        comparisons_per_stage = n // 2
        total_comparisons = num_stages * comparisons_per_stage

        # Latency (pipelined)
        latency = self.config.pipeline_stages

        # Energy
        energy = (total_comparisons * self.config.comparison_energy_pj +
                  n * self.config.register_energy_pj)

        # Update statistics
        self.get_statistic("sort_ops").add_data(1)
        self.get_statistic("comparisons").add_data(total_comparisons)
        self.get_statistic("energy").add_data(energy)
        self.get_statistic("cycles").add_data(latency)
        self.get_statistic("avg_input_size").add_data(n)

        self._output.debug(f"TopK sort: n={n}, k={k}, latency={latency} cycles")

        return top_k_indices, top_k_scores, latency

    def sort_by_hamming(self, hamming_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sort by Hamming distance (lower = more similar).

        Args:
            hamming_distances: Array of Hamming distances

        Returns:
            Tuple of (top_k_indices, top_k_distances, latency_cycles)
        """
        # Convert to similarity scores (negate so lower distance = higher score)
        max_dist = np.max(hamming_distances) if len(hamming_distances) > 0 else 1
        similarity_scores = max_dist - hamming_distances

        indices, scores, latency = self.sort(similarity_scores)

        # Convert back to distances
        distances = max_dist - scores

        return indices, distances, latency

    def get_area_mm2(self) -> float:
        """Get sorter area"""
        return self.config.get_area_mm2()

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'max_input_size': self.config.max_input_size,
            'k_value': self._current_k,
            'num_comparators': self.config.get_num_comparators(),
            'area_mm2': self.get_area_mm2(),
        }

    def __str__(self) -> str:
        return f"TopKSorter({self.name}, k={self._current_k})"
