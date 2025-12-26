"""
SoftMax Unit Component

Computes SoftMax normalization for sparse attention scores.
Uses hardware-efficient approximations for exp() and division.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic, AverageStatistic


@dataclass
class SoftMaxConfig:
    """Configuration for SoftMax unit"""

    # Input parameters
    max_input_size: int = 64     # Maximum number of inputs (top-k)

    # Implementation: LUT-based or piecewise linear approximation
    implementation: str = "lut"   # "lut" or "piecewise"
    lut_bits: int = 8            # Bits for LUT addressing

    # Timing (cycles at 500 MHz)
    exp_latency_cycles: int = 2   # Exponential computation
    add_latency_cycles: int = 1   # Accumulation
    div_latency_cycles: int = 4   # Division
    pipeline_stages: int = 8      # Total pipelined latency

    # Power parameters (28nm)
    exp_energy_pj: float = 0.5    # Per exponential
    add_energy_pj: float = 0.1    # Per addition
    div_energy_pj: float = 1.0    # Per division
    lut_energy_pj: float = 0.2    # Per LUT access

    # Area
    lut_area_um2: float = 1000.0  # LUT storage
    adder_area_um2: float = 100.0
    divider_area_um2: float = 500.0

    def get_area_mm2(self) -> float:
        """Get SoftMax unit area"""
        total_um2 = (self.lut_area_um2 +
                     self.max_input_size * self.adder_area_um2 +
                     self.divider_area_um2)
        return total_um2 * 1e-6


class SoftMaxUnit(Component):
    """
    Hardware SoftMax Unit.

    Computes softmax(x) = exp(x_i) / sum(exp(x_j))

    Uses LUT-based approximation for efficient hardware implementation.
    Supports sparse attention with variable-length input.
    """

    def __init__(self, name: str, config: Optional[SoftMaxConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or SoftMaxConfig()

        # Temperature for scaling (can be adjusted)
        self._temperature = 1.0

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("softmax_ops", CounterStatistic(
            "softmax_ops", "Total softmax operations", "ops"
        ))
        self.add_statistic("exp_computations", CounterStatistic(
            "exp_computations", "Total exp computations", "ops"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total softmax energy", "pJ"
        ))
        self.add_statistic("cycles", AccumulatorStatistic(
            "cycles", "Total softmax cycles", "cycles"
        ))
        self.add_statistic("avg_input_size", AverageStatistic(
            "avg_input_size", "Average input size", "elements"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"SoftMaxUnit initialized: max_input={self.config.max_input_size}, "
                         f"impl={self.config.implementation}")

    def set_temperature(self, temp: float) -> None:
        """Set softmax temperature"""
        self._temperature = max(0.01, temp)

    def compute(self, scores: np.ndarray,
                hamming_mode: bool = False,
                max_distance: int = 64) -> Tuple[np.ndarray, int]:
        """
        Compute softmax on input scores.

        Args:
            scores: Input scores (higher = more similar)
            hamming_mode: If True, convert Hamming distances to scores first
            max_distance: Maximum Hamming distance (for conversion)

        Returns:
            Tuple of (softmax_weights, latency_cycles)
        """
        n = len(scores)
        if n == 0:
            return np.array([]), 0

        # Convert Hamming distances to similarity scores if needed
        if hamming_mode:
            # Lower Hamming distance = higher similarity
            # Use negative distance scaled appropriately
            similarity = (max_distance - scores) / self._temperature
        else:
            similarity = scores / self._temperature

        # Numerical stability: subtract max
        similarity_shifted = similarity - np.max(similarity)

        # Compute exponentials
        exp_values = np.exp(similarity_shifted)

        # Sum and normalize
        exp_sum = np.sum(exp_values)
        softmax_weights = exp_values / (exp_sum + 1e-10)

        # Calculate hardware cost
        # Latency: pipelined exp + reduction + division
        latency = self.config.pipeline_stages

        # Energy
        energy = (n * self.config.exp_energy_pj +       # n exponentials
                  n * self.config.add_energy_pj +        # n additions for sum
                  n * self.config.div_energy_pj +        # n divisions
                  n * self.config.lut_energy_pj)         # n LUT accesses

        # Update statistics
        self.get_statistic("softmax_ops").add_data(1)
        self.get_statistic("exp_computations").add_data(n)
        self.get_statistic("energy").add_data(energy)
        self.get_statistic("cycles").add_data(latency)
        self.get_statistic("avg_input_size").add_data(n)

        self._output.debug(f"SoftMax computed: n={n}, latency={latency} cycles")

        return softmax_weights, latency

    def compute_sparse(self, scores: np.ndarray, indices: np.ndarray,
                       total_length: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute sparse softmax for selected indices.

        Args:
            scores: Scores for selected indices
            indices: Which indices these scores correspond to
            total_length: Total sequence length

        Returns:
            Tuple of (sparse_weights, indices, latency_cycles)
        """
        weights, latency = self.compute(scores)

        # In sparse attention, only return weights for selected indices
        # Other positions effectively have zero attention weight

        return weights, indices, latency

    def get_area_mm2(self) -> float:
        """Get SoftMax unit area"""
        return self.config.get_area_mm2()

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'max_input_size': self.config.max_input_size,
            'implementation': self.config.implementation,
            'temperature': self._temperature,
            'area_mm2': self.get_area_mm2(),
        }

    def __str__(self) -> str:
        return f"SoftMaxUnit({self.name}, max_size={self.config.max_input_size})"
