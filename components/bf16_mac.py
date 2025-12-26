"""
BF16 MAC (Multiply-Accumulate) Component

Performs BFloat16 multiply-accumulate for Attention × Value computation.
Used in the contextualization stage of CAMformer.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic


@dataclass
class BF16MACConfig:
    """Configuration for BF16 MAC unit"""

    # MAC array dimensions
    num_mac_units: int = 64      # Number of parallel MAC units
    accumulator_width: int = 32  # FP32 accumulator for precision

    # Timing (cycles at 500 MHz)
    multiply_latency_cycles: int = 1
    accumulate_latency_cycles: int = 1
    pipeline_depth: int = 2  # Total MAC pipeline depth

    # Power parameters (28nm, BF16)
    mac_energy_pj: float = 0.4   # Energy per MAC operation
    register_energy_pj: float = 0.05  # Accumulator register

    # Area (28nm)
    mac_area_um2: float = 200.0  # Per MAC unit
    register_area_um2: float = 50.0  # Per accumulator

    def get_area_mm2(self) -> float:
        """Get MAC array area"""
        mac_area = self.num_mac_units * self.mac_area_um2
        reg_area = self.num_mac_units * self.register_area_um2
        return (mac_area + reg_area) * 1e-6


class BF16MAC(Component):
    """
    BF16 MAC Array for Contextualization.

    Computes weighted sum: output = sum(attention_weights[i] * V[i])

    Features:
    - Parallel MAC units for efficient computation
    - FP32 accumulator for numerical stability
    - Supports sparse attention (selective V access)
    """

    def __init__(self, name: str, config: Optional[BF16MACConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or BF16MACConfig()

        # Accumulators
        self._accumulators: np.ndarray = np.zeros(
            self.config.num_mac_units,
            dtype=np.float32
        )

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("mac_ops", CounterStatistic(
            "mac_ops", "Total MAC operations", "ops"
        ))
        self.add_statistic("vectors_processed", CounterStatistic(
            "vectors_processed", "Total vectors processed", "vectors"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total MAC energy", "pJ"
        ))
        self.add_statistic("cycles", AccumulatorStatistic(
            "cycles", "Total MAC cycles", "cycles"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"BF16MAC initialized: {self.config.num_mac_units} MAC units")

    def reset_accumulators(self) -> None:
        """Reset accumulators to zero"""
        self._accumulators.fill(0)

    def mac(self, weights: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Perform weighted accumulation.

        Computes: acc += sum(weights[i] * values[i])

        Args:
            weights: Attention weights (k,)
            values: Value vectors (k, d_v)

        Returns:
            Tuple of (accumulated_result, latency_cycles)
        """
        k = len(weights)
        d_v = values.shape[1] if len(values.shape) > 1 else len(values)

        # Compute weighted sum
        if len(values.shape) == 1:
            result = weights[0] * values
        else:
            # Weighted sum: weights^T @ values
            result = np.sum(weights[:, np.newaxis] * values, axis=0)

        # Store in accumulators
        acc_size = min(d_v, self.config.num_mac_units)
        self._accumulators[:acc_size] = result[:acc_size]

        # Calculate hardware cost
        # Each weight-value pair requires d_v MAC operations
        total_macs = k * d_v

        # Parallelized across MAC units
        mac_batches = int(np.ceil(d_v / self.config.num_mac_units))
        latency = k * mac_batches * self.config.pipeline_depth

        # Energy
        energy = total_macs * self.config.mac_energy_pj

        # Update statistics
        self.get_statistic("mac_ops").add_data(total_macs)
        self.get_statistic("vectors_processed").add_data(k)
        self.get_statistic("energy").add_data(energy)
        self.get_statistic("cycles").add_data(latency)

        self._output.debug(f"MAC: k={k}, d_v={d_v}, macs={total_macs}, latency={latency}")

        return result, latency

    def mac_sparse(self, weights: np.ndarray, values: np.ndarray,
                   indices: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Perform sparse weighted accumulation.

        Only uses values at specified indices.

        Args:
            weights: Attention weights for selected positions
            values: Full value matrix (N, d_v)
            indices: Which positions to use

        Returns:
            Tuple of (accumulated_result, latency_cycles)
        """
        # Select values at sparse indices
        selected_values = values[indices]

        return self.mac(weights, selected_values)

    def get_result(self) -> np.ndarray:
        """Get accumulated result"""
        return self._accumulators.copy()

    def get_area_mm2(self) -> float:
        """Get MAC array area"""
        return self.config.get_area_mm2()

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'num_mac_units': self.config.num_mac_units,
            'pipeline_depth': self.config.pipeline_depth,
            'area_mm2': self.get_area_mm2(),
        }

    def __str__(self) -> str:
        return f"BF16MAC({self.name}, {self.config.num_mac_units} units)"
