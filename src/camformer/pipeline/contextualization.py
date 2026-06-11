"""
Contextualization Stage

Third stage of CAMformer pipeline.
Computes Attention × Value using sparse attention weights.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic
from camformer.components.bf16_mac import BF16MAC, BF16MACConfig
from camformer.components.sram import SRAM, SRAMConfig
from camformer.components.buffers import OutputBuffer, BufferConfig


@dataclass
class ContextualizationConfig:
    """Configuration for Contextualization stage"""

    # Dimension parameters (updated to match paper)
    head_dim: int = 64  # d_v dimension
    k_value: int = 32   # Sparsity level (top-k)
    max_seq_length: int = 1024

    # BF16 MAC configuration
    mac_config: BF16MACConfig = field(default_factory=lambda: BF16MACConfig(
        num_mac_units=64,  # Parallel MAC units
    ))

    # V SRAM configuration
    v_sram_config: SRAMConfig = field(default_factory=lambda: SRAMConfig(
        num_rows=1024,
        num_cols=64,
        word_width_bits=16,  # BF16
    ))

    # Output buffer configuration
    output_buffer_depth: int = 8

    def __post_init__(self):
        """Sync V-SRAM config with stage parameters"""
        self.v_sram_config.num_rows = self.max_seq_length
        self.v_sram_config.num_cols = self.head_dim


class ContextualizationStage(Component):
    """
    Contextualization Stage for CAMformer.

    Computes weighted sum of Values:
    output[i] = Σ(j in top-k) attention_weight[i,j] * V[j]

    Takes advantage of sparsity from Selection stage:
    - Only k values accessed per query (vs N in dense attention)
    - Reduces memory traffic by factor of N/k
    """

    def __init__(self, name: str, config: Optional[ContextualizationConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or ContextualizationConfig()

        # Create subcomponents
        self._mac = BF16MAC(
            f"{name}_mac",
            config=self.config.mac_config
        )

        self._v_sram = SRAM(
            f"{name}_v_sram",
            config=self.config.v_sram_config
        )

        self._output_buffer = OutputBuffer(
            f"{name}_output_buf",
            config=BufferConfig(
                depth=self.config.output_buffer_depth,
                width=self.config.head_dim,
            )
        )

        # Register subcomponents
        self.set_subcomponent("mac", self._mac)
        self.set_subcomponent("v_sram", self._v_sram)
        self.set_subcomponent("output_buffer", self._output_buffer)

        # State
        self._values_loaded = False

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("outputs_computed", CounterStatistic(
            "outputs_computed", "Total output vectors computed", "outputs"
        ))
        self.add_statistic("mac_operations", AccumulatorStatistic(
            "mac_operations", "Total MAC operations", "ops"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total stage cycles", "cycles"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total stage energy", "pJ"
        ))
        self.add_statistic("sparse_v_reads", CounterStatistic(
            "sparse_v_reads", "Sparse V vector reads", "reads"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"ContextualizationStage initialized: "
                         f"head_dim={self.config.head_dim}, "
                         f"k={self.config.k_value}")

    def load_values(self, values: np.ndarray) -> int:
        """
        Load Value matrix into V-SRAM.

        Args:
            values: Value matrix (N, d_v) in BF16

        Returns:
            Latency in cycles
        """
        # Store values in SRAM (batch write starting from row 0)
        latency = self._v_sram.write_batch(0, values)

        self._values_loaded = True
        self._output.debug(f"Loaded {values.shape[0]} value vectors, "
                          f"latency={latency} cycles")

        return latency

    def compute_output(self, attention_weights: np.ndarray,
                       value_indices: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute single output vector using sparse attention.

        Args:
            attention_weights: Sparse attention weights (k,)
            value_indices: Indices of selected values (k,)

        Returns:
            Tuple of (output_vector, latency_cycles)
        """
        if not self._values_loaded:
            raise RuntimeError("Values not loaded in V-SRAM")

        k = len(attention_weights)
        d_v = self.config.head_dim

        # Initialize output
        output = np.zeros(d_v, dtype=np.float32)
        total_latency = 0

        # For each selected value
        for j in range(k):
            # Read V[index] from SRAM
            v_vector, read_latency = self._v_sram.read(int(value_indices[j]))
            total_latency += read_latency

            self.get_statistic("sparse_v_reads").add_data(1)

            # Accumulate: output += weight * V
            weight = attention_weights[j]
            output += weight * v_vector
            # Simple MAC latency estimation
            mac_latency = self.config.head_dim // self._mac.config.num_mac_units + 1
            total_latency += mac_latency

        # Track statistics
        self.get_statistic("outputs_computed").add_data(1)
        self.get_statistic("mac_operations").add_data(k * d_v)
        self.get_statistic("total_cycles").add_data(total_latency)

        return output, total_latency

    def compute_output_vectorized(self, attention_weights: np.ndarray,
                                   value_indices: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute single output vector - vectorized version.

        Reads all k values in parallel (if SRAM supports multiport),
        then computes weighted sum.

        Args:
            attention_weights: Sparse attention weights (k,)
            value_indices: Indices of selected values (k,)

        Returns:
            Tuple of (output_vector, latency_cycles)
        """
        if not self._values_loaded:
            raise RuntimeError("Values not loaded in V-SRAM")

        k = len(attention_weights)

        # Read all values (parallel access via selected rows)
        values, read_latency = self._v_sram.read_selected_rows(value_indices.astype(int))

        # Vectorized MAC: output = weights @ values
        output, mac_latency = self._mac.mac(attention_weights, values)

        total_latency = read_latency + mac_latency

        # Track statistics
        self.get_statistic("outputs_computed").add_data(1)
        self.get_statistic("mac_operations").add_data(k * self.config.head_dim)
        self.get_statistic("total_cycles").add_data(total_latency)
        self.get_statistic("sparse_v_reads").add_data(k)

        return output, total_latency

    def compute_all_outputs(self, attention_weights: np.ndarray,
                            value_indices: np.ndarray,
                            vectorized: bool = True) -> Tuple[np.ndarray, int]:
        """
        Compute all output vectors.

        Args:
            attention_weights: Sparse attention weights (N, k)
            value_indices: Indices of selected values (N, k)
            vectorized: Use vectorized computation

        Returns:
            Tuple of (output_matrix, total_latency)
        """
        num_queries = attention_weights.shape[0]
        d_v = self.config.head_dim

        outputs = np.zeros((num_queries, d_v), dtype=np.float32)
        total_latency = 0

        compute_fn = (self.compute_output_vectorized if vectorized
                      else self.compute_output)

        for i in range(num_queries):
            output, latency = compute_fn(
                attention_weights[i],
                value_indices[i]
            )
            outputs[i] = output
            total_latency += latency

        self._output.debug(f"Computed {num_queries} outputs, "
                          f"total_latency={total_latency} cycles")

        return outputs, total_latency

    def get_area_mm2(self) -> float:
        """Get total stage area"""
        return (self._mac.get_area_mm2() +
                self._v_sram.get_area_mm2() +
                self._output_buffer.get_area_mm2())

    def get_stage_stats(self) -> Dict[str, Any]:
        """Get stage statistics"""
        return {
            'area_mm2': self.get_area_mm2(),
            'head_dim': self.config.head_dim,
            'k_value': self.config.k_value,
            'mac_stats': self._mac.get_hardware_stats(),
            'v_sram_stats': self._v_sram.get_hardware_stats(),
        }

    def clear(self) -> None:
        """Clear stage state"""
        self._v_sram.clear()
        self._output_buffer.clear()
        self._values_loaded = False

    def __str__(self) -> str:
        return (f"ContextualizationStage({self.name}, "
                f"d_v={self.config.head_dim}, k={self.config.k_value})")
