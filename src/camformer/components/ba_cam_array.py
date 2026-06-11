"""
BA-CAM (Binary Attention CAM) Array Component

The core component of CAMformer - implements binary attention using
Content Addressable Memory with voltage-domain matchline sensing.

Based on 10T1C cell design from CAMformer paper.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.event import Event, EventType
from camformer.core.statistics import (
    CounterStatistic, AccumulatorStatistic, AverageStatistic
)


@dataclass
class BACAMConfig:
    """Configuration for BA-CAM array"""

    # Array dimensions
    num_rows: int = 1024          # Number of rows (sequence length N) - updated to match paper
    num_cols: int = 64            # Number of columns (head dimension d_k)

    # Technology parameters (28nm)
    cell_area_um2: float = 0.325  # Cell area in μm²

    # Timing parameters (cycles at 500 MHz = 2ns period)
    write_latency_cycles: int = 2
    search_latency_cycles: int = 1  # Single-cycle search
    precharge_cycles: int = 1

    # Power parameters (from CAMformer paper)
    search_energy_fj_per_bit: float = 0.03    # fJ per bit for search
    write_energy_fj_per_bit: float = 0.08     # fJ per bit for write
    leakage_power_uw_per_cell: float = 0.001  # μW leakage per cell

    # Voltage parameters
    vdd: float = 0.9              # Supply voltage
    vml_precharge: float = 0.9    # Matchline precharge voltage

    # Matchline sensing
    ml_capacitance_ff: float = 50.0  # Matchline capacitance in fF

    def get_total_cells(self) -> int:
        """Get total number of cells"""
        return self.num_rows * self.num_cols

    def get_array_area_mm2(self) -> float:
        """Get array area in mm²"""
        area_um2 = self.get_total_cells() * self.cell_area_um2
        return area_um2 * 1e-6  # Convert to mm²

    def get_peripheral_area_mm2(self) -> float:
        """Estimate peripheral circuitry area (drivers, sense amps, etc.)"""
        # Peripherals typically add ~30% overhead
        return self.get_array_area_mm2() * 0.3

    def get_total_area_mm2(self) -> float:
        """Get total area including peripherals"""
        return self.get_array_area_mm2() + self.get_peripheral_area_mm2()


@dataclass
class BACAMCell:
    """
    Single BA-CAM cell (10T1C design).

    Stores a single bit and performs XOR-based matching.
    """
    stored_bit: int = 0

    def write(self, bit: int) -> None:
        """Write bit to cell"""
        self.stored_bit = bit & 1

    def match(self, search_bit: int) -> bool:
        """Check if stored bit matches search bit"""
        return self.stored_bit == (search_bit & 1)

    def get_mismatch(self, search_bit: int) -> int:
        """Get mismatch count (0 or 1)"""
        return 0 if self.match(search_bit) else 1


class BACAMArray(Component):
    """
    BA-CAM Array Component.

    Implements voltage-domain matchline sensing for binary attention:
    - Stores binarized Keys (K) in rows
    - Performs parallel similarity search with Query (Q)
    - Returns Hamming distances as matchline voltages

    Key features:
    - Single-cycle parallel search across all rows
    - Analog voltage encoding of Hamming distance
    - Supports multi-head attention (multiple arrays)
    """

    def __init__(self, name: str, config: Optional[BACAMConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or BACAMConfig()

        # Initialize cell array
        self._cells: np.ndarray = np.zeros(
            (self.config.num_rows, self.config.num_cols),
            dtype=np.int8
        )

        # Matchline voltages (analog output)
        self._matchline_voltages: np.ndarray = np.zeros(
            self.config.num_rows,
            dtype=np.float32
        )

        # Hamming distances (computed from matchline voltages)
        self._hamming_distances: np.ndarray = np.zeros(
            self.config.num_rows,
            dtype=np.int32
        )

        # State tracking
        self._keys_loaded = False
        self._num_keys = 0
        self._current_query: Optional[np.ndarray] = None

        # Initialize statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics tracking"""
        # Operation counters
        self.add_statistic("search_ops", CounterStatistic(
            "search_ops", "Number of search operations", "ops"
        ))
        self.add_statistic("write_ops", CounterStatistic(
            "write_ops", "Number of write operations", "ops"
        ))
        self.add_statistic("rows_written", CounterStatistic(
            "rows_written", "Number of rows written", "rows"
        ))

        # Energy tracking
        self.add_statistic("search_energy", AccumulatorStatistic(
            "search_energy", "Total search energy", "pJ"
        ))
        self.add_statistic("write_energy", AccumulatorStatistic(
            "write_energy", "Total write energy", "pJ"
        ))

        # Latency tracking
        self.add_statistic("total_search_cycles", AccumulatorStatistic(
            "total_search_cycles", "Total search cycles", "cycles"
        ))
        self.add_statistic("total_write_cycles", AccumulatorStatistic(
            "total_write_cycles", "Total write cycles", "cycles"
        ))

        # Average metrics
        self.add_statistic("avg_hamming_distance", AverageStatistic(
            "avg_hamming_distance", "Average Hamming distance", "bits"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"BA-CAM Array initialized: {self.config.num_rows}x{self.config.num_cols}")
        self._output.info(f"  Total cells: {self.config.get_total_cells():,}")
        self._output.info(f"  Array area: {self.config.get_array_area_mm2():.6f} mm²")
        self._output.info(f"  Total area: {self.config.get_total_area_mm2():.6f} mm²")

    def _finish_impl(self) -> None:
        """Component finish"""
        stats = self.get_statistics().get_summary()
        self._output.info(f"BA-CAM Array final statistics:")
        for name, info in stats.items():
            self._output.info(f"  {name}: {info['value']} {info['unit']}")

    def write_keys(self, keys: np.ndarray) -> int:
        """
        Write binarized keys to CAM array.

        Args:
            keys: Binary key matrix of shape (num_keys, key_dim)
                  Values should be in {-1, +1} or {0, 1}

        Returns:
            Latency in cycles
        """
        num_keys, key_dim = keys.shape

        if num_keys > self.config.num_rows:
            raise ValueError(f"Too many keys: {num_keys} > {self.config.num_rows}")
        if key_dim > self.config.num_cols:
            raise ValueError(f"Key dimension too large: {key_dim} > {self.config.num_cols}")

        # Convert {-1, +1} to {0, 1} if needed
        binary_keys = np.where(keys > 0, 1, 0).astype(np.int8)

        # Write to cell array
        self._cells[:num_keys, :key_dim] = binary_keys
        self._num_keys = num_keys
        self._keys_loaded = True

        # Update statistics
        write_stat = self.get_statistic("write_ops")
        write_stat.add_data(1)

        rows_stat = self.get_statistic("rows_written")
        rows_stat.add_data(num_keys)

        # Calculate energy (fJ to pJ conversion: 1 pJ = 1000 fJ)
        bits_written = num_keys * key_dim
        energy_fj = bits_written * self.config.write_energy_fj_per_bit
        energy_pj = energy_fj / 1000.0

        energy_stat = self.get_statistic("write_energy")
        energy_stat.add_data(energy_pj)

        # Calculate latency
        latency = self.config.write_latency_cycles * num_keys

        latency_stat = self.get_statistic("total_write_cycles")
        latency_stat.add_data(latency)

        self._output.debug(f"Wrote {num_keys} keys, latency={latency} cycles, energy={energy_pj:.4f} pJ")

        return latency

    def search(self, query: np.ndarray) -> tuple:
        """
        Search CAM array with query vector.

        Performs parallel Hamming distance computation across all rows.
        Returns matchline voltages proportional to similarity.

        Args:
            query: Binary query vector of shape (key_dim,)
                   Values should be in {-1, +1} or {0, 1}

        Returns:
            Tuple of (hamming_distances, matchline_voltages, latency_cycles)
        """
        if not self._keys_loaded:
            raise RuntimeError("No keys loaded in CAM array")

        key_dim = query.shape[0]
        if key_dim > self.config.num_cols:
            raise ValueError(f"Query dimension too large: {key_dim} > {self.config.num_cols}")

        # Convert {-1, +1} to {0, 1} if needed
        binary_query = np.where(query > 0, 1, 0).astype(np.int8)
        self._current_query = binary_query

        # Compute Hamming distances (XOR and sum)
        # XOR between each row and query, then count mismatches
        xor_result = np.bitwise_xor(
            self._cells[:self._num_keys, :key_dim],
            binary_query[:key_dim]
        )
        hamming_distances = np.sum(xor_result, axis=1)

        # Store results
        self._hamming_distances[:self._num_keys] = hamming_distances

        # Convert Hamming distance to matchline voltage
        # Higher similarity (lower Hamming distance) = higher voltage
        max_distance = key_dim
        normalized_similarity = 1.0 - (hamming_distances / max_distance)
        matchline_voltages = self.config.vml_precharge * normalized_similarity
        self._matchline_voltages[:self._num_keys] = matchline_voltages

        # Update statistics
        search_stat = self.get_statistic("search_ops")
        search_stat.add_data(1)

        # Calculate energy
        bits_searched = self._num_keys * key_dim
        energy_fj = bits_searched * self.config.search_energy_fj_per_bit
        energy_pj = energy_fj / 1000.0

        energy_stat = self.get_statistic("search_energy")
        energy_stat.add_data(energy_pj)

        # Record latency
        latency = (self.config.precharge_cycles +
                   self.config.search_latency_cycles)

        latency_stat = self.get_statistic("total_search_cycles")
        latency_stat.add_data(latency)

        # Track average Hamming distance
        avg_dist_stat = self.get_statistic("avg_hamming_distance")
        for dist in hamming_distances:
            avg_dist_stat.add_data(dist)

        self._output.debug(
            f"Search completed: {self._num_keys} rows, "
            f"avg_hamming={np.mean(hamming_distances):.2f}, "
            f"latency={latency} cycles"
        )

        return (
            self._hamming_distances[:self._num_keys].copy(),
            self._matchline_voltages[:self._num_keys].copy(),
            latency
        )

    def get_similarity_scores(self) -> np.ndarray:
        """
        Get similarity scores from matchline voltages.

        Converts analog matchline voltages to normalized similarity scores
        suitable for attention computation.

        Returns:
            Similarity scores in range [0, 1]
        """
        if not self._keys_loaded:
            return np.array([])

        # Normalize voltages to [0, 1] similarity scores
        scores = self._matchline_voltages[:self._num_keys] / self.config.vml_precharge
        return scores

    def get_top_k_indices(self, k: int) -> np.ndarray:
        """
        Get indices of top-k most similar rows.

        Args:
            k: Number of top matches to return

        Returns:
            Indices of top-k rows (highest similarity first)
        """
        if not self._keys_loaded:
            return np.array([])

        k = min(k, self._num_keys)

        # Lower Hamming distance = higher similarity
        # Use argpartition for efficient top-k
        distances = self._hamming_distances[:self._num_keys]
        partition_idx = np.argpartition(distances, k)[:k]
        top_k_idx = partition_idx[np.argsort(distances[partition_idx])]

        return top_k_idx

    def clear(self) -> None:
        """Clear CAM array contents"""
        self._cells.fill(0)
        self._matchline_voltages.fill(0)
        self._hamming_distances.fill(0)
        self._keys_loaded = False
        self._num_keys = 0
        self._current_query = None

    def get_area_mm2(self) -> float:
        """Get total array area in mm²"""
        return self.config.get_total_area_mm2()

    def get_leakage_power_mw(self) -> float:
        """Get leakage power in mW"""
        # Convert from μW per cell to mW total
        total_uw = (self.config.get_total_cells() *
                    self.config.leakage_power_uw_per_cell)
        return total_uw / 1000.0

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics summary"""
        return {
            'array_size': f"{self.config.num_rows}x{self.config.num_cols}",
            'total_cells': self.config.get_total_cells(),
            'array_area_mm2': self.config.get_array_area_mm2(),
            'total_area_mm2': self.config.get_total_area_mm2(),
            'leakage_power_mw': self.get_leakage_power_mw(),
            'search_latency_cycles': self.config.search_latency_cycles,
            'write_latency_cycles': self.config.write_latency_cycles,
            'keys_loaded': self._num_keys,
        }

    def __str__(self) -> str:
        return (f"BACAMArray({self.name}, "
                f"{self.config.num_rows}x{self.config.num_cols}, "
                f"keys={self._num_keys})")
