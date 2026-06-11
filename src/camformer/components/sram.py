"""
SRAM Component

Standard SRAM for storing Values (V) and intermediate results.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic


@dataclass
class SRAMConfig:
    """Configuration for SRAM"""

    # Memory size (updated to match paper)
    num_rows: int = 1024
    num_cols: int = 64
    word_width_bits: int = 16  # BF16 for values

    # Timing (cycles at 500 MHz)
    read_latency_cycles: int = 1
    write_latency_cycles: int = 1

    # Power parameters (28nm)
    read_energy_pj_per_word: float = 0.5
    write_energy_pj_per_word: float = 0.6
    leakage_power_uw_per_kb: float = 10.0

    # Area (28nm SRAM)
    cell_area_um2: float = 0.127  # 6T SRAM cell

    def get_capacity_bits(self) -> int:
        """Get total capacity in bits"""
        return self.num_rows * self.num_cols * self.word_width_bits

    def get_capacity_kb(self) -> float:
        """Get capacity in KB"""
        return self.get_capacity_bits() / 8 / 1024

    def get_area_mm2(self) -> float:
        """Get SRAM area"""
        # Total bits
        total_bits = self.get_capacity_bits()
        # Area in um² (including ~30% peripheral overhead)
        array_area_um2 = total_bits * self.cell_area_um2
        total_area_um2 = array_area_um2 * 1.3
        return total_area_um2 * 1e-6


class SRAM(Component):
    """
    SRAM Memory Component.

    Used for storing:
    - Value (V) vectors for contextualization
    - Intermediate attention scores
    - Output vectors
    """

    def __init__(self, name: str, config: Optional[SRAMConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or SRAMConfig()

        # Initialize storage
        self._data: np.ndarray = np.zeros(
            (self.config.num_rows, self.config.num_cols),
            dtype=np.float32
        )

        # Track valid rows
        self._valid_rows = 0

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("reads", CounterStatistic(
            "reads", "Total read operations", "ops"
        ))
        self.add_statistic("writes", CounterStatistic(
            "writes", "Total write operations", "ops"
        ))
        self.add_statistic("read_energy", AccumulatorStatistic(
            "read_energy", "Total read energy", "pJ"
        ))
        self.add_statistic("write_energy", AccumulatorStatistic(
            "write_energy", "Total write energy", "pJ"
        ))
        self.add_statistic("read_cycles", AccumulatorStatistic(
            "read_cycles", "Total read cycles", "cycles"
        ))
        self.add_statistic("write_cycles", AccumulatorStatistic(
            "write_cycles", "Total write cycles", "cycles"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"SRAM initialized: {self.config.num_rows}x{self.config.num_cols}, "
                         f"{self.config.get_capacity_kb():.2f} KB")

    def write(self, row: int, data: np.ndarray) -> int:
        """
        Write data to a row.

        Args:
            row: Row index
            data: Data to write

        Returns:
            Latency in cycles
        """
        if row >= self.config.num_rows:
            raise ValueError(f"Row {row} out of range")

        cols = min(len(data), self.config.num_cols)
        self._data[row, :cols] = data[:cols]
        self._valid_rows = max(self._valid_rows, row + 1)

        # Statistics
        num_words = cols
        self.get_statistic("writes").add_data(1)
        self.get_statistic("write_energy").add_data(
            num_words * self.config.write_energy_pj_per_word
        )
        self.get_statistic("write_cycles").add_data(
            self.config.write_latency_cycles
        )

        return self.config.write_latency_cycles

    def write_batch(self, start_row: int, data: np.ndarray) -> int:
        """
        Write multiple rows.

        Args:
            start_row: Starting row index
            data: 2D array of data

        Returns:
            Total latency in cycles
        """
        num_rows = data.shape[0]
        if start_row + num_rows > self.config.num_rows:
            raise ValueError("Write would exceed SRAM capacity")

        cols = min(data.shape[1], self.config.num_cols)
        self._data[start_row:start_row + num_rows, :cols] = data[:, :cols]
        self._valid_rows = max(self._valid_rows, start_row + num_rows)

        # Statistics (row-by-row writes)
        latency = num_rows * self.config.write_latency_cycles
        num_words = num_rows * cols
        self.get_statistic("writes").add_data(num_rows)
        self.get_statistic("write_energy").add_data(
            num_words * self.config.write_energy_pj_per_word
        )
        self.get_statistic("write_cycles").add_data(latency)

        return latency

    def read(self, row: int) -> tuple:
        """
        Read data from a row.

        Args:
            row: Row index

        Returns:
            Tuple of (data, latency_cycles)
        """
        if row >= self.config.num_rows:
            raise ValueError(f"Row {row} out of range")

        data = self._data[row].copy()

        # Statistics
        self.get_statistic("reads").add_data(1)
        self.get_statistic("read_energy").add_data(
            self.config.num_cols * self.config.read_energy_pj_per_word
        )
        self.get_statistic("read_cycles").add_data(
            self.config.read_latency_cycles
        )

        return data, self.config.read_latency_cycles

    def read_batch(self, rows: np.ndarray) -> tuple:
        """
        Read multiple rows.

        Args:
            rows: Array of row indices

        Returns:
            Tuple of (data_array, latency_cycles)
        """
        data = self._data[rows].copy()

        # Statistics
        num_rows = len(rows)
        latency = num_rows * self.config.read_latency_cycles
        num_words = num_rows * self.config.num_cols
        self.get_statistic("reads").add_data(num_rows)
        self.get_statistic("read_energy").add_data(
            num_words * self.config.read_energy_pj_per_word
        )
        self.get_statistic("read_cycles").add_data(latency)

        return data, latency

    def read_selected_rows(self, indices: np.ndarray) -> tuple:
        """
        Read specific rows (for sparse attention).

        Args:
            indices: Array of row indices to read

        Returns:
            Tuple of (data_array, latency_cycles)
        """
        return self.read_batch(indices)

    def clear(self) -> None:
        """Clear SRAM contents"""
        self._data.fill(0)
        self._valid_rows = 0

    def get_area_mm2(self) -> float:
        """Get SRAM area"""
        return self.config.get_area_mm2()

    def get_leakage_power_mw(self) -> float:
        """Get leakage power"""
        return self.config.leakage_power_uw_per_kb * self.config.get_capacity_kb() / 1000.0

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'size': f"{self.config.num_rows}x{self.config.num_cols}",
            'capacity_kb': self.config.get_capacity_kb(),
            'area_mm2': self.get_area_mm2(),
            'valid_rows': self._valid_rows,
        }

    def __str__(self) -> str:
        return f"SRAM({self.name}, {self.config.num_rows}x{self.config.num_cols})"
