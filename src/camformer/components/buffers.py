"""
Buffer Components

Input and output buffers for pipeline stages.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic


@dataclass
class BufferConfig:
    """Configuration for buffers"""

    # Buffer size
    depth: int = 16              # Number of entries
    width: int = 64              # Width per entry (elements)
    element_bits: int = 16       # Bits per element (BF16)

    # Timing
    read_latency_cycles: int = 1
    write_latency_cycles: int = 1

    # Power
    read_energy_pj: float = 0.1
    write_energy_pj: float = 0.15

    # Area (register-based buffer)
    area_per_bit_um2: float = 0.5  # Register-based storage

    def get_capacity_bits(self) -> int:
        """Get capacity in bits"""
        return self.depth * self.width * self.element_bits

    def get_area_mm2(self) -> float:
        """Get buffer area"""
        return self.get_capacity_bits() * self.area_per_bit_um2 * 1e-6


class InputBuffer(Component):
    """
    Input buffer for receiving data from previous stage.

    FIFO-style buffer for pipeline input.
    """

    def __init__(self, name: str, config: Optional[BufferConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or BufferConfig()

        # FIFO storage
        self._buffer: deque = deque(maxlen=self.config.depth)

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("pushes", CounterStatistic(
            "pushes", "Total push operations", "ops"
        ))
        self.add_statistic("pops", CounterStatistic(
            "pops", "Total pop operations", "ops"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total buffer energy", "pJ"
        ))
        self.add_statistic("stalls", CounterStatistic(
            "stalls", "Buffer full stalls", "stalls"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"InputBuffer initialized: depth={self.config.depth}, "
                         f"width={self.config.width}")

    def push(self, data: np.ndarray) -> bool:
        """
        Push data into buffer.

        Args:
            data: Data to push

        Returns:
            True if successful, False if buffer full
        """
        if len(self._buffer) >= self.config.depth:
            self.get_statistic("stalls").add_data(1)
            return False

        self._buffer.append(data.copy())

        self.get_statistic("pushes").add_data(1)
        self.get_statistic("energy").add_data(self.config.write_energy_pj)

        return True

    def pop(self) -> Optional[np.ndarray]:
        """
        Pop data from buffer.

        Returns:
            Data if available, None if empty
        """
        if not self._buffer:
            return None

        data = self._buffer.popleft()

        self.get_statistic("pops").add_data(1)
        self.get_statistic("energy").add_data(self.config.read_energy_pj)

        return data

    def peek(self) -> Optional[np.ndarray]:
        """Peek at front of buffer without removing"""
        if not self._buffer:
            return None
        return self._buffer[0]

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self._buffer) >= self.config.depth

    def occupancy(self) -> int:
        """Get current occupancy"""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear buffer"""
        self._buffer.clear()

    def get_area_mm2(self) -> float:
        """Get buffer area"""
        return self.config.get_area_mm2()

    def __str__(self) -> str:
        return f"InputBuffer({self.name}, {self.occupancy()}/{self.config.depth})"


class OutputBuffer(Component):
    """
    Output buffer for sending data to next stage.

    FIFO-style buffer for pipeline output.
    """

    def __init__(self, name: str, config: Optional[BufferConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or BufferConfig()

        # FIFO storage
        self._buffer: deque = deque(maxlen=self.config.depth)

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("pushes", CounterStatistic(
            "pushes", "Total push operations", "ops"
        ))
        self.add_statistic("pops", CounterStatistic(
            "pops", "Total pop operations", "ops"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total buffer energy", "pJ"
        ))
        self.add_statistic("stalls", CounterStatistic(
            "stalls", "Buffer full stalls", "stalls"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"OutputBuffer initialized: depth={self.config.depth}, "
                         f"width={self.config.width}")

    def push(self, data: np.ndarray) -> bool:
        """Push data into buffer"""
        if len(self._buffer) >= self.config.depth:
            self.get_statistic("stalls").add_data(1)
            return False

        self._buffer.append(data.copy())

        self.get_statistic("pushes").add_data(1)
        self.get_statistic("energy").add_data(self.config.write_energy_pj)

        return True

    def pop(self) -> Optional[np.ndarray]:
        """Pop data from buffer"""
        if not self._buffer:
            return None

        data = self._buffer.popleft()

        self.get_statistic("pops").add_data(1)
        self.get_statistic("energy").add_data(self.config.read_energy_pj)

        return data

    def peek(self) -> Optional[np.ndarray]:
        """Peek at front of buffer"""
        if not self._buffer:
            return None
        return self._buffer[0]

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self._buffer) >= self.config.depth

    def occupancy(self) -> int:
        """Get current occupancy"""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear buffer"""
        self._buffer.clear()

    def get_area_mm2(self) -> float:
        """Get buffer area"""
        return self.config.get_area_mm2()

    def __str__(self) -> str:
        return f"OutputBuffer({self.name}, {self.occupancy()}/{self.config.depth})"
