"""
Memory Controller Component

Manages data movement between external DRAM and on-chip memory.
Handles tiling and prefetching for efficient attention computation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic


class MemoryAccessType(Enum):
    """Memory access types"""
    READ = "read"
    WRITE = "write"
    PREFETCH = "prefetch"


@dataclass
class MemoryControllerConfig:
    """Configuration for memory controller"""

    # Memory interface
    dram_bandwidth_gbps: float = 32.0  # GB/s bandwidth
    dram_latency_ns: float = 100.0     # ns latency

    # On-chip buffers
    input_buffer_kb: int = 64
    weight_buffer_kb: int = 128
    output_buffer_kb: int = 64

    # Timing (at 500 MHz = 2ns period)
    clock_period_ns: float = 2.0

    # Power parameters
    dram_read_energy_pj_per_byte: float = 20.0
    dram_write_energy_pj_per_byte: float = 25.0
    sram_access_energy_pj_per_byte: float = 0.5

    # Tiling parameters
    tile_size_tokens: int = 64
    tile_size_dim: int = 64

    def get_dram_latency_cycles(self) -> int:
        """Get DRAM latency in cycles"""
        return int(np.ceil(self.dram_latency_ns / self.clock_period_ns))

    def get_bandwidth_bytes_per_cycle(self) -> float:
        """Get bandwidth in bytes per cycle"""
        bytes_per_ns = self.dram_bandwidth_gbps  # GB/s = bytes/ns
        return bytes_per_ns * self.clock_period_ns


class MemoryController(Component):
    """
    Memory Controller for CAMformer.

    Manages:
    - DRAM ↔ on-chip data movement
    - Tiling strategy for large sequences
    - Prefetching for hiding memory latency
    - Buffer management for Q, K, V, and outputs
    """

    def __init__(self, name: str, config: Optional[MemoryControllerConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or MemoryControllerConfig()

        # Buffer tracking
        self._input_buffer_used = 0
        self._weight_buffer_used = 0
        self._output_buffer_used = 0

        # Pending transfers
        self._pending_reads: List[Dict] = []
        self._pending_writes: List[Dict] = []

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("dram_reads", CounterStatistic(
            "dram_reads", "Total DRAM read operations", "ops"
        ))
        self.add_statistic("dram_writes", CounterStatistic(
            "dram_writes", "Total DRAM write operations", "ops"
        ))
        self.add_statistic("bytes_read", AccumulatorStatistic(
            "bytes_read", "Total bytes read from DRAM", "bytes"
        ))
        self.add_statistic("bytes_written", AccumulatorStatistic(
            "bytes_written", "Total bytes written to DRAM", "bytes"
        ))
        self.add_statistic("dram_energy", AccumulatorStatistic(
            "dram_energy", "Total DRAM energy", "pJ"
        ))
        self.add_statistic("sram_energy", AccumulatorStatistic(
            "sram_energy", "Total SRAM buffer energy", "pJ"
        ))
        self.add_statistic("memory_cycles", AccumulatorStatistic(
            "memory_cycles", "Total memory access cycles", "cycles"
        ))
        self.add_statistic("stall_cycles", AccumulatorStatistic(
            "stall_cycles", "Cycles stalled on memory", "cycles"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"MemoryController initialized: "
                         f"bandwidth={self.config.dram_bandwidth_gbps} GB/s, "
                         f"latency={self.config.dram_latency_ns} ns")

    def load_tile(self, tile_type: str, num_tokens: int,
                  dimension: int, bytes_per_element: int = 2) -> Tuple[int, int]:
        """
        Load a tile from DRAM to on-chip buffer.

        Args:
            tile_type: Type of data ("Q", "K", "V", "output")
            num_tokens: Number of tokens in tile
            dimension: Dimension per token
            bytes_per_element: Bytes per element (2 for BF16)

        Returns:
            Tuple of (latency_cycles, bytes_transferred)
        """
        # Calculate data size
        num_bytes = num_tokens * dimension * bytes_per_element

        # Calculate transfer time
        bandwidth_bpc = self.config.get_bandwidth_bytes_per_cycle()
        transfer_cycles = int(np.ceil(num_bytes / bandwidth_bpc))
        latency_cycles = self.config.get_dram_latency_cycles() + transfer_cycles

        # Energy
        dram_energy = num_bytes * self.config.dram_read_energy_pj_per_byte
        sram_energy = num_bytes * self.config.sram_access_energy_pj_per_byte

        # Update statistics
        self.get_statistic("dram_reads").add_data(1)
        self.get_statistic("bytes_read").add_data(num_bytes)
        self.get_statistic("dram_energy").add_data(dram_energy)
        self.get_statistic("sram_energy").add_data(sram_energy)
        self.get_statistic("memory_cycles").add_data(latency_cycles)

        self._output.debug(f"Loaded {tile_type} tile: {num_tokens}x{dimension}, "
                          f"{num_bytes} bytes, {latency_cycles} cycles")

        return latency_cycles, num_bytes

    def store_tile(self, tile_type: str, num_tokens: int,
                   dimension: int, bytes_per_element: int = 2) -> Tuple[int, int]:
        """
        Store a tile from on-chip buffer to DRAM.

        Args:
            tile_type: Type of data
            num_tokens: Number of tokens
            dimension: Dimension per token
            bytes_per_element: Bytes per element

        Returns:
            Tuple of (latency_cycles, bytes_transferred)
        """
        num_bytes = num_tokens * dimension * bytes_per_element

        bandwidth_bpc = self.config.get_bandwidth_bytes_per_cycle()
        transfer_cycles = int(np.ceil(num_bytes / bandwidth_bpc))
        latency_cycles = self.config.get_dram_latency_cycles() + transfer_cycles

        dram_energy = num_bytes * self.config.dram_write_energy_pj_per_byte
        sram_energy = num_bytes * self.config.sram_access_energy_pj_per_byte

        self.get_statistic("dram_writes").add_data(1)
        self.get_statistic("bytes_written").add_data(num_bytes)
        self.get_statistic("dram_energy").add_data(dram_energy)
        self.get_statistic("sram_energy").add_data(sram_energy)
        self.get_statistic("memory_cycles").add_data(latency_cycles)

        self._output.debug(f"Stored {tile_type} tile: {num_tokens}x{dimension}, "
                          f"{num_bytes} bytes, {latency_cycles} cycles")

        return latency_cycles, num_bytes

    def calculate_tiling(self, seq_length: int, head_dim: int,
                         num_heads: int) -> Dict[str, Any]:
        """
        Calculate optimal tiling strategy.

        Args:
            seq_length: Total sequence length
            head_dim: Dimension per head
            num_heads: Number of attention heads

        Returns:
            Tiling strategy dictionary
        """
        tile_tokens = min(self.config.tile_size_tokens, seq_length)
        tile_dim = min(self.config.tile_size_dim, head_dim)

        num_token_tiles = int(np.ceil(seq_length / tile_tokens))
        num_dim_tiles = int(np.ceil(head_dim / tile_dim))

        return {
            'tile_tokens': tile_tokens,
            'tile_dim': tile_dim,
            'num_token_tiles': num_token_tiles,
            'num_dim_tiles': num_dim_tiles,
            'total_tiles': num_token_tiles * num_dim_tiles * num_heads,
            'seq_length': seq_length,
            'head_dim': head_dim,
            'num_heads': num_heads,
        }

    def estimate_memory_traffic(self, seq_length: int, head_dim: int,
                                 num_heads: int, k_value: int,
                                 bytes_per_element: int = 2) -> Dict[str, int]:
        """
        Estimate total memory traffic for attention computation.

        Args:
            seq_length: Sequence length N
            head_dim: Head dimension d_k/d_v
            num_heads: Number of heads
            k_value: Top-k sparsity
            bytes_per_element: Bytes per element

        Returns:
            Dictionary of memory traffic estimates
        """
        # Q, K loading (binarized for K in CAM)
        q_bytes = seq_length * head_dim * bytes_per_element * num_heads
        k_bytes = seq_length * head_dim * 1  # 1 bit per element for binary

        # V loading (full precision, but sparse access)
        # With top-k, we only need k values per query
        v_bytes_sparse = seq_length * k_value * head_dim * bytes_per_element * num_heads

        # Output writing
        output_bytes = seq_length * head_dim * bytes_per_element * num_heads

        return {
            'q_bytes': q_bytes,
            'k_bytes': k_bytes,
            'v_bytes_sparse': v_bytes_sparse,
            'output_bytes': output_bytes,
            'total_read_bytes': q_bytes + k_bytes + v_bytes_sparse,
            'total_write_bytes': output_bytes,
            'total_bytes': q_bytes + k_bytes + v_bytes_sparse + output_bytes,
        }

    def get_area_mm2(self) -> float:
        """Get memory controller area (minimal, mainly buffers)"""
        # Controller logic area estimate
        return 0.01  # 0.01 mm² for controller logic

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'dram_bandwidth_gbps': self.config.dram_bandwidth_gbps,
            'dram_latency_ns': self.config.dram_latency_ns,
            'input_buffer_kb': self.config.input_buffer_kb,
            'weight_buffer_kb': self.config.weight_buffer_kb,
            'output_buffer_kb': self.config.output_buffer_kb,
        }

    def __str__(self) -> str:
        return f"MemoryController({self.name}, {self.config.dram_bandwidth_gbps} GB/s)"
