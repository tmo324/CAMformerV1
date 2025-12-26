"""
Tiled BA-CAM Array Component

Implements the paper's tiled architecture:
- 64 tiles of 16x64 BA-CAM
- Per-tile Top-2/4 selection
- Cross-tile Top-32 selection
- Pipelined operation support
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.statistics import CounterStatistic, AccumulatorStatistic


@dataclass
class TiledBACAMConfig:
    """Configuration for Tiled BA-CAM array"""

    # Sequence and dimension
    seq_length: int = 1024
    head_dim: int = 64

    # Tile configuration (from paper)
    num_tiles: int = 64           # 1024 / 16 = 64 tiles
    rows_per_tile: int = 16       # Rows per tile
    cols_per_tile: int = 64       # Columns per tile (head_dim)

    # Programming configuration
    rows_per_program: int = 4     # Program 4 rows at a time

    # Timing (cycles) - from modules.csv
    cam_program_cycles: int = 1   # Per programming operation
    cam_search_cycles: int = 1    # Single-cycle search
    adc_cycles: int = 2           # ADC conversion
    scaler_cycles: int = 2        # Fixed scalers
    top2_cycles: int = 2          # Per-tile Top-2/4 selection
    ptop_reg_cycles: int = 1      # Potential top register
    top32_cycles: int = 6         # Cross-tile Top-32 selection

    # Power (mW) - from modules.csv at 45nm
    cam_power_mw: float = 13.15   # BA-CAM 16x64 power
    adc_power_mw: float = 15.2    # 16 ADCs total
    scaler_power_mw: float = 0.39 # 16 scalers total
    top2_power_mw: float = 6.08   # Top-2 selector
    ptop_reg_power_mw: float = 1.53  # Potential top register
    top32_power_mw: float = 52.08 # Top-32 selector
    query_buffer_power_mw: float = 0.80
    key_sram_power_mw: float = 28.4

    # Area (um²) - from modules.csv
    cam_area_um2: float = 2230.0
    adc_area_um2: float = 64000.0
    scaler_area_um2: float = 1338.0
    top2_area_um2: float = 8801.0
    ptop_reg_area_um2: float = 6042.0
    top32_area_um2: float = 67017.0
    query_buffer_area_um2: float = 2475.0
    key_sram_area_um2: float = 27729.0

    def get_programs_per_tile(self) -> int:
        """Number of programming operations per tile"""
        return self.rows_per_tile // self.rows_per_program

    def get_total_area_um2(self) -> float:
        """Get total association stage area"""
        return (self.cam_area_um2 + self.adc_area_um2 + self.scaler_area_um2 +
                self.top2_area_um2 + self.ptop_reg_area_um2 + self.top32_area_um2 +
                self.query_buffer_area_um2 + self.key_sram_area_um2)

    def get_total_area_mm2(self) -> float:
        """Get total area in mm²"""
        return self.get_total_area_um2() / 1e6


class TiledBACAM(Component):
    """
    Tiled BA-CAM Array implementing paper's architecture.

    Architecture:
    - 64 tiles, each 16x64 (16 tokens × 64-dim keys)
    - Per-tile operations: CAM program, search, ADC, scale, Top-2
    - Cross-tile: Top-32 selection every 16 tiles
    - Supports pipelined and non-pipelined modes
    """

    def __init__(self, name: str, config: Optional[TiledBACAMConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or TiledBACAMConfig()

        # Storage for each tile
        self._tiles: List[np.ndarray] = [
            np.zeros((self.config.rows_per_tile, self.config.cols_per_tile), dtype=np.int8)
            for _ in range(self.config.num_tiles)
        ]

        # Results storage
        self._tile_distances: List[np.ndarray] = []
        self._tile_top_indices: List[np.ndarray] = []
        self._global_top_indices: Optional[np.ndarray] = None
        self._global_top_distances: Optional[np.ndarray] = None

        # State
        self._keys_loaded = False
        self._num_keys = 0

        # Accumulated metrics
        self._total_cycles = 0
        self._total_energy_pj = 0.0

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("search_ops", CounterStatistic(
            "search_ops", "Search operations", "ops"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total cycles", "cycles"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total energy", "pJ"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"TiledBACAM initialized: {self.config.num_tiles} tiles × "
                         f"{self.config.rows_per_tile}×{self.config.cols_per_tile}")

    def write_keys(self, keys: np.ndarray, pipelined: bool = True) -> int:
        """
        Write keys to tiled CAM array.

        Args:
            keys: Key matrix (N, d_k), will be binarized
            pipelined: Use pipelined timing

        Returns:
            Total latency in cycles
        """
        num_keys = keys.shape[0]
        if num_keys > self.config.seq_length:
            raise ValueError(f"Too many keys: {num_keys} > {self.config.seq_length}")

        # Binarize keys
        binary_keys = np.where(keys > 0, 1, 0).astype(np.int8)

        # Distribute keys across tiles
        keys_per_tile = self.config.rows_per_tile
        for tile_idx in range(self.config.num_tiles):
            start_row = tile_idx * keys_per_tile
            end_row = min(start_row + keys_per_tile, num_keys)
            if start_row < num_keys:
                rows_to_write = end_row - start_row
                self._tiles[tile_idx][:rows_to_write] = binary_keys[start_row:end_row]

        self._keys_loaded = True
        self._num_keys = num_keys

        # Calculate write latency (programming happens during search in paper model)
        # Keys are loaded from Key SRAM, programmed 4 rows at a time
        # This is accounted for in search timing
        return 0  # Write latency included in search

    def search(self, query: np.ndarray, pipelined: bool = True) -> tuple:
        """
        Search all tiles with query.

        Implements the paper's tiled search flow:
        1. For each tile: program CAM, search, ADC, scale, Top-2
        2. Every 16 tiles: Top-32 selection
        3. Final Top-32 for global result

        Args:
            query: Query vector (d_k,), will be binarized
            pipelined: Use pipelined timing

        Returns:
            Tuple of (top_k_indices, top_k_distances, latency_cycles, energy_pj)
        """
        if not self._keys_loaded:
            raise RuntimeError("Keys not loaded")

        # Binarize query
        binary_query = np.where(query > 0, 1, 0).astype(np.int8)

        # Reset accumulators
        total_cycles = 0
        total_energy_pj = 0.0

        # Per-tile results
        all_distances = []
        all_indices = []

        cfg = self.config

        # Process each tile
        for tile_idx in range(cfg.num_tiles):
            tile_data = self._tiles[tile_idx]

            # --- Tile Operations ---

            # 1. Program CAM (4 rows at a time)
            for _ in range(cfg.get_programs_per_tile()):
                # Key SRAM read + CAM program
                total_energy_pj += cfg.key_sram_power_mw * cfg.cam_program_cycles
                total_energy_pj += cfg.cam_power_mw * cfg.cam_program_cycles
                if not pipelined:
                    total_cycles += cfg.cam_program_cycles
                else:
                    total_cycles += cfg.cam_program_cycles  # CAM program not pipelined

            # 2. CAM Search
            total_energy_pj += cfg.query_buffer_power_mw * cfg.cam_search_cycles
            total_energy_pj += cfg.cam_power_mw * cfg.cam_search_cycles
            if not pipelined:
                total_cycles += cfg.cam_search_cycles
            else:
                total_cycles += cfg.cam_search_cycles  # Search adds cycles

            # 3. ADC conversion
            total_energy_pj += cfg.adc_power_mw * cfg.adc_cycles
            if not pipelined:
                total_cycles += cfg.adc_cycles

            # 4. Fixed scalers
            total_energy_pj += cfg.scaler_power_mw * cfg.scaler_cycles
            if not pipelined:
                total_cycles += cfg.scaler_cycles

            # 5. Top-2 selection per tile
            total_energy_pj += cfg.top2_power_mw * cfg.top2_cycles
            if not pipelined:
                total_cycles += cfg.top2_cycles

            # 6. Potential top register
            total_energy_pj += cfg.ptop_reg_power_mw * cfg.ptop_reg_cycles
            if not pipelined:
                total_cycles += cfg.ptop_reg_cycles

            # Compute Hamming distances for this tile
            xor_result = np.bitwise_xor(tile_data, binary_query)
            distances = np.sum(xor_result, axis=1)

            # Get top-2 from this tile (lowest Hamming distance = highest similarity)
            top2_idx = np.argpartition(distances, 2)[:2]
            top2_idx = top2_idx[np.argsort(distances[top2_idx])]

            # Convert to global indices
            global_idx = tile_idx * cfg.rows_per_tile + top2_idx
            all_indices.extend(global_idx.tolist())
            all_distances.extend(distances[top2_idx].tolist())

            # 7. Top-32 every 16 tiles
            if tile_idx >= 32 and (tile_idx + 1) % 16 == 0:
                total_energy_pj += cfg.top32_power_mw * cfg.top32_cycles
                total_energy_pj += cfg.ptop_reg_power_mw * cfg.ptop_reg_cycles
                if not pipelined:
                    total_cycles += cfg.top32_cycles
                    total_cycles += cfg.ptop_reg_cycles

        # Pipeline drain for pipelined mode
        if pipelined:
            total_cycles += cfg.adc_cycles
            total_cycles += cfg.scaler_cycles
            total_cycles += cfg.top2_cycles
            total_cycles += cfg.ptop_reg_cycles

        # Store results
        all_indices = np.array(all_indices)
        all_distances = np.array(all_distances)

        # Final global Top-32 selection (done in normalization stage)
        # Just store the per-tile results here
        self._tile_top_indices = all_indices
        self._tile_distances = all_distances

        # Update statistics
        self.get_statistic("search_ops").add_data(1)
        self.get_statistic("total_cycles").add_data(total_cycles)
        self.get_statistic("total_energy").add_data(total_energy_pj)

        self._total_cycles = total_cycles
        self._total_energy_pj = total_energy_pj

        return all_indices, all_distances, total_cycles, total_energy_pj

    def get_top_k(self, k: int = 32) -> tuple:
        """
        Get global top-k indices from tile results.

        Args:
            k: Number of top results

        Returns:
            Tuple of (top_k_indices, top_k_distances)
        """
        if self._tile_top_indices is None:
            raise RuntimeError("No search results available")

        # Sort by distance and take top-k
        sorted_idx = np.argsort(self._tile_distances)[:k]
        top_k_indices = self._tile_top_indices[sorted_idx]
        top_k_distances = self._tile_distances[sorted_idx]

        return top_k_indices, top_k_distances

    def get_last_cycles(self) -> int:
        """Get cycles from last operation"""
        return self._total_cycles

    def get_last_energy_pj(self) -> float:
        """Get energy from last operation"""
        return self._total_energy_pj

    def clear(self) -> None:
        """Clear all state"""
        for tile in self._tiles:
            tile.fill(0)
        self._tile_distances = []
        self._tile_top_indices = []
        self._global_top_indices = None
        self._global_top_distances = None
        self._keys_loaded = False
        self._num_keys = 0
        self._total_cycles = 0
        self._total_energy_pj = 0.0

    def get_area_mm2(self) -> float:
        """Get total area"""
        return self.config.get_total_area_mm2()

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'num_tiles': self.config.num_tiles,
            'tile_size': f"{self.config.rows_per_tile}×{self.config.cols_per_tile}",
            'total_area_mm2': self.get_area_mm2(),
            'keys_loaded': self._num_keys,
        }

    def __str__(self) -> str:
        return f"TiledBACAM({self.name}, {self.config.num_tiles} tiles)"
