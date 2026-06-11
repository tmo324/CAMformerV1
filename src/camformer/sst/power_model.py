"""
Power × Cycles Energy Model

Energy model matching Ben's implementation:
E = P (mW) × cycles = pJ (at 1 GHz)

This model uses hardware module specifications from the paper
to calculate energy as Power × Active Cycles.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.paper_hardware import PipelineMode


@dataclass
class HardwareModuleSpec:
    """Specification for a hardware module"""
    name: str
    power_mw: float      # Active power in mW
    area_um2: float      # Area in um²
    delay_cycles: int    # Latency in cycles
    count: int = 1       # Number of instances

    def energy_pj(self, times: int = 1) -> float:
        """Calculate energy in pJ: P (mW) × cycles × count × times"""
        return self.power_mw * self.delay_cycles * self.count * times

    def total_power_mw(self) -> float:
        """Total power including all instances"""
        return self.power_mw * self.count

    def total_area_um2(self) -> float:
        """Total area including all instances"""
        return self.area_um2 * self.count


# Hardware module specifications from paper (45nm)
MODULES = {
    # Association Stage
    "query_buffer": HardwareModuleSpec("Query Buffer", 0.80, 2475.0, 1),
    "key_sram": HardwareModuleSpec("Key SRAM", 28.4, 27729.0, 1),
    "ba_cam": HardwareModuleSpec("BA-CAM 16×64", 13.15, 2230.0, 1),
    "adc": HardwareModuleSpec("6b ADCs", 0.95, 4000.0, 2, count=16),
    "scaler": HardwareModuleSpec("Fixed Scalers", 0.0245, 83.6, 2, count=16),
    "top2": HardwareModuleSpec("Top2", 6.08, 8801.0, 2),
    "ptop_reg": HardwareModuleSpec("Ptop Reg", 1.53, 6042.0, 1),

    # Normalization Stage
    "top32": HardwareModuleSpec("Top32", 52.08, 67017.0, 6),
    "softmax_lut": HardwareModuleSpec("SoftMax LUT", 1.42, 1612.0, 1),
    "softmax_div": HardwareModuleSpec("SoftMax Div", 0.071, 442.0, 3),
    "softmax_acc": HardwareModuleSpec("SoftMax Acc", 1.12, 2880.0, 1),
    "output_buffer": HardwareModuleSpec("Output Buff", 1.55, 5991.0, 1),

    # Contextualization Stage
    "value_sram": HardwareModuleSpec("Value SRAM", 34.7, 44820.0, 1),
    "bf16_mac": HardwareModuleSpec("BF16 MACs", 1.12, 2880.0, 4, count=8),

    # Off-chip
    "dram": HardwareModuleSpec("DRAM", 12.43, 0.0, 24),
}


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    cycles: int = 0
    energy_pj: float = 0.0
    module_energy: Dict[str, float] = field(default_factory=dict)

    def add_module(self, module_name: str, times: int = 1) -> None:
        """Add energy for a module"""
        if module_name in MODULES:
            module = MODULES[module_name]
            energy = module.energy_pj(times)
            self.energy_pj += energy
            self.module_energy[module_name] = self.module_energy.get(module_name, 0) + energy

    def add_cycles(self, cycles: int) -> None:
        """Add cycles"""
        self.cycles += cycles


class PowerCyclesModel:
    """
    Power × Cycles energy model for CAMformer.

    Calculates energy and timing matching the paper's methodology.
    Supports different pipeline modes.
    """

    def __init__(self, seq_length: int = 1024, head_dim: int = 64,
                 k_value: int = 32, num_tiles: int = 64,
                 rows_per_tile: int = 16, c_par: int = 8):
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.k_value = k_value
        self.num_tiles = num_tiles
        self.rows_per_tile = rows_per_tile
        self.c_par = c_par  # MAC parallelism
        self.rows_per_program = 4

        # Stage metrics
        self.association = StageMetrics()
        self.normalization = StageMetrics()
        self.contextualization = StageMetrics()

        # Pipeline mode
        self.pipeline_mode = PipelineMode.REALISTIC

    def reset(self) -> None:
        """Reset all metrics"""
        self.association = StageMetrics()
        self.normalization = StageMetrics()
        self.contextualization = StageMetrics()

    def run_association(self, pipelined: bool = True) -> StageMetrics:
        """
        Calculate association stage metrics.

        Implements tiled CAM architecture:
        - 64 tiles × (4 program + 1 search + ADC + scale + Top-2)
        - Top-32 every 16 tiles
        """
        self.association = StageMetrics()
        metrics = self.association

        programs_per_tile = self.rows_per_tile // self.rows_per_program

        for tile in range(self.num_tiles):
            # Program CAM (4 rows at a time)
            for _ in range(programs_per_tile):
                metrics.add_module("key_sram", 1)
                metrics.add_module("ba_cam", 1)
                metrics.add_cycles(MODULES["ba_cam"].delay_cycles)

            # Search
            metrics.add_module("query_buffer", 1)
            metrics.add_module("ba_cam", 1)
            metrics.add_cycles(MODULES["ba_cam"].delay_cycles)

            # ADC, Scaler, Top-2, Ptop Reg (pipelined or not)
            metrics.add_module("adc", 1)
            metrics.add_module("scaler", 1)
            metrics.add_module("top2", 1)
            metrics.add_module("ptop_reg", 1)

            if not pipelined:
                metrics.add_cycles(MODULES["adc"].delay_cycles)
                metrics.add_cycles(MODULES["scaler"].delay_cycles)
                metrics.add_cycles(MODULES["top2"].delay_cycles)
                metrics.add_cycles(MODULES["ptop_reg"].delay_cycles)

            # Top-32 every 16 tiles (after tile 32)
            if tile >= 32 and (tile + 1) % 16 == 0:
                metrics.add_module("top32", 1)
                metrics.add_module("ptop_reg", 1)
                if not pipelined:
                    metrics.add_cycles(MODULES["top32"].delay_cycles)
                    metrics.add_cycles(MODULES["ptop_reg"].delay_cycles)

            # DRAM prefetch (energy only, overlapped)
            metrics.add_module("dram", 1)
            metrics.add_module("value_sram", 1)

        # Pipeline drain
        if pipelined:
            metrics.add_cycles(MODULES["adc"].delay_cycles)
            metrics.add_cycles(MODULES["scaler"].delay_cycles)
            metrics.add_cycles(MODULES["top2"].delay_cycles)
            metrics.add_cycles(MODULES["ptop_reg"].delay_cycles)

        return metrics

    def run_normalization(self, pipelined: bool = False) -> StageMetrics:
        """
        Calculate normalization stage metrics.

        Implements softmax on k elements:
        - Final Top-32 selection
        - Softmax LUT + Accumulator for each element
        - Division for normalization
        """
        self.normalization = StageMetrics()
        metrics = self.normalization

        # Final Top-32
        metrics.add_module("top32", 1)
        metrics.add_cycles(MODULES["top32"].delay_cycles)

        # Softmax loop: exp lookup and accumulate
        for _ in range(self.k_value):
            metrics.add_module("softmax_lut", 1)
            metrics.add_module("softmax_acc", 1)
            metrics.add_module("output_buffer", 1)
            metrics.add_cycles(MODULES["softmax_acc"].delay_cycles)

        # Division loop
        for i in range(self.k_value):
            metrics.add_module("output_buffer", 1)  # Read
            metrics.add_module("softmax_div", 1)
            metrics.add_module("output_buffer", 1)  # Write
            if not pipelined:
                metrics.add_cycles(MODULES["softmax_div"].delay_cycles)
            else:
                metrics.add_cycles(1)

        if pipelined:
            metrics.add_cycles(MODULES["softmax_div"].delay_cycles)

        return metrics

    def run_contextualization(self, pipelined: bool = True) -> StageMetrics:
        """
        Calculate contextualization stage metrics.

        Implements sparse matmul:
        - 64 × 32 MACs / 8 parallel = 256 iterations
        - Each iteration: Value SRAM read + MAC
        """
        self.contextualization = StageMetrics()
        metrics = self.contextualization

        # Number of MAC iterations
        num_iterations = (self.head_dim * self.k_value) // self.c_par

        for _ in range(num_iterations):
            metrics.add_module("value_sram", 1)
            metrics.add_module("bf16_mac", 1)
            if pipelined:
                metrics.add_cycles(1)  # Pipelined: 1 cycle per iteration
            else:
                metrics.add_cycles(MODULES["bf16_mac"].delay_cycles)

        # Pipeline drain
        if pipelined:
            metrics.add_cycles(MODULES["bf16_mac"].delay_cycles)

        return metrics

    def run_attention(self, mode: Optional[PipelineMode] = None) -> Dict[str, Any]:
        """
        Run complete attention and return metrics.

        Args:
            mode: Pipeline mode

        Returns:
            Dictionary with cycles, energy, and breakdown
        """
        mode = mode or self.pipeline_mode
        self.reset()

        if mode == PipelineMode.NO_PIPELINE:
            self.run_association(pipelined=False)
            self.run_normalization(pipelined=False)
            self.run_contextualization(pipelined=False)
        elif mode == PipelineMode.FULL_PIPELINE:
            self.run_association(pipelined=True)
            self.run_normalization(pipelined=True)
            self.run_contextualization(pipelined=True)
        else:  # REALISTIC
            self.run_association(pipelined=True)
            self.run_normalization(pipelined=False)
            self.run_contextualization(pipelined=True)

        return self.get_metrics()

    def get_total_cycles(self) -> int:
        """Get total cycles across all stages"""
        return (self.association.cycles +
                self.normalization.cycles +
                self.contextualization.cycles)

    def get_total_energy_pj(self) -> float:
        """Get total energy in pJ"""
        return (self.association.energy_pj +
                self.normalization.energy_pj +
                self.contextualization.energy_pj)

    def get_total_energy_nj(self) -> float:
        """Get total energy in nJ"""
        return self.get_total_energy_pj() / 1000.0

    def get_onchip_energy_pj(self) -> float:
        """Get on-chip energy (excluding DRAM)"""
        total = self.get_total_energy_pj()
        dram_energy = (self.association.module_energy.get("dram", 0) +
                       self.normalization.module_energy.get("dram", 0) +
                       self.contextualization.module_energy.get("dram", 0))
        return total - dram_energy

    def get_dram_energy_pj(self) -> float:
        """Get DRAM energy"""
        return (self.association.module_energy.get("dram", 0) +
                self.normalization.module_energy.get("dram", 0) +
                self.contextualization.module_energy.get("dram", 0))

    def get_area_mm2(self) -> float:
        """Get total on-chip area in mm²"""
        total_um2 = sum(m.total_area_um2() for name, m in MODULES.items()
                        if name != "dram")
        return total_um2 / 1e6

    def get_energy_breakdown(self) -> Dict[str, float]:
        """Get energy breakdown by module in pJ"""
        breakdown = {}
        for stage in [self.association, self.normalization, self.contextualization]:
            for module, energy in stage.module_energy.items():
                breakdown[module] = breakdown.get(module, 0) + energy
        return breakdown

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary"""
        total_cycles = self.get_total_cycles()
        max_stage = max(self.association.cycles,
                        self.normalization.cycles,
                        self.contextualization.cycles)

        return {
            'cycles': {
                'total': total_cycles,
                'association': self.association.cycles,
                'normalization': self.normalization.cycles,
                'contextualization': self.contextualization.cycles,
                'max_stage': max_stage,
            },
            'energy_pj': {
                'total': self.get_total_energy_pj(),
                'onchip': self.get_onchip_energy_pj(),
                'dram': self.get_dram_energy_pj(),
            },
            'energy_nj': {
                'total': self.get_total_energy_nj(),
                'onchip': self.get_onchip_energy_pj() / 1000.0,
                'dram': self.get_dram_energy_pj() / 1000.0,
            },
            'area_mm2': self.get_area_mm2(),
            'breakdown_pj': self.get_energy_breakdown(),
        }
