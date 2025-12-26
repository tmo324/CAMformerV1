"""
CAMformer Energy Model

Comprehensive energy modeling for CAMformer SST simulation.
Uses centralized technology parameters from core/technology.py.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.technology import TechnologyNode, DEFAULT_TECH, TECH_28NM, TECH_45NM


@dataclass
class EnergyConfig:
    """
    Energy configuration for CAMformer.

    All energies in picojoules (pJ).
    Uses centralized technology node parameters for consistency.
    """

    # Technology node (default: 45nm from CAMformer paper)
    technology: TechnologyNode = field(default_factory=lambda: TECH_45NM)

    # Clock parameters
    clock_freq_hz: float = 500e6                  # 500 MHz

    # Derived properties from technology node
    @property
    def tech_node_nm(self) -> int:
        """Get technology node in nm"""
        return self.technology.node_nm

    @property
    def cam_search_energy_fj_per_bit(self) -> float:
        return self.technology.cam_search_energy_fj_per_bit

    @property
    def cam_write_energy_fj_per_bit(self) -> float:
        return self.technology.cam_write_energy_fj_per_bit

    @property
    def cam_leakage_uw_per_cell(self) -> float:
        return self.technology.cam_leakage_uw_per_cell

    @property
    def adc_energy_pj_per_conversion(self) -> float:
        return self.technology.adc_energy_pj_per_conversion

    @property
    def topk_comparison_energy_pj(self) -> float:
        return self.technology.comparison_energy_pj

    @property
    def topk_register_energy_pj(self) -> float:
        return self.technology.register_energy_pj

    @property
    def softmax_exp_energy_pj(self) -> float:
        return self.technology.softmax_exp_energy_pj

    @property
    def softmax_add_energy_pj(self) -> float:
        return self.technology.softmax_add_energy_pj

    @property
    def softmax_div_energy_pj(self) -> float:
        return self.technology.softmax_div_energy_pj

    @property
    def softmax_lut_energy_pj(self) -> float:
        return self.technology.softmax_lut_energy_pj

    @property
    def mac_energy_pj(self) -> float:
        return self.technology.mac_energy_pj

    @property
    def mac_register_energy_pj(self) -> float:
        return self.technology.mac_register_energy_pj

    @property
    def sram_read_energy_pj_per_word(self) -> float:
        return self.technology.sram_read_energy_pj_per_word

    @property
    def sram_write_energy_pj_per_word(self) -> float:
        return self.technology.sram_write_energy_pj_per_word

    @property
    def sram_leakage_uw_per_kb(self) -> float:
        return self.technology.sram_leakage_uw_per_kb

    @property
    def buffer_read_energy_pj(self) -> float:
        return self.technology.buffer_read_energy_pj

    @property
    def buffer_write_energy_pj(self) -> float:
        return self.technology.buffer_write_energy_pj

    @property
    def dram_read_energy_pj_per_byte(self) -> float:
        return self.technology.dram_read_energy_pj_per_byte

    @property
    def dram_write_energy_pj_per_byte(self) -> float:
        return self.technology.dram_write_energy_pj_per_byte

    def get_clock_period_ns(self) -> float:
        """Get clock period in nanoseconds"""
        return 1e9 / self.clock_freq_hz

    def get_tech_summary(self) -> str:
        """Get technology summary string"""
        return f"{self.technology.name} ({self.technology.node_nm}nm, VDD={self.technology.vdd_v}V)"


@dataclass
class EnergyBreakdown:
    """Breakdown of energy consumption by component"""

    # Per-stage energy (pJ)
    cam_search_energy_pj: float = 0.0
    cam_write_energy_pj: float = 0.0
    adc_energy_pj: float = 0.0
    topk_energy_pj: float = 0.0
    softmax_energy_pj: float = 0.0
    mac_energy_pj: float = 0.0
    sram_energy_pj: float = 0.0
    buffer_energy_pj: float = 0.0
    dram_energy_pj: float = 0.0

    # Leakage (for active time)
    leakage_energy_pj: float = 0.0

    @property
    def total_energy_pj(self) -> float:
        """Total energy in pJ"""
        return (self.cam_search_energy_pj + self.cam_write_energy_pj +
                self.adc_energy_pj + self.topk_energy_pj +
                self.softmax_energy_pj + self.mac_energy_pj +
                self.sram_energy_pj + self.buffer_energy_pj +
                self.dram_energy_pj + self.leakage_energy_pj)

    @property
    def total_energy_nj(self) -> float:
        """Total energy in nJ"""
        return self.total_energy_pj / 1000.0

    @property
    def total_energy_uj(self) -> float:
        """Total energy in μJ"""
        return self.total_energy_pj / 1e6

    @property
    def total_energy_mj(self) -> float:
        """Total energy in mJ"""
        return self.total_energy_pj / 1e9

    def get_breakdown_dict(self) -> Dict[str, float]:
        """Get breakdown as dictionary (pJ)"""
        return {
            'cam_search': self.cam_search_energy_pj,
            'cam_write': self.cam_write_energy_pj,
            'adc': self.adc_energy_pj,
            'topk': self.topk_energy_pj,
            'softmax': self.softmax_energy_pj,
            'mac': self.mac_energy_pj,
            'sram': self.sram_energy_pj,
            'buffer': self.buffer_energy_pj,
            'dram': self.dram_energy_pj,
            'leakage': self.leakage_energy_pj,
            'total': self.total_energy_pj,
        }

    def get_percentage_breakdown(self) -> Dict[str, float]:
        """Get breakdown as percentages"""
        total = self.total_energy_pj
        if total == 0:
            return {}

        breakdown = self.get_breakdown_dict()
        return {k: (v / total * 100) for k, v in breakdown.items() if k != 'total'}

    def __add__(self, other: 'EnergyBreakdown') -> 'EnergyBreakdown':
        """Add two energy breakdowns"""
        return EnergyBreakdown(
            cam_search_energy_pj=self.cam_search_energy_pj + other.cam_search_energy_pj,
            cam_write_energy_pj=self.cam_write_energy_pj + other.cam_write_energy_pj,
            adc_energy_pj=self.adc_energy_pj + other.adc_energy_pj,
            topk_energy_pj=self.topk_energy_pj + other.topk_energy_pj,
            softmax_energy_pj=self.softmax_energy_pj + other.softmax_energy_pj,
            mac_energy_pj=self.mac_energy_pj + other.mac_energy_pj,
            sram_energy_pj=self.sram_energy_pj + other.sram_energy_pj,
            buffer_energy_pj=self.buffer_energy_pj + other.buffer_energy_pj,
            dram_energy_pj=self.dram_energy_pj + other.dram_energy_pj,
            leakage_energy_pj=self.leakage_energy_pj + other.leakage_energy_pj,
        )


class EnergyTracker:
    """
    Tracks energy consumption during simulation.

    Collects energy from all components and provides aggregate metrics.
    """

    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
        self._breakdown = EnergyBreakdown()
        self._query_count = 0
        self._active_cycles = 0

    def reset(self) -> None:
        """Reset all energy counters"""
        self._breakdown = EnergyBreakdown()
        self._query_count = 0
        self._active_cycles = 0

    def add_cam_search(self, num_bits: int) -> float:
        """Add CAM search energy"""
        energy_fj = num_bits * self.config.cam_search_energy_fj_per_bit
        energy_pj = energy_fj / 1000.0
        self._breakdown.cam_search_energy_pj += energy_pj
        return energy_pj

    def add_cam_write(self, num_bits: int) -> float:
        """Add CAM write energy"""
        energy_fj = num_bits * self.config.cam_write_energy_fj_per_bit
        energy_pj = energy_fj / 1000.0
        self._breakdown.cam_write_energy_pj += energy_pj
        return energy_pj

    def add_adc(self, num_conversions: int) -> float:
        """Add ADC conversion energy"""
        energy_pj = num_conversions * self.config.adc_energy_pj_per_conversion
        self._breakdown.adc_energy_pj += energy_pj
        return energy_pj

    def add_topk(self, num_comparisons: int, num_registers: int) -> float:
        """Add Top-K sorter energy"""
        energy_pj = (num_comparisons * self.config.topk_comparison_energy_pj +
                     num_registers * self.config.topk_register_energy_pj)
        self._breakdown.topk_energy_pj += energy_pj
        return energy_pj

    def add_softmax(self, num_elements: int) -> float:
        """Add SoftMax energy"""
        energy_pj = num_elements * (self.config.softmax_exp_energy_pj +
                                     self.config.softmax_add_energy_pj +
                                     self.config.softmax_div_energy_pj +
                                     self.config.softmax_lut_energy_pj)
        self._breakdown.softmax_energy_pj += energy_pj
        return energy_pj

    def add_mac(self, num_macs: int) -> float:
        """Add MAC operation energy"""
        energy_pj = num_macs * self.config.mac_energy_pj
        self._breakdown.mac_energy_pj += energy_pj
        return energy_pj

    def add_sram_read(self, num_words: int) -> float:
        """Add SRAM read energy"""
        energy_pj = num_words * self.config.sram_read_energy_pj_per_word
        self._breakdown.sram_energy_pj += energy_pj
        return energy_pj

    def add_sram_write(self, num_words: int) -> float:
        """Add SRAM write energy"""
        energy_pj = num_words * self.config.sram_write_energy_pj_per_word
        self._breakdown.sram_energy_pj += energy_pj
        return energy_pj

    def add_buffer_access(self, reads: int, writes: int) -> float:
        """Add buffer access energy"""
        energy_pj = (reads * self.config.buffer_read_energy_pj +
                     writes * self.config.buffer_write_energy_pj)
        self._breakdown.buffer_energy_pj += energy_pj
        return energy_pj

    def add_dram_read(self, num_bytes: int) -> float:
        """Add DRAM read energy"""
        energy_pj = num_bytes * self.config.dram_read_energy_pj_per_byte
        self._breakdown.dram_energy_pj += energy_pj
        return energy_pj

    def add_dram_write(self, num_bytes: int) -> float:
        """Add DRAM write energy"""
        energy_pj = num_bytes * self.config.dram_write_energy_pj_per_byte
        self._breakdown.dram_energy_pj += energy_pj
        return energy_pj

    def add_leakage(self, cycles: int, leakage_power_mw: float) -> float:
        """Add leakage energy for active cycles"""
        # Power (mW) * time (ns) = energy (pJ)
        time_ns = cycles * self.config.get_clock_period_ns()
        energy_pj = leakage_power_mw * time_ns
        self._breakdown.leakage_energy_pj += energy_pj
        self._active_cycles += cycles
        return energy_pj

    def record_query(self) -> None:
        """Record a query processed"""
        self._query_count += 1

    def get_breakdown(self) -> EnergyBreakdown:
        """Get energy breakdown"""
        return self._breakdown

    def get_total_energy_pj(self) -> float:
        """Get total energy in pJ"""
        return self._breakdown.total_energy_pj

    def get_total_energy_mj(self) -> float:
        """Get total energy in mJ"""
        return self._breakdown.total_energy_mj

    def get_queries_per_mj(self) -> float:
        """Get energy efficiency in queries per mJ"""
        energy_mj = self.get_total_energy_mj()
        if energy_mj == 0:
            return 0
        return self._query_count / energy_mj

    def get_energy_per_query_pj(self) -> float:
        """Get energy per query in pJ"""
        if self._query_count == 0:
            return 0
        return self.get_total_energy_pj() / self._query_count

    def get_energy_per_query_nj(self) -> float:
        """Get energy per query in nJ"""
        return self.get_energy_per_query_pj() / 1000.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get all energy metrics"""
        return {
            'total_energy_pj': self.get_total_energy_pj(),
            'total_energy_nj': self._breakdown.total_energy_nj,
            'total_energy_uj': self._breakdown.total_energy_uj,
            'total_energy_mj': self.get_total_energy_mj(),
            'query_count': self._query_count,
            'energy_per_query_pj': self.get_energy_per_query_pj(),
            'energy_per_query_nj': self.get_energy_per_query_nj(),
            'queries_per_mj': self.get_queries_per_mj(),
            'active_cycles': self._active_cycles,
            'breakdown': self._breakdown.get_breakdown_dict(),
            'breakdown_percent': self._breakdown.get_percentage_breakdown(),
        }

    def print_summary(self) -> None:
        """Print energy summary"""
        print("\n" + "="*60)
        print("Energy Summary")
        print("="*60)

        breakdown = self._breakdown.get_breakdown_dict()
        percentages = self._breakdown.get_percentage_breakdown()

        print(f"\n  Total Energy: {self.get_total_energy_pj():.2f} pJ "
              f"({self._breakdown.total_energy_nj:.4f} nJ)")
        print(f"  Queries Processed: {self._query_count}")
        print(f"  Energy per Query: {self.get_energy_per_query_pj():.2f} pJ")
        print(f"  Energy Efficiency: {self.get_queries_per_mj():.0f} qry/mJ")

        print("\n  Breakdown:")
        for component, energy in breakdown.items():
            if component != 'total' and energy > 0:
                pct = percentages.get(component, 0)
                print(f"    {component:12s}: {energy:10.2f} pJ ({pct:5.1f}%)")

        print("="*60)


def estimate_attention_energy(seq_length: int, head_dim: int, k_value: int,
                              config: Optional[EnergyConfig] = None) -> EnergyBreakdown:
    """
    Estimate energy for one attention head.

    Args:
        seq_length: Sequence length N
        head_dim: Head dimension d
        k_value: Top-k sparsity
        config: Energy configuration

    Returns:
        Energy breakdown
    """
    config = config or EnergyConfig()
    breakdown = EnergyBreakdown()

    N = seq_length
    d = head_dim
    k = k_value

    # Stage 1: Association (BA-CAM)
    # Write N keys of d bits each
    cam_write_bits = N * d
    breakdown.cam_write_energy_pj = cam_write_bits * config.cam_write_energy_fj_per_bit / 1000.0

    # Search N queries, each searching N*d bits
    cam_search_bits = N * N * d
    breakdown.cam_search_energy_pj = cam_search_bits * config.cam_search_energy_fj_per_bit / 1000.0

    # ADC: N queries × N matchlines (but can be parallelized)
    adc_conversions = N * N
    breakdown.adc_energy_pj = adc_conversions * config.adc_energy_pj_per_conversion

    # Stage 2: Selection (Top-K + SoftMax)
    # Top-K: N rows, each with O(N log k) comparisons
    topk_comparisons = N * N  # Simplified
    topk_registers = N * k
    breakdown.topk_energy_pj = (topk_comparisons * config.topk_comparison_energy_pj +
                                 topk_registers * config.topk_register_energy_pj)

    # SoftMax: N rows × k elements
    softmax_ops = N * k
    breakdown.softmax_energy_pj = softmax_ops * (config.softmax_exp_energy_pj +
                                                   config.softmax_add_energy_pj +
                                                   config.softmax_div_energy_pj)

    # Stage 3: Contextualization (Sparse MAC)
    # N outputs, each requiring k × d MACs
    mac_ops = N * k * d
    breakdown.mac_energy_pj = mac_ops * config.mac_energy_pj

    # SRAM: Read N × d values, but only k accessed per query
    sram_words = N * k * d // 16  # BF16 = 16 bits
    breakdown.sram_energy_pj = sram_words * config.sram_read_energy_pj_per_word

    return breakdown
