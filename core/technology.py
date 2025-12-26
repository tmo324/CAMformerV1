"""
Technology Node Configuration

Centralized technology parameters for CAMformer simulation.
All components should reference these parameters for consistency.

Based on CAMformer paper (28nm CMOS technology).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TechnologyNode:
    """
    Technology node parameters.

    Frozen to prevent accidental modification.
    """
    name: str
    node_nm: int
    vdd_v: float

    # 10T1C BA-CAM Cell
    cam_cell_area_um2: float
    cam_search_energy_fj_per_bit: float
    cam_write_energy_fj_per_bit: float
    cam_leakage_uw_per_cell: float

    # 6T SRAM Cell
    sram_cell_area_um2: float
    sram_read_energy_pj_per_word: float
    sram_write_energy_pj_per_word: float
    sram_leakage_uw_per_kb: float

    # ADC (6-bit SAR)
    adc_area_per_channel_um2: float
    adc_energy_pj_per_conversion: float

    # BF16 MAC Unit
    mac_area_um2: float
    mac_energy_pj: float
    mac_register_energy_pj: float

    # Comparator (for TopK sorter)
    comparator_area_um2: float
    comparison_energy_pj: float
    register_energy_pj: float

    # SoftMax LUT
    softmax_lut_area_um2: float
    softmax_exp_energy_pj: float
    softmax_add_energy_pj: float
    softmax_div_energy_pj: float
    softmax_lut_energy_pj: float

    # Buffer/Register
    buffer_area_per_bit_um2: float
    buffer_read_energy_pj: float
    buffer_write_energy_pj: float

    # DRAM (off-chip)
    dram_read_energy_pj_per_byte: float
    dram_write_energy_pj_per_byte: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'node_nm': self.node_nm,
            'vdd_v': self.vdd_v,
            'cam': {
                'cell_area_um2': self.cam_cell_area_um2,
                'search_energy_fj_per_bit': self.cam_search_energy_fj_per_bit,
                'write_energy_fj_per_bit': self.cam_write_energy_fj_per_bit,
                'leakage_uw_per_cell': self.cam_leakage_uw_per_cell,
            },
            'sram': {
                'cell_area_um2': self.sram_cell_area_um2,
                'read_energy_pj_per_word': self.sram_read_energy_pj_per_word,
                'write_energy_pj_per_word': self.sram_write_energy_pj_per_word,
                'leakage_uw_per_kb': self.sram_leakage_uw_per_kb,
            },
            'adc': {
                'area_per_channel_um2': self.adc_area_per_channel_um2,
                'energy_pj_per_conversion': self.adc_energy_pj_per_conversion,
            },
            'mac': {
                'area_um2': self.mac_area_um2,
                'energy_pj': self.mac_energy_pj,
                'register_energy_pj': self.mac_register_energy_pj,
            },
            'comparator': {
                'area_um2': self.comparator_area_um2,
                'comparison_energy_pj': self.comparison_energy_pj,
                'register_energy_pj': self.register_energy_pj,
            },
            'softmax': {
                'lut_area_um2': self.softmax_lut_area_um2,
                'exp_energy_pj': self.softmax_exp_energy_pj,
                'add_energy_pj': self.softmax_add_energy_pj,
                'div_energy_pj': self.softmax_div_energy_pj,
                'lut_energy_pj': self.softmax_lut_energy_pj,
            },
            'buffer': {
                'area_per_bit_um2': self.buffer_area_per_bit_um2,
                'read_energy_pj': self.buffer_read_energy_pj,
                'write_energy_pj': self.buffer_write_energy_pj,
            },
            'dram': {
                'read_energy_pj_per_byte': self.dram_read_energy_pj_per_byte,
                'write_energy_pj_per_byte': self.dram_write_energy_pj_per_byte,
            },
        }


# =============================================================================
# Predefined Technology Nodes
# =============================================================================

# 28nm CMOS - CAMformer paper baseline
TECH_28NM = TechnologyNode(
    name="28nm CMOS",
    node_nm=28,
    vdd_v=0.9,

    # 10T1C BA-CAM Cell (from CAMformer paper)
    cam_cell_area_um2=0.325,
    cam_search_energy_fj_per_bit=0.03,
    cam_write_energy_fj_per_bit=0.08,
    cam_leakage_uw_per_cell=0.001,

    # 6T SRAM Cell
    sram_cell_area_um2=0.127,
    sram_read_energy_pj_per_word=0.5,
    sram_write_energy_pj_per_word=0.6,
    sram_leakage_uw_per_kb=10.0,

    # ADC (6-bit SAR)
    adc_area_per_channel_um2=100.0,
    adc_energy_pj_per_conversion=0.5,

    # BF16 MAC Unit
    mac_area_um2=200.0,
    mac_energy_pj=0.4,
    mac_register_energy_pj=0.05,

    # Comparator (for TopK sorter)
    comparator_area_um2=50.0,
    comparison_energy_pj=0.05,
    register_energy_pj=0.02,

    # SoftMax LUT
    softmax_lut_area_um2=1000.0,
    softmax_exp_energy_pj=0.5,
    softmax_add_energy_pj=0.1,
    softmax_div_energy_pj=1.0,
    softmax_lut_energy_pj=0.2,

    # Buffer/Register
    buffer_area_per_bit_um2=0.5,
    buffer_read_energy_pj=0.1,
    buffer_write_energy_pj=0.15,

    # DRAM (off-chip)
    dram_read_energy_pj_per_byte=20.0,
    dram_write_energy_pj_per_byte=25.0,
)


# 16nm FinFET - scaled from 28nm
TECH_16NM = TechnologyNode(
    name="16nm FinFET",
    node_nm=16,
    vdd_v=0.8,

    # ~0.5x area, ~0.6x energy vs 28nm
    cam_cell_area_um2=0.163,
    cam_search_energy_fj_per_bit=0.018,
    cam_write_energy_fj_per_bit=0.048,
    cam_leakage_uw_per_cell=0.0008,

    sram_cell_area_um2=0.064,
    sram_read_energy_pj_per_word=0.3,
    sram_write_energy_pj_per_word=0.36,
    sram_leakage_uw_per_kb=8.0,

    adc_area_per_channel_um2=50.0,
    adc_energy_pj_per_conversion=0.3,

    mac_area_um2=100.0,
    mac_energy_pj=0.24,
    mac_register_energy_pj=0.03,

    comparator_area_um2=25.0,
    comparison_energy_pj=0.03,
    register_energy_pj=0.012,

    softmax_lut_area_um2=500.0,
    softmax_exp_energy_pj=0.3,
    softmax_add_energy_pj=0.06,
    softmax_div_energy_pj=0.6,
    softmax_lut_energy_pj=0.12,

    buffer_area_per_bit_um2=0.25,
    buffer_read_energy_pj=0.06,
    buffer_write_energy_pj=0.09,

    dram_read_energy_pj_per_byte=15.0,
    dram_write_energy_pj_per_byte=18.0,
)


# 45nm CMOS - Paper baseline (from Ben's scaled modules.csv)
# This is the technology node used in the CAMformer paper
TECH_45NM = TechnologyNode(
    name="45nm CMOS (Paper)",
    node_nm=45,
    vdd_v=1.0,

    # 10T1C BA-CAM Cell (scaled from 65nm: area × 0.66, energy × 0.52)
    cam_cell_area_um2=0.215,          # Approximate
    cam_search_energy_fj_per_bit=0.016,
    cam_write_energy_fj_per_bit=0.042,
    cam_leakage_uw_per_cell=0.0008,

    # SRAM (from 40nm CACTI, close to 45nm)
    sram_cell_area_um2=0.100,
    sram_read_energy_pj_per_word=0.42,
    sram_write_energy_pj_per_word=0.50,
    sram_leakage_uw_per_kb=8.0,

    # ADC (6-bit SAR, from 40nm)
    adc_area_per_channel_um2=4000.0,
    adc_energy_pj_per_conversion=0.95,  # Per ADC, × 16 for total

    # BF16 MAC Unit (scaled from 65nm)
    mac_area_um2=2879.58,
    mac_energy_pj=1.12,                 # Per MAC
    mac_register_energy_pj=0.04,

    # Comparator (for TopK sorter)
    comparator_area_um2=33.0,
    comparison_energy_pj=0.03,
    register_energy_pj=0.015,

    # SoftMax LUT (from 40nm)
    softmax_lut_area_um2=1612.0,
    softmax_exp_energy_pj=0.4,
    softmax_add_energy_pj=0.08,
    softmax_div_energy_pj=0.07,
    softmax_lut_energy_pj=0.15,

    # Buffer/Register
    buffer_area_per_bit_um2=0.35,
    buffer_read_energy_pj=0.08,
    buffer_write_energy_pj=0.12,

    # DRAM (off-chip, from paper)
    dram_read_energy_pj_per_byte=18.0,
    dram_write_energy_pj_per_byte=22.0,
)


# 7nm FinFET - further scaled
TECH_7NM = TechnologyNode(
    name="7nm FinFET",
    node_nm=7,
    vdd_v=0.7,

    # ~0.2x area, ~0.35x energy vs 28nm
    cam_cell_area_um2=0.065,
    cam_search_energy_fj_per_bit=0.01,
    cam_write_energy_fj_per_bit=0.028,
    cam_leakage_uw_per_cell=0.0006,

    sram_cell_area_um2=0.025,
    sram_read_energy_pj_per_word=0.18,
    sram_write_energy_pj_per_word=0.21,
    sram_leakage_uw_per_kb=6.0,

    adc_area_per_channel_um2=20.0,
    adc_energy_pj_per_conversion=0.18,

    mac_area_um2=40.0,
    mac_energy_pj=0.14,
    mac_register_energy_pj=0.018,

    comparator_area_um2=10.0,
    comparison_energy_pj=0.018,
    register_energy_pj=0.007,

    softmax_lut_area_um2=200.0,
    softmax_exp_energy_pj=0.18,
    softmax_add_energy_pj=0.035,
    softmax_div_energy_pj=0.35,
    softmax_lut_energy_pj=0.07,

    buffer_area_per_bit_um2=0.1,
    buffer_read_energy_pj=0.035,
    buffer_write_energy_pj=0.053,

    dram_read_energy_pj_per_byte=10.0,
    dram_write_energy_pj_per_byte=12.0,
)


# Default technology node
DEFAULT_TECH = TECH_28NM


# =============================================================================
# Helper Functions
# =============================================================================

def get_technology(node_nm: int) -> TechnologyNode:
    """
    Get technology node by feature size.

    Args:
        node_nm: Feature size in nanometers (45, 28, 16, or 7)

    Returns:
        TechnologyNode for the specified node
    """
    nodes = {
        45: TECH_45NM,
        28: TECH_28NM,
        16: TECH_16NM,
        7: TECH_7NM,
    }

    if node_nm not in nodes:
        raise ValueError(f"Unknown technology node: {node_nm}nm. "
                        f"Available: {list(nodes.keys())}")

    return nodes[node_nm]


def print_technology_summary(tech: TechnologyNode) -> None:
    """Print technology summary"""
    print(f"\n{'='*60}")
    print(f"Technology: {tech.name}")
    print(f"{'='*60}")
    print(f"  Node: {tech.node_nm}nm")
    print(f"  VDD: {tech.vdd_v}V")

    print(f"\n  BA-CAM (10T1C):")
    print(f"    Cell area: {tech.cam_cell_area_um2:.3f} μm²")
    print(f"    Search energy: {tech.cam_search_energy_fj_per_bit:.3f} fJ/bit")
    print(f"    Write energy: {tech.cam_write_energy_fj_per_bit:.3f} fJ/bit")

    print(f"\n  SRAM (6T):")
    print(f"    Cell area: {tech.sram_cell_area_um2:.3f} μm²")
    print(f"    Read energy: {tech.sram_read_energy_pj_per_word:.2f} pJ/word")
    print(f"    Write energy: {tech.sram_write_energy_pj_per_word:.2f} pJ/word")

    print(f"\n  MAC (BF16):")
    print(f"    Area: {tech.mac_area_um2:.0f} μm²")
    print(f"    Energy: {tech.mac_energy_pj:.2f} pJ/op")

    print(f"\n  ADC (6-bit SAR):")
    print(f"    Area: {tech.adc_area_per_channel_um2:.0f} μm²/ch")
    print(f"    Energy: {tech.adc_energy_pj_per_conversion:.2f} pJ/conv")

    print(f"{'='*60}")
