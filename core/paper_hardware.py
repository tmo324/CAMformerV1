"""
CAMformer Paper Hardware Parameters

Hardware module parameters matching Ben's reference implementation
for reproducing paper results (arXiv:2511.19740v1).

All values are scaled to 45nm technology node.
Energy model: E = P × cycles (mW × cycles = nJ at 1 GHz)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class PipelineMode(Enum):
    """Pipeline execution modes matching Ben's implementation"""
    NO_PIPELINE = "no_pipeline"      # No fine-grained pipelining
    FULL_PIPELINE = "full_pipeline"  # All stages pipelined
    REALISTIC = "realistic"          # Association/Contextualization pipelined, Normalization not


@dataclass(frozen=True)
class HardwareModule:
    """
    Single hardware module specification.

    Attributes:
        name: Module name
        power_mw: Active power in milliwatts
        area_um2: Area in square micrometers
        delay_cycles: Latency in cycles
        count: Number of instances (for parallel units like ADCs, MACs)
        power2_mw: Secondary power (for alternate modes like CAM programming)
        delay2_cycles: Secondary delay
    """
    name: str
    power_mw: float
    area_um2: float
    delay_cycles: int
    count: int = 1
    power2_mw: Optional[float] = None
    delay2_cycles: Optional[int] = None

    @property
    def total_power_mw(self) -> float:
        """Total power including all instances"""
        return self.power_mw * self.count

    @property
    def total_area_um2(self) -> float:
        """Total area including all instances"""
        return self.area_um2 * self.count

    def energy_mw_cycles(self, times: int = 1, use_secondary: bool = False) -> float:
        """
        Calculate energy in mW·cycles (same as pJ at 1 GHz).

        E = P (mW) × count × delay (cycles) × times
        Note: Includes count multiplier for parallel units (ADCs, MACs, etc.)
        """
        if use_secondary and self.power2_mw is not None:
            return self.power2_mw * self.count * self.delay2_cycles * times
        return self.power_mw * self.count * self.delay_cycles * times

    def energy_nj(self, times: int = 1, use_secondary: bool = False) -> float:
        """
        Calculate energy in nanojoules.

        At 1 GHz: mW·cycles = mW·ns = pJ
        Convert pJ to nJ by dividing by 1000.
        """
        return self.energy_mw_cycles(times, use_secondary) / 1000.0


# =============================================================================
# Paper Hardware Modules (45nm technology node)
# =============================================================================
# Scaled from Ben's modules.csv using:
#   - 65nm → 45nm: power × 0.52 (energy ratio), area × 0.66
#   - 40nm: no scaling needed (close to 45nm)
#   - Values match Ben's Simulate.ipynb output exactly

PAPER_MODULES = {
    # Association Stage
    "Query Buffer": HardwareModule(
        name="Query Buffer",
        power_mw=0.8034782608695652,  # 1.54 * 0.52
        area_um2=2475.0,              # 3750 * 0.66
        delay_cycles=1,
        count=1,
    ),
    "Key SRAM": HardwareModule(
        name="Key SRAM",
        power_mw=28.4,                # Already at 40nm
        area_um2=27728.6,
        delay_cycles=1,
        count=1,
    ),
    "BA-CAM 16X64": HardwareModule(
        name="BA-CAM 16X64",
        power_mw=13.14782608695652,   # (100.8 * 0.52) / 4
        area_um2=2230.272,            # (13516.8 * 0.66) / 4
        delay_cycles=1,
        count=1,
        power2_mw=13.14782608695652,  # Programming = Search
        delay2_cycles=1,
    ),
    "6b ADCs": HardwareModule(
        name="6b ADCs",
        power_mw=0.95,                # Per-ADC power (40nm)
        area_um2=4000.0,              # Per-ADC area
        delay_cycles=2,
        count=16,                     # 16 ADC channels
    ),
    "Fixed Scalers": HardwareModule(
        name="Fixed Scalers",
        power_mw=0.0245217391304348,  # 0.047 * 0.52 per scaler
        area_um2=83.6352,             # 126.72 * 0.66 per scaler
        delay_cycles=2,
        count=16,
    ),
    "Top2": HardwareModule(
        name="Top2",
        power_mw=6.083673913043478,   # 11.66 * 0.52
        area_um2=8801.3376924,        # 13335.36 * 0.66
        delay_cycles=2,
        count=1,
    ),
    "Ptop Reg": HardwareModule(
        name="Ptop Reg",
        power_mw=1.528695652173913,   # 2.93 * 0.52
        area_um2=6042.3,              # 9155 * 0.66
        delay_cycles=1,
        count=1,
    ),

    # Normalization Stage
    "Top32": HardwareModule(
        name="Top32",
        power_mw=52.077147824347826,  # 99.81 * 0.52
        area_um2=67016.929254,        # 101540.8 * 0.66
        delay_cycles=6,
        count=1,
    ),
    "SoftMax LUT": HardwareModule(
        name="SoftMax LUT",
        power_mw=1.418,               # 40nm
        area_um2=1612.373532,
        delay_cycles=1,
        count=1,
    ),
    "SoftMax Div.": HardwareModule(
        name="SoftMax Div.",
        power_mw=0.07109,             # 45nm
        area_um2=441.529,
        delay_cycles=3,
        count=1,
    ),
    "SoftMax Acc.": HardwareModule(
        name="SoftMax Acc.",
        power_mw=1.1170434782608696,  # 2.141 * 0.52
        area_um2=2879.58,             # 4363 * 0.66
        delay_cycles=1,
        count=1,
    ),
    "Output Buff": HardwareModule(
        name="Output Buff",
        power_mw=1.5547826086956522,  # 2.98 * 0.52
        area_um2=5990.82,             # 9077 * 0.66
        delay_cycles=1,
        count=1,
    ),

    # Contextualization Stage
    "Value SRAM": HardwareModule(
        name="Value SRAM",
        power_mw=34.7,                # 40nm
        area_um2=44820.0,
        delay_cycles=1,
        count=1,
    ),
    "BF16 MACs": HardwareModule(
        name="BF16 MACs",
        power_mw=1.1170434782608696,  # 2.141 * 0.52 per MAC
        area_um2=2879.58,             # 4363 * 0.66 per MAC
        delay_cycles=4,
        count=8,                      # 8 parallel MACs
    ),

    # Off-chip Memory
    "DRAM": HardwareModule(
        name="DRAM",
        power_mw=12.42666667,         # 45nm, energy/cycles
        area_um2=0.0,                 # Off-chip
        delay_cycles=24,
        count=1,
    ),
}


@dataclass
class PaperConfig:
    """
    Configuration matching paper parameters.

    Default values produce the paper's reported metrics:
    - Energy efficiency: 9045 qry/mJ
    - Throughput: 191 qry/ms (single-core)
    - Area: 0.258 mm²
    """
    # Sequence and head parameters
    seq_length: int = 1024
    head_dim: int = 64
    num_heads: int = 16
    k_value: int = 32

    # Architecture parameters
    num_tiles: int = 64            # 1024 / 16 = 64 tiles
    cam_rows: int = 16             # Rows per CAM tile
    cam_cols: int = 64             # Columns per CAM tile
    rows_per_program: int = 4      # Program 4 rows at a time

    # Parallelism
    n_par: int = 1                 # Softmax parallelism level
    c_par: int = 8                 # MAC parallelism (8 parallel MACs)

    # Clock (1 GHz assumed for energy calculation)
    clock_freq_hz: float = 1e9

    # Pipeline mode
    pipeline_mode: PipelineMode = PipelineMode.REALISTIC


class PaperHardwareModel:
    """
    Hardware model matching Ben's Camformer.py implementation.

    Provides cycle-accurate timing and energy estimation
    for reproducing paper results.
    """

    def __init__(self, config: Optional[PaperConfig] = None,
                 modules: Optional[Dict[str, HardwareModule]] = None):
        self.config = config or PaperConfig()
        self.modules = modules or PAPER_MODULES.copy()

        # Accumulated metrics per module
        self._module_cycles: Dict[str, int] = {name: 0 for name in self.modules}
        self._module_energy: Dict[str, float] = {name: 0.0 for name in self.modules}

        # Stage cycle tracking
        self.association_cycles = 0
        self.normalization_cycles = 0
        self.contextualization_cycles = 0
        self.total_cycles = 0

    def reset(self) -> None:
        """Reset all accumulated metrics"""
        self._module_cycles = {name: 0 for name in self.modules}
        self._module_energy = {name: 0.0 for name in self.modules}
        self.association_cycles = 0
        self.normalization_cycles = 0
        self.contextualization_cycles = 0
        self.total_cycles = 0

    def _run_module(self, name: str, times: int = 1,
                    use_secondary: bool = False) -> float:
        """
        Run a module and accumulate energy.

        Returns energy in nJ.
        """
        module = self.modules[name]
        energy = module.energy_nj(times, use_secondary)
        self._module_energy[name] += energy
        return energy

    def _add_cycles(self, name: str, cycles: int) -> None:
        """Add cycles for a module"""
        self._module_cycles[name] += cycles
        self.total_cycles += cycles

    def run_association(self, pipelined: bool = True) -> int:
        """
        Run association phase.

        Returns cycles for this phase.
        """
        starting_cycles = self.total_cycles
        cfg = self.config

        # Load query buffer
        self._run_module("Query Buffer", 1)

        if not pipelined:
            # Non-pipelined: add all delays sequentially
            for tile in range(cfg.num_tiles):
                # Program CAM 4 rows at a time
                for row_batch in range(cfg.cam_rows // cfg.rows_per_program):
                    self._run_module("Key SRAM", 1)
                    self._run_module("BA-CAM 16X64", 1, use_secondary=True)
                    self._add_cycles("BA-CAM 16X64",
                                    self.modules["BA-CAM 16X64"].delay2_cycles)

                # Search
                self._run_module("Query Buffer", 1)
                self._run_module("BA-CAM 16X64", 1)
                self._add_cycles("BA-CAM 16X64",
                                self.modules["BA-CAM 16X64"].delay_cycles)

                # ADC conversion
                self._run_module("6b ADCs", 1)
                self._add_cycles("6b ADCs", self.modules["6b ADCs"].delay_cycles)

                # Scaling
                self._run_module("Fixed Scalers", 1)
                self._add_cycles("Fixed Scalers",
                                self.modules["Fixed Scalers"].delay_cycles)

                # Top-4 selection
                self._run_module("Top2", 1)
                self._add_cycles("Top2", self.modules["Top2"].delay_cycles)

                # Potential top register
                self._run_module("Ptop Reg", 1)
                self._add_cycles("Ptop Reg", self.modules["Ptop Reg"].delay_cycles)

                # Top-32 every 16 tiles
                if tile >= 32 and tile % 16 == 0:
                    self._run_module("Top32", 1)
                    self._add_cycles("Top32", self.modules["Top32"].delay_cycles)
                    self._run_module("Ptop Reg", 1)
                    self._add_cycles("Ptop Reg", self.modules["Ptop Reg"].delay_cycles)

                # DRAM prefetch (energy only, no cycle count in association)
                self._run_module("DRAM", 1)
                self._run_module("Value SRAM", 1)
        else:
            # Pipelined: only CAM program/search adds cycles
            for tile in range(cfg.num_tiles):
                # Program CAM 4 rows at a time
                for row_batch in range(cfg.cam_rows // cfg.rows_per_program):
                    self._run_module("Key SRAM", 1)
                    self._run_module("BA-CAM 16X64", 1, use_secondary=True)
                    self._add_cycles("BA-CAM 16X64",
                                    self.modules["BA-CAM 16X64"].delay2_cycles)

                # Search
                self._run_module("Query Buffer", 1)
                self._run_module("BA-CAM 16X64", 1)
                self._add_cycles("BA-CAM 16X64",
                                self.modules["BA-CAM 16X64"].delay_cycles)

                # Pipelined operations (energy only)
                self._run_module("6b ADCs", 1)
                self._run_module("Fixed Scalers", 1)
                self._run_module("Top2", 1)
                self._run_module("Ptop Reg", 1)

                # Top-32 every 16 tiles
                if tile >= 32 and tile % 16 == 0:
                    self._run_module("Top32", 1)
                    self._run_module("Ptop Reg", 1)

                # DRAM prefetch
                self._run_module("DRAM", 1)
                self._run_module("Value SRAM", 1)

            # Pipeline drain: add delays for final operations
            self._add_cycles("6b ADCs", self.modules["6b ADCs"].delay_cycles)
            self._add_cycles("Fixed Scalers",
                            self.modules["Fixed Scalers"].delay_cycles)
            self._add_cycles("Top2", self.modules["Top2"].delay_cycles)
            self._add_cycles("Ptop Reg", self.modules["Ptop Reg"].delay_cycles)

        self.association_cycles = self.total_cycles - starting_cycles
        return self.association_cycles

    def run_normalization(self, pipelined: bool = False) -> int:
        """
        Run normalization phase (softmax).

        Returns cycles for this phase.
        """
        starting_cycles = self.total_cycles
        cfg = self.config

        # Final Top-32 selection
        self._run_module("Top32", 1)
        self._add_cycles("Top32", self.modules["Top32"].delay_cycles)

        # Softmax loop
        for i in range(cfg.k_value // cfg.n_par):
            self._run_module("SoftMax LUT", 1)
            self._run_module("SoftMax Acc.", 1)
            self._run_module("Output Buff", 1)
            self._add_cycles("SoftMax Acc.",
                            self.modules["SoftMax Acc."].delay_cycles)

        # Division for normalization
        for i in range(cfg.k_value // cfg.n_par):
            self._run_module("Output Buff", 1)  # Read
            self._run_module("SoftMax Div.", 1)
            self._run_module("Output Buff", 1)  # Write back
            if not pipelined:
                self._add_cycles("SoftMax Div.",
                                self.modules["SoftMax Div."].delay_cycles)
            else:
                self._add_cycles("SoftMax Div.", 1)

        if pipelined:
            self._add_cycles("SoftMax Div.",
                            self.modules["SoftMax Div."].delay_cycles)

        self.normalization_cycles = self.total_cycles - starting_cycles
        return self.normalization_cycles

    def run_contextualization(self, pipelined: bool = True) -> int:
        """
        Run contextualization phase (sparse matmul).

        Returns cycles for this phase.
        """
        starting_cycles = self.total_cycles
        cfg = self.config

        # MACs: 64 × 32 operations / parallelism
        num_mac_iterations = (cfg.head_dim * cfg.k_value) // cfg.c_par

        for i in range(num_mac_iterations):
            self._run_module("Value SRAM", 1)
            self._run_module("BF16 MACs", 1)
            if pipelined:
                self._add_cycles("BF16 MACs", 1)
            else:
                self._add_cycles("BF16 MACs",
                                self.modules["BF16 MACs"].delay_cycles)

        if pipelined:
            # Pipeline drain
            self._add_cycles("BF16 MACs",
                            self.modules["BF16 MACs"].delay_cycles)

        self.contextualization_cycles = self.total_cycles - starting_cycles
        return self.contextualization_cycles

    def run_attention(self, mode: Optional[PipelineMode] = None) -> Dict[str, Any]:
        """
        Run complete attention computation.

        Args:
            mode: Pipeline mode (defaults to config)

        Returns:
            Dictionary with all metrics
        """
        mode = mode or self.config.pipeline_mode
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

    def get_onchip_energy_nj(self) -> float:
        """Get on-chip energy (excluding DRAM)"""
        return sum(e for name, e in self._module_energy.items()
                   if name != "DRAM")

    def get_total_energy_nj(self) -> float:
        """Get total energy including DRAM"""
        return sum(self._module_energy.values())

    def get_total_area_um2(self) -> float:
        """Get total on-chip area"""
        return sum(m.total_area_um2 for name, m in self.modules.items()
                   if name != "DRAM")

    def get_total_area_mm2(self) -> float:
        """Get total on-chip area in mm²"""
        return self.get_total_area_um2() / 1e6

    def get_max_stage_cycles(self) -> int:
        """Get cycles of longest stage (for coarse-grained pipelining)"""
        return max(self.association_cycles,
                   self.normalization_cycles,
                   self.contextualization_cycles)

    def get_power_mw(self, use_max_stage: bool = True) -> float:
        """
        Get average power in mW.

        At 1 GHz: P = E / t, where E is in nJ and t is in ns (cycles).
        nJ / ns = W, so multiply by 1000 to get mW.

        Args:
            use_max_stage: If True, use max stage cycles (CG pipelined)
        """
        cycles = self.get_max_stage_cycles() if use_max_stage else self.total_cycles
        if cycles == 0:
            return 0.0
        # Energy (nJ) / Time (ns) = Power (W), then × 1000 for mW
        return (self.get_onchip_energy_nj() / cycles) * 1000

    def get_throughput_per_ms(self, use_max_stage: bool = True) -> float:
        """
        Get throughput in attentions per millisecond.

        Args:
            use_max_stage: If True, use max stage cycles (CG pipelined)
        """
        cycles = self.get_max_stage_cycles() if use_max_stage else self.total_cycles
        if cycles == 0:
            return 0.0
        # At 1 GHz: cycles = ns, throughput = 1e6 / cycles (per ms)
        return 1e6 / cycles

    def get_energy_efficiency(self) -> float:
        """Get energy efficiency in attentions per mJ"""
        energy_mj = self.get_total_energy_nj() / 1e6
        if energy_mj == 0:
            return 0.0
        return 1.0 / energy_mj

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary"""
        max_stage = self.get_max_stage_cycles()
        # DRAM power: energy (nJ) / time (ns) * 1000 = mW
        dram_energy_nj = self._module_energy.get("DRAM", 0)
        dram_power = (dram_energy_nj / max_stage * 1000) if max_stage > 0 else 0
        onchip_power = self.get_power_mw(use_max_stage=True)
        total_power = onchip_power + dram_power

        return {
            "cycles": {
                "total": self.total_cycles,
                "association": self.association_cycles,
                "normalization": self.normalization_cycles,
                "contextualization": self.contextualization_cycles,
                "max_stage": max_stage,
            },
            "energy_nj": {
                "onchip": self.get_onchip_energy_nj(),
                "dram": self._module_energy.get("DRAM", 0),
                "total": self.get_total_energy_nj(),
            },
            "power_mw": {
                "onchip": onchip_power,
                "dram": dram_power,
                "total": total_power,
            },
            "area_mm2": self.get_total_area_mm2(),
            "throughput": {
                "single_core_per_ms": self.get_throughput_per_ms() / self.config.num_heads,
                "multi_core_per_ms": self.get_throughput_per_ms(),
            },
            "efficiency": {
                "queries_per_mj": self.get_energy_efficiency() * 1000,  # Convert to qry/mJ
            },
            "energy_breakdown": dict(self._module_energy),
        }

    def print_summary(self) -> None:
        """Print formatted summary matching Ben's output"""
        metrics = self.get_metrics()

        print(f"\n{'='*60}")
        print("CAMformer Hardware Model Results")
        print(f"{'='*60}")

        print(f"\nCycles:")
        print(f"  Association:      {metrics['cycles']['association']:>6} cycles")
        print(f"  Normalization:    {metrics['cycles']['normalization']:>6} cycles")
        print(f"  Contextualization:{metrics['cycles']['contextualization']:>6} cycles")
        print(f"  Total:            {metrics['cycles']['total']:>6} cycles")
        print(f"  Max Stage:        {metrics['cycles']['max_stage']:>6} cycles")

        print(f"\nEnergy:")
        print(f"  On-chip:  {metrics['energy_nj']['onchip']:.2f} nJ")
        print(f"  DRAM:     {metrics['energy_nj']['dram']:.2f} nJ")
        print(f"  Total:    {metrics['energy_nj']['total']:.2f} nJ")

        print(f"\nPower (CG pipelined):")
        print(f"  On-chip:  {metrics['power_mw']['onchip']:.2f} mW")
        print(f"  DRAM:     {metrics['power_mw']['dram']:.2f} mW")
        print(f"  Total:    {metrics['power_mw']['total']:.2f} mW")

        print(f"\nArea: {metrics['area_mm2']:.6f} mm²")

        print(f"\nThroughput:")
        print(f"  Single-core MHA: {metrics['throughput']['single_core_per_ms']:.2f} att/ms")
        print(f"  Multi-core MHA:  {metrics['throughput']['multi_core_per_ms']:.2f} att/ms")

        print(f"\nEnergy Efficiency:")
        print(f"  {metrics['efficiency']['queries_per_mj']:.0f} qry/mJ")

        print(f"\nEnergy Breakdown by Module:")
        for name, energy in sorted(metrics['energy_breakdown'].items(),
                                   key=lambda x: -x[1]):
            print(f"  {name:20s}: {energy:10.2f} nJ")

        print(f"{'='*60}")


def validate_against_paper() -> bool:
    """
    Validate model produces paper's expected results.

    Expected (REALISTIC mode):
    - Total cycles: 721
    - Total energy: ~54.9 nJ
    - Throughput: ~191 qry/ms (single-core)
    - Area: ~0.258 mm²
    """
    model = PaperHardwareModel()
    metrics = model.run_attention(PipelineMode.REALISTIC)

    print("Validating against paper results...")

    # Expected values from Ben's notebook
    expected = {
        "cycles": 721,
        "energy_nj": 54.92,
        "throughput_per_ms": 191.13,
        "area_mm2": 0.258414,
    }

    # Check each metric with tolerance
    tolerance = 0.05  # 5%

    checks = [
        ("Cycles", metrics["cycles"]["total"], expected["cycles"]),
        ("Energy (nJ)", metrics["energy_nj"]["total"], expected["energy_nj"]),
        ("Throughput (att/ms)", metrics["throughput"]["single_core_per_ms"],
         expected["throughput_per_ms"]),
        ("Area (mm²)", metrics["area_mm2"], expected["area_mm2"]),
    ]

    all_pass = True
    for name, actual, expected_val in checks:
        pct_diff = abs(actual - expected_val) / expected_val
        status = "PASS" if pct_diff < tolerance else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: {actual:.2f} (expected {expected_val:.2f}) - {status}")

    return all_pass


if __name__ == "__main__":
    # Run validation
    print("\n" + "="*60)
    print("Paper Hardware Model Validation")
    print("="*60)

    model = PaperHardwareModel()
    model.run_attention(PipelineMode.REALISTIC)
    model.print_summary()

    print("\n")
    if validate_against_paper():
        print("All validations PASSED!")
    else:
        print("Some validations FAILED!")
