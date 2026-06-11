"""
CAMformer SST Event-Driven Pipeline

Complete event-driven CAMformer attention pipeline with proper Links.
Supports both functional simulation and paper-accurate metrics.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.event import Event, EventType
from camformer.core.link import Link
from camformer.core.clock import Clock, ClockFrequencies, TimeConverter
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic
from camformer.core.simulation import Simulation
from camformer.core.paper_hardware import PaperHardwareModel, PaperConfig, PipelineMode

from camformer.pipeline.association import AssociationConfig
from camformer.pipeline.selection import SelectionConfig
from camformer.pipeline.contextualization import ContextualizationConfig

from .association_wrapper import AssociationWrapper
from .selection_wrapper import SelectionWrapper
from .contextualization_wrapper import ContextualizationWrapper
from .energy_model import EnergyConfig, EnergyBreakdown, EnergyTracker


@dataclass
class CAMformerSSTConfig:
    """Configuration for CAMformer SST pipeline

    Default values match the paper (arXiv:2511.19740v1):
    - seq_length: 1024 (BERT-Large sequence length)
    - head_dim: 64
    - num_heads: 16 (BERT-Large attention heads)
    - k_value: 32 (top-k sparsity)
    """

    # Model parameters (paper defaults)
    seq_length: int = 1024      # Changed from 512 to match paper
    head_dim: int = 64
    num_heads: int = 16         # Changed from 8 to match paper
    k_value: int = 32

    # Clock frequency
    clock_freq_hz: float = ClockFrequencies.SYSTEM_CLOCK_HZ  # 500 MHz

    # Pipeline mode for paper-accurate metrics
    pipeline_mode: PipelineMode = PipelineMode.REALISTIC

    # Use paper hardware model for metrics (matches Ben's implementation)
    use_paper_model: bool = True

    # Link latencies (in cycles)
    assoc_to_select_latency: int = 1
    select_to_context_latency: int = 1

    # Stage configs (auto-generated if None)
    association_config: Optional[AssociationConfig] = None
    selection_config: Optional[SelectionConfig] = None
    contextualization_config: Optional[ContextualizationConfig] = None

    # Energy configuration (auto-generated if None)
    energy_config: Optional[EnergyConfig] = None

    def __post_init__(self):
        """Initialize stage configs"""
        if self.association_config is None:
            self.association_config = AssociationConfig(
                max_seq_length=self.seq_length,
                head_dim=self.head_dim,
            )

        if self.selection_config is None:
            self.selection_config = SelectionConfig(
                k_value=self.k_value,
                max_seq_length=self.seq_length,
            )

        if self.contextualization_config is None:
            self.contextualization_config = ContextualizationConfig(
                head_dim=self.head_dim,
                k_value=self.k_value,
                max_seq_length=self.seq_length,
            )

        if self.energy_config is None:
            self.energy_config = EnergyConfig(clock_freq_hz=self.clock_freq_hz)


class CAMformerSST(Component):
    """
    CAMformer SST Event-Driven Pipeline.

    Connects three stages via Links:
    Association --[Link]--> Selection --[Link]--> Contextualization

    Provides event-driven simulation with proper timing.
    """

    def __init__(self, name: str = "camformer_sst",
                 config: Optional[CAMformerSSTConfig] = None,
                 simulation: Optional[Simulation] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or CAMformerSSTConfig()
        self._sim = simulation

        # Create time converter
        self._time_converter = TimeConverter.from_frequency(self.config.clock_freq_hz)

        # Create stage wrappers with shared energy config and pipeline mode
        self._association = AssociationWrapper(
            f"{name}_assoc",
            config=self.config.association_config,
            simulation=simulation,
            energy_config=self.config.energy_config,
            pipeline_mode=self.config.pipeline_mode
        )

        self._selection = SelectionWrapper(
            f"{name}_select",
            config=self.config.selection_config,
            simulation=simulation,
            energy_config=self.config.energy_config,
            pipeline_mode=self.config.pipeline_mode
        )

        self._contextualization = ContextualizationWrapper(
            f"{name}_context",
            config=self.config.contextualization_config,
            simulation=simulation,
            energy_config=self.config.energy_config,
            pipeline_mode=self.config.pipeline_mode
        )

        # Create Links
        self._link_assoc_select = Link(
            latency_cycles=self.config.assoc_to_select_latency,
            buffer_size=16
        )

        self._link_select_context = Link(
            latency_cycles=self.config.select_to_context_latency,
            buffer_size=16
        )

        # Connect stages via links
        self._association.configure_links(
            input_link=Link(),  # Input from external
            output_link=self._link_assoc_select
        )

        self._selection.configure_links(
            input_link=self._link_assoc_select,
            output_link=self._link_select_context
        )

        self._contextualization.configure_links(
            input_link=self._link_select_context,
            output_link=Link()  # Output to external
        )

        # Register subcomponents
        self.set_subcomponent("association", self._association)
        self.set_subcomponent("selection", self._selection)
        self.set_subcomponent("contextualization", self._contextualization)

        # Add links to simulation
        self.add_link("assoc_to_select", self._link_assoc_select)
        self.add_link("select_to_context", self._link_select_context)

        # Paper hardware model for accurate metrics
        if self.config.use_paper_model:
            paper_config = PaperConfig(
                seq_length=self.config.seq_length,
                head_dim=self.config.head_dim,
                num_heads=self.config.num_heads,
                k_value=self.config.k_value,
                pipeline_mode=self.config.pipeline_mode,
            )
            self._paper_model = PaperHardwareModel(config=paper_config)
            self._paper_metrics: Optional[Dict[str, Any]] = None
        else:
            self._paper_model = None
            self._paper_metrics = None

        # State
        self._current_head = 0
        self._total_heads = 0
        self._results = []

        # Callbacks
        self._on_complete: Optional[Callable] = None

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup pipeline statistics"""
        self.add_statistic("attention_ops", CounterStatistic(
            "attention_ops", "Attention operations", "ops"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total cycles", "cycles"
        ))
        self.add_statistic("heads_processed", CounterStatistic(
            "heads_processed", "Heads processed", "heads"
        ))

    def _setup_impl(self) -> None:
        """Pipeline setup"""
        self._output.info(f"CAMformerSST initialized: "
                         f"N={self.config.seq_length}, "
                         f"d={self.config.head_dim}, "
                         f"h={self.config.num_heads}, "
                         f"k={self.config.k_value}")

    def set_simulation(self, sim: Simulation) -> None:
        """Set simulation for all stages"""
        self._sim = sim
        self._association.set_simulation(sim)
        self._selection.set_simulation(sim)
        self._contextualization.set_simulation(sim)

        # Register links with simulation
        if sim:
            sim.add_link("assoc_to_select", self._link_assoc_select)
            sim.add_link("select_to_context", self._link_select_context)

    def forward_single_head(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            callback: Optional[Callable] = None) -> None:
        """
        Forward pass for single head (event-driven).

        Args:
            Q: Query matrix (N, d_k)
            K: Key matrix (N, d_k)
            V: Value matrix (N, d_v)
            callback: Called when complete with output
        """
        self._on_complete = callback

        # Set up chain: contextualization calls back when done
        def on_context_complete(event: Event):
            outputs = event.get_data('outputs')
            self.get_statistic("attention_ops").add_data(1)
            self.get_statistic("heads_processed").add_data(1)

            if callback:
                callback(outputs, self._get_latency_stats())

        self._contextualization.set_on_complete(on_context_complete)

        # Set up selection -> contextualization chain
        def on_select_complete(event: Event):
            # Forward to contextualization (via link)
            pass  # Handled by link

        self._selection.set_on_complete(on_select_complete)

        # Set up association -> selection chain
        def on_assoc_complete(event: Event):
            # Forward to selection (via link)
            pass  # Handled by link

        self._association.set_on_complete(on_assoc_complete)

        # Load values into contextualization
        self._contextualization.load_values(V)

        # Load keys into association
        self._association.load_keys(K)

        # Process queries (triggers the chain)
        self._association.process_queries(Q)

    def forward_sync(self, Q: np.ndarray, K: np.ndarray,
                     V: np.ndarray) -> tuple:
        """
        Synchronous forward pass (blocks until complete).

        Directly chains stages without relying on event callbacks.
        For testing and simple use cases.

        Args:
            Q: Query matrix (N, d_k)
            K: Key matrix (N, d_k)
            V: Value matrix (N, d_v)

        Returns:
            Tuple of (outputs, stats)
        """
        # Stage 1: Association - load keys and process queries
        self._association.load_keys(K)
        self._association.process_queries(Q)
        similarity_matrix = self._association.get_results()

        # Stage 2: Selection - process similarity matrix
        self._selection.process_similarity(similarity_matrix)
        selection_results = self._selection.get_results()

        # Stage 3: Contextualization - load values and compute outputs
        self._contextualization.load_values(V)
        self._contextualization.process_attention(
            selection_results['attention_weights'],
            selection_results['top_indices']
        )
        outputs = self._contextualization.get_results()

        # Update statistics
        self.get_statistic("attention_ops").add_data(1)
        self.get_statistic("heads_processed").add_data(1)

        # Run paper hardware model for accurate metrics
        if self._paper_model:
            self._paper_metrics = self._paper_model.run_attention(
                self.config.pipeline_mode
            )

        stats = self._get_latency_stats()

        return outputs, stats

    def forward_multi_head(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           callback: Optional[Callable] = None) -> None:
        """
        Forward pass for multi-head attention (event-driven).

        Args:
            Q: Query tensor (num_heads, N, d_k)
            K: Key tensor (num_heads, N, d_k)
            V: Value tensor (num_heads, N, d_v)
            callback: Called when all heads complete
        """
        num_heads = Q.shape[0]
        self._total_heads = num_heads
        self._current_head = 0
        self._results = []

        def process_next_head():
            if self._current_head >= num_heads:
                # All heads done
                outputs = np.stack(self._results, axis=0)
                if callback:
                    callback(outputs, self._get_latency_stats())
                return

            h = self._current_head

            def on_head_complete(output, stats):
                self._results.append(output)
                self._current_head += 1
                # Clear state between heads
                self._association.clear()
                self._contextualization.clear()
                process_next_head()

            self.forward_single_head(Q[h], K[h], V[h], callback=on_head_complete)

        process_next_head()

    def _get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics from all stages"""
        return {
            'association': {
                'latency': self._association.get_statistic("total_latency").get_value(),
                'events': self._association.get_statistic("events_received").get_value(),
            },
            'selection': {
                'latency': self._selection.get_statistic("total_latency").get_value(),
                'rows': self._selection.get_statistic("rows_processed").get_value(),
            },
            'contextualization': {
                'latency': self._contextualization.get_statistic("total_latency").get_value(),
                'outputs': self._contextualization.get_statistic("outputs_computed").get_value(),
            },
        }

    def get_total_latency_cycles(self) -> int:
        """Get total latency in cycles"""
        return (
            self._association.get_statistic("total_latency").get_value() +
            self._selection.get_statistic("total_latency").get_value() +
            self._contextualization.get_statistic("total_latency").get_value()
        )

    def get_total_latency_ns(self) -> float:
        """Get total latency in nanoseconds"""
        cycles = self.get_total_latency_cycles()
        return self._time_converter.cycles_to_ns(cycles)

    def get_throughput_qps(self, num_queries: int) -> float:
        """Get throughput in queries per second"""
        latency_ns = self.get_total_latency_ns()
        if latency_ns == 0:
            return 0
        latency_s = latency_ns * 1e-9
        return num_queries / latency_s

    def get_area_mm2(self) -> float:
        """Get total area"""
        return (
            self._association.get_area_mm2() +
            self._selection.get_area_mm2() +
            self._contextualization.get_area_mm2()
        )

    def get_total_energy_breakdown(self) -> EnergyBreakdown:
        """Get aggregated energy breakdown from all stages"""
        assoc_breakdown = self._association.get_energy_tracker().get_breakdown()
        select_breakdown = self._selection.get_energy_tracker().get_breakdown()
        context_breakdown = self._contextualization.get_energy_tracker().get_breakdown()

        return assoc_breakdown + select_breakdown + context_breakdown

    def get_total_energy_pj(self) -> float:
        """Get total energy in pJ"""
        return self.get_total_energy_breakdown().total_energy_pj

    def get_total_energy_nj(self) -> float:
        """Get total energy in nJ"""
        return self.get_total_energy_pj() / 1000.0

    def get_queries_per_mj(self, num_queries: int = 1) -> float:
        """Get energy efficiency in queries per mJ"""
        total_energy_mj = self.get_total_energy_pj() / 1e9
        if total_energy_mj == 0:
            return 0
        return num_queries / total_energy_mj

    def get_energy_per_query_pj(self) -> float:
        """Get energy per query in pJ"""
        return self.get_total_energy_pj()

    def reset_energy(self) -> None:
        """Reset energy tracking in all stages"""
        self._association.reset_energy()
        self._selection.reset_energy()
        self._contextualization.reset_energy()

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics from native SST simulation.

        Returns metrics from the Power × Cycles model implemented in wrappers.
        """
        # Get stage metrics from wrappers
        assoc_cycles = self._association.get_stage_cycles()
        norm_cycles = self._selection.get_stage_cycles()
        context_cycles = self._contextualization.get_stage_cycles()
        total_cycles = assoc_cycles + norm_cycles + context_cycles
        max_stage = max(assoc_cycles, norm_cycles, context_cycles)

        # Get energy from wrappers
        assoc_energy = self._association.get_stage_energy_pj()
        norm_energy = self._selection.get_stage_energy_pj()
        context_energy = self._contextualization.get_stage_energy_pj()
        total_energy_pj = assoc_energy + norm_energy + context_energy
        total_energy_nj = total_energy_pj / 1000.0

        # Combine energy breakdowns
        breakdown = {}
        for wrapper in [self._association, self._selection, self._contextualization]:
            for module, energy in wrapper.get_module_energy_breakdown().items():
                breakdown[module] = breakdown.get(module, 0) + energy

        # Calculate DRAM energy separately
        dram_energy_pj = breakdown.get('dram', 0)
        onchip_energy_pj = total_energy_pj - dram_energy_pj

        # Calculate power (mW): Energy (pJ) / Time (ns) at 1 GHz
        # Since cycles = ns at 1 GHz: Power = Energy_pJ / cycles
        if max_stage > 0:
            total_power_mw = total_energy_pj / max_stage
            onchip_power_mw = onchip_energy_pj / max_stage
            dram_power_mw = dram_energy_pj / max_stage
        else:
            total_power_mw = onchip_power_mw = dram_power_mw = 0

        # Throughput at 1 GHz: 1e6 / max_stage_cycles (att/ms)
        if max_stage > 0:
            throughput_per_ms = 1e6 / max_stage
        else:
            throughput_per_ms = 0

        # Area from power model
        from .power_model import MODULES
        area_um2 = sum(m.total_area_um2() for name, m in MODULES.items() if name != 'dram')
        area_mm2 = area_um2 / 1e6

        return {
            'config': {
                'seq_length': self.config.seq_length,
                'head_dim': self.config.head_dim,
                'num_heads': self.config.num_heads,
                'k_value': self.config.k_value,
                'clock_freq_mhz': self.config.clock_freq_hz / 1e6,
                'pipeline_mode': self.config.pipeline_mode.value,
            },
            'timing': {
                'total_cycles': total_cycles,
                'association_cycles': assoc_cycles,
                'normalization_cycles': norm_cycles,
                'contextualization_cycles': context_cycles,
                'max_stage_cycles': max_stage,
                'total_ns': total_cycles,  # At 1 GHz, cycles = ns
            },
            'energy': {
                'total_nj': total_energy_nj,
                'onchip_nj': onchip_energy_pj / 1000.0,
                'dram_nj': dram_energy_pj / 1000.0,
                'total_pj': total_energy_pj,
                'breakdown': breakdown,
            },
            'power': {
                'total_mw': total_power_mw,
                'onchip_mw': onchip_power_mw,
                'dram_mw': dram_power_mw,
            },
            'throughput': {
                'single_core_per_ms': throughput_per_ms / self.config.num_heads,
                'multi_core_per_ms': throughput_per_ms,
            },
            'area_mm2': area_mm2,
        }

    def get_paper_metrics(self) -> Optional[Dict[str, Any]]:
        """Get paper hardware model metrics (if available)"""
        return self._paper_metrics

    def run_paper_model(self, mode: Optional[PipelineMode] = None) -> Dict[str, Any]:
        """Run paper hardware model and return metrics

        Args:
            mode: Pipeline mode (defaults to config.pipeline_mode)

        Returns:
            Dictionary with paper-accurate metrics
        """
        if not self._paper_model:
            raise RuntimeError("Paper model not initialized (use_paper_model=False)")

        mode = mode or self.config.pipeline_mode
        self._paper_metrics = self._paper_model.run_attention(mode)
        return self._paper_metrics

    def set_pipeline_mode(self, mode: PipelineMode) -> None:
        """Change pipeline mode for all stages"""
        self.config.pipeline_mode = mode
        self._association.set_pipeline_mode(mode)
        self._selection.set_pipeline_mode(mode)
        self._contextualization.set_pipeline_mode(mode)
        if self._paper_model:
            self._paper_model.config.pipeline_mode = mode

    def clear(self) -> None:
        """Clear all state"""
        self._association.clear()
        self._selection.clear()
        self._contextualization.clear()
        self._current_head = 0
        self._results = []
        if self._paper_model:
            self._paper_model.reset()

    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self._association.get_statistics().reset_all()
        self._selection.get_statistics().reset_all()
        self._contextualization.get_statistics().reset_all()
        self.get_statistics().reset_all()

    def __str__(self) -> str:
        return (f"CAMformerSST({self.name}, "
                f"N={self.config.seq_length}, "
                f"d={self.config.head_dim}, "
                f"k={self.config.k_value})")
