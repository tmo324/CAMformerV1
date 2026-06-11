"""
Contextualization Stage Event-Driven Wrapper

Wraps the functional ContextualizationStage with event-driven interface.
Uses Power × Cycles energy model for pipelined MAC operations.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.event import Event, EventType
from camformer.core.link import Link
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic
from camformer.core.paper_hardware import PipelineMode
from camformer.pipeline.contextualization import ContextualizationStage, ContextualizationConfig
from .energy_model import EnergyTracker, EnergyConfig
from .power_model import MODULES


class ContextualizationWrapper(Component):
    """
    Event-driven wrapper for Contextualization stage.

    Uses Power × Cycles energy model:
    - Pipelined MAC operations: 64×32/8 = 256 iterations
    - Value SRAM reads
    - 8 parallel BF16 MACs

    Events handled:
    - VALUE_LOAD: Load value matrix into V-SRAM
    - NORMALIZATION_COMPLETE: Sparse attention weights from Selection stage

    Events emitted:
    - CONTEXTUALIZATION_COMPLETE: Output vectors ready
    """

    def __init__(self, name: str, config: Optional[ContextualizationConfig] = None,
                 simulation: Optional['Simulation'] = None,
                 energy_config: Optional[EnergyConfig] = None,
                 pipeline_mode: PipelineMode = PipelineMode.REALISTIC,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        # Wrap the functional component
        self._stage = ContextualizationStage(f"{name}_func", config=config)
        self.config = self._stage.config

        # Simulation reference
        self._sim = simulation

        # Pipeline mode for timing calculation
        self._pipeline_mode = pipeline_mode
        self._pipelined = (pipeline_mode != PipelineMode.NO_PIPELINE)

        # MAC parallelism
        self._c_par = 8  # 8 parallel MACs

        # Energy tracking (legacy)
        self._energy_config = energy_config or EnergyConfig()
        self._energy_tracker = EnergyTracker(self._energy_config)

        # Power × Cycles metrics
        self._stage_cycles = 0
        self._stage_energy_pj = 0.0
        self._module_energy: Dict[str, float] = {}

        # Links
        self._input_link: Optional[Link] = None
        self._output_link: Optional[Link] = None

        # State
        self._values_loaded = False
        self._current_output_idx = 0
        self._total_outputs = 0
        self._busy = False

        # Input data
        self._attention_weights = None
        self._value_indices = None

        # Results buffer
        self._outputs = None

        # Callbacks
        self._on_complete: Optional[Callable] = None

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup wrapper statistics"""
        self.add_statistic("events_received", CounterStatistic(
            "events_received", "Events received", "events"
        ))
        self.add_statistic("events_sent", CounterStatistic(
            "events_sent", "Events sent", "events"
        ))
        self.add_statistic("total_latency", AccumulatorStatistic(
            "total_latency", "Total processing latency", "cycles"
        ))
        self.add_statistic("outputs_computed", CounterStatistic(
            "outputs_computed", "Outputs computed", "outputs"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total energy", "pJ"
        ))

    def _add_module_energy(self, module_name: str, times: int = 1) -> None:
        """Add energy for a module using Power × Cycles model"""
        if module_name in MODULES:
            module = MODULES[module_name]
            energy = module.energy_pj(times)
            self._stage_energy_pj += energy
            self._module_energy[module_name] = self._module_energy.get(module_name, 0) + energy

    def _calculate_contextualization_metrics(self) -> None:
        """
        Calculate timing and energy for contextualization stage.

        Implements paper's sparse matmul:
        - 64 × 32 MACs / 8 parallel = 256 iterations
        - Each iteration: Value SRAM read + MAC
        - Pipelined: 1 cycle per iteration + MAC latency drain
        """
        self._stage_cycles = 0
        self._stage_energy_pj = 0.0
        self._module_energy = {}

        d_v = self.config.head_dim  # 64
        k = self.config.k_value      # 32
        pipelined = self._pipelined

        # Number of MAC iterations
        num_iterations = (d_v * k) // self._c_par  # 64 × 32 / 8 = 256

        for _ in range(num_iterations):
            self._add_module_energy("value_sram", 1)
            self._add_module_energy("bf16_mac", 1)
            if pipelined:
                self._stage_cycles += 1  # Pipelined: 1 cycle per iteration
            else:
                self._stage_cycles += MODULES["bf16_mac"].delay_cycles

        # Pipeline drain
        if pipelined:
            self._stage_cycles += MODULES["bf16_mac"].delay_cycles

        # Update statistics
        self.get_statistic("total_latency").add_data(self._stage_cycles)
        self.get_statistic("total_energy").add_data(self._stage_energy_pj)

    def _setup_impl(self) -> None:
        """Component setup"""
        self._stage.setup()
        self._output.info(f"ContextualizationWrapper initialized, "
                         f"d_v={self.config.head_dim}, k={self.config.k_value}")

    def set_simulation(self, sim: 'Simulation') -> None:
        """Set simulation reference"""
        self._sim = sim

    def configure_links(self, input_link: Link, output_link: Link) -> None:
        """Configure input and output links"""
        self._input_link = input_link
        self._output_link = output_link

        input_link.set_target(self, self.handle_event)

    def set_on_complete(self, callback: Callable) -> None:
        """Set callback for completion"""
        self._on_complete = callback

    def handle_event(self, event: Event) -> None:
        """Handle incoming events"""
        self.get_statistic("events_received").add_data(1)

        if event.event_type == EventType.VALUE_LOAD:
            self._handle_value_load(event)
        elif event.event_type == EventType.NORMALIZATION_COMPLETE:
            self._handle_normalization_complete(event)
        elif event.event_type == EventType.MAC_COMPUTE:
            self._handle_mac_compute(event)
        else:
            self._output.warning(f"Unknown event type: {event.event_type}")

    def _handle_value_load(self, event: Event) -> None:
        """Handle value matrix loading"""
        values = event.get_data('values')

        # Call functional component
        latency = self._stage.load_values(values)

        self._values_loaded = True
        self.get_statistic("total_latency").add_data(latency)

        # Track energy: SRAM write for all values (BF16 = 16 bits = 2 bytes per word)
        num_values, value_dim = values.shape
        num_words = num_values * value_dim
        self._energy_tracker.add_sram_write(num_words)

        self._output.debug(f"Values loaded, latency={latency} cycles")

        # Schedule completion
        if self._sim:
            complete_event = Event(
                event_type=EventType.DATA_READY,
                data={'type': 'values_loaded', 'latency': latency},
                source=self.name
            )
            self._sim.schedule_event(
                delay=latency,
                event=complete_event,
                component=self,
                handler=self._on_values_loaded
            )

    def _on_values_loaded(self, event: Event) -> None:
        """Called when values are loaded"""
        self._output.debug("Values loading complete")

    def _handle_normalization_complete(self, event: Event) -> None:
        """Handle sparse attention from Selection stage"""
        attention_weights = event.get_data('attention_weights')
        value_indices = event.get_data('top_indices')

        if attention_weights is None or value_indices is None:
            self._output.error("Missing attention weights or indices")
            return

        if not self._values_loaded:
            self._output.error("Values not loaded, cannot compute outputs")
            return

        self._attention_weights = attention_weights
        self._value_indices = value_indices
        self._total_outputs = attention_weights.shape[0]
        self._current_output_idx = 0

        # Initialize output buffer
        d_v = self.config.head_dim
        self._outputs = np.zeros((self._total_outputs, d_v), dtype=np.float32)

        self._output.debug(f"Computing {self._total_outputs} outputs")

        # Start processing
        self._process_next_output()

    def _handle_mac_compute(self, event: Event) -> None:
        """Handle single output computation"""
        output_idx = event.get_data('output_idx')
        weights = event.get_data('weights')
        indices = event.get_data('indices')

        # Call functional component
        output, latency = self._stage.compute_output_vectorized(weights, indices)

        self.get_statistic("total_latency").add_data(latency)
        self.get_statistic("outputs_computed").add_data(1)

        # Track energy: SRAM reads for k values
        k = len(indices)
        d_v = self.config.head_dim
        sram_words = k * d_v
        self._energy_tracker.add_sram_read(sram_words)

        # Track energy: k * d_v MACs for weighted sum
        num_macs = k * d_v
        self._energy_tracker.add_mac(num_macs)

        # Store result
        self._outputs[output_idx] = output

        # Schedule completion (only when simulation exists)
        if self._sim:
            complete_event = Event(
                event_type=EventType.MAC_COMPLETE,
                data={'output_idx': output_idx, 'latency': latency},
                source=self.name
            )
            self._sim.schedule_event(
                delay=latency,
                event=complete_event,
                component=self,
                handler=self._on_output_complete
            )
        # Note: When no simulation, _process_next_output handles iteration

    def _process_next_output(self) -> None:
        """Process next output (iterative when no simulation)"""
        # Use iterative approach when no simulation to avoid recursion
        while self._current_output_idx < self._total_outputs:
            weights = self._attention_weights[self._current_output_idx]
            indices = self._value_indices[self._current_output_idx]

            # Create MAC event
            mac_event = Event(
                event_type=EventType.MAC_COMPUTE,
                data={
                    'output_idx': self._current_output_idx,
                    'weights': weights,
                    'indices': indices
                },
                source=self.name
            )

            self._handle_mac_compute(mac_event)

            # If simulation exists, it will handle the callback chain
            if self._sim:
                return

            # No simulation: advance to next output iteratively
            self._current_output_idx += 1

        # All outputs complete
        self._on_all_outputs_complete()

    def _on_output_complete(self, event: Event) -> None:
        """Called when an output completes"""
        self._current_output_idx += 1
        self._process_next_output()

    def _on_all_outputs_complete(self) -> None:
        """Called when all outputs are complete"""
        self._output.debug(f"All {self._total_outputs} outputs complete")

        # Calculate timing and energy using Power × Cycles model
        self._calculate_contextualization_metrics()

        # Send completion event
        complete_event = Event(
            event_type=EventType.CONTEXTUALIZATION_COMPLETE,
            data={
                'outputs': self._outputs,
                'num_outputs': self._total_outputs,
            },
            source=self.name
        )

        self.get_statistic("events_sent").add_data(1)

        if self._output_link:
            self._output_link.send(complete_event)

        if self._on_complete:
            self._on_complete(complete_event)

    # Convenience methods
    def load_values(self, values: np.ndarray) -> int:
        """Direct value loading"""
        event = Event(
            event_type=EventType.VALUE_LOAD,
            data={'values': values},
            source="direct"
        )
        self._handle_value_load(event)
        return 0  # Latency tracked internally

    def process_attention(self, attention_weights: np.ndarray,
                          value_indices: np.ndarray,
                          callback: Optional[Callable] = None) -> None:
        """
        Process attention weights (event-driven).

        Args:
            attention_weights: Sparse attention weights (N, k)
            value_indices: Value indices (N, k)
            callback: Called when complete
        """
        if callback:
            self._on_complete = callback

        event = Event(
            event_type=EventType.NORMALIZATION_COMPLETE,
            data={
                'attention_weights': attention_weights,
                'top_indices': value_indices,
            },
            source="direct"
        )
        self._handle_normalization_complete(event)

    def get_results(self) -> Optional[np.ndarray]:
        """Get output results"""
        return self._outputs

    def clear(self) -> None:
        """Clear state"""
        self._stage.clear()
        self._values_loaded = False
        self._current_output_idx = 0
        self._total_outputs = 0
        self._attention_weights = None
        self._value_indices = None
        self._outputs = None

    def get_area_mm2(self) -> float:
        """Get area from wrapped component"""
        return self._stage.get_area_mm2()

    def get_energy_tracker(self) -> EnergyTracker:
        """Get energy tracker"""
        return self._energy_tracker

    def reset_energy(self) -> None:
        """Reset energy tracking"""
        self._energy_tracker.reset()
        self._stage_cycles = 0
        self._stage_energy_pj = 0.0
        self._module_energy = {}

    def set_pipeline_mode(self, mode: PipelineMode) -> None:
        """Set pipeline mode"""
        self._pipeline_mode = mode
        self._pipelined = (mode != PipelineMode.NO_PIPELINE)

    def get_stage_cycles(self) -> int:
        """Get stage cycles from Power × Cycles model"""
        return self._stage_cycles

    def get_stage_energy_pj(self) -> float:
        """Get stage energy in pJ from Power × Cycles model"""
        return self._stage_energy_pj

    def get_module_energy_breakdown(self) -> Dict[str, float]:
        """Get energy breakdown by module"""
        return self._module_energy.copy()

    def __str__(self) -> str:
        return f"ContextualizationWrapper({self.name})"
