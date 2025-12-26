"""
Selection Stage Event-Driven Wrapper

Wraps the functional SelectionStage with event-driven interface.
Uses Power × Cycles energy model for normalization stage.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.event import Event, EventType
from core.link import Link
from core.statistics import CounterStatistic, AccumulatorStatistic
from core.paper_hardware import PipelineMode
from pipeline.selection import SelectionStage, SelectionConfig
from .energy_model import EnergyTracker, EnergyConfig
from .power_model import MODULES


class SelectionWrapper(Component):
    """
    Event-driven wrapper for Selection stage (Normalization).

    Uses Power × Cycles energy model:
    - Final Top-32 selection
    - Softmax (LUT + Accumulator + Division)

    Events handled:
    - ASSOCIATION_COMPLETE: Similarity matrix from Association stage

    Events emitted:
    - NORMALIZATION_COMPLETE: Sparse attention weights ready
    """

    def __init__(self, name: str, config: Optional[SelectionConfig] = None,
                 simulation: Optional['Simulation'] = None,
                 energy_config: Optional[EnergyConfig] = None,
                 pipeline_mode: PipelineMode = PipelineMode.REALISTIC,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        # Wrap the functional component
        self._stage = SelectionStage(f"{name}_func", config=config)
        self.config = self._stage.config

        # Simulation reference
        self._sim = simulation

        # Pipeline mode for timing calculation
        self._pipeline_mode = pipeline_mode
        # Normalization is NOT pipelined in realistic mode
        self._pipelined = (pipeline_mode == PipelineMode.FULL_PIPELINE)

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
        self._current_row_idx = 0
        self._total_rows = 0
        self._busy = False

        # Results buffers
        self._top_values = None
        self._top_indices = None
        self._attention_weights = None
        self._input_similarity = None

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
        self.add_statistic("rows_processed", CounterStatistic(
            "rows_processed", "Rows processed", "rows"
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

    def _calculate_normalization_metrics(self) -> None:
        """
        Calculate timing and energy for normalization stage.

        Implements paper's normalization:
        - Final Top-32 selection
        - Softmax: LUT + Accumulator for each element
        - Division for normalization
        """
        self._stage_cycles = 0
        self._stage_energy_pj = 0.0
        self._module_energy = {}

        k = self.config.k_value
        pipelined = self._pipelined

        # Final Top-32
        self._add_module_energy("top32", 1)
        self._stage_cycles += MODULES["top32"].delay_cycles

        # Softmax loop: exp lookup and accumulate
        for _ in range(k):
            self._add_module_energy("softmax_lut", 1)
            self._add_module_energy("softmax_acc", 1)
            self._add_module_energy("output_buffer", 1)
            self._stage_cycles += MODULES["softmax_acc"].delay_cycles

        # Division loop
        for i in range(k):
            self._add_module_energy("output_buffer", 1)  # Read
            self._add_module_energy("softmax_div", 1)
            self._add_module_energy("output_buffer", 1)  # Write
            if not pipelined:
                self._stage_cycles += MODULES["softmax_div"].delay_cycles
            else:
                self._stage_cycles += 1

        if pipelined:
            self._stage_cycles += MODULES["softmax_div"].delay_cycles

        # Update statistics
        self.get_statistic("total_latency").add_data(self._stage_cycles)
        self.get_statistic("total_energy").add_data(self._stage_energy_pj)

    def _setup_impl(self) -> None:
        """Component setup"""
        self._stage.setup()
        self._output.info(f"SelectionWrapper initialized, k={self.config.k_value}")

    def set_simulation(self, sim: 'Simulation') -> None:
        """Set simulation reference"""
        self._sim = sim

    def configure_links(self, input_link: Link, output_link: Link) -> None:
        """Configure input and output links"""
        self._input_link = input_link
        self._output_link = output_link

        # Set this component as handler for input link
        input_link.set_target(self, self.handle_event)

    def set_on_complete(self, callback: Callable) -> None:
        """Set callback for completion"""
        self._on_complete = callback

    def handle_event(self, event: Event) -> None:
        """Handle incoming events"""
        self.get_statistic("events_received").add_data(1)

        if event.event_type == EventType.ASSOCIATION_COMPLETE:
            self._handle_association_complete(event)
        elif event.event_type == EventType.TOPK_SORT:
            self._handle_topk_sort(event)
        else:
            self._output.warning(f"Unknown event type: {event.event_type}")

    def _handle_association_complete(self, event: Event) -> None:
        """Handle similarity matrix from Association stage"""
        similarity_matrix = event.get_data('similarity_matrix')

        if similarity_matrix is None:
            self._output.error("No similarity matrix in event")
            return

        self._input_similarity = similarity_matrix
        self._total_rows = similarity_matrix.shape[0]
        self._current_row_idx = 0

        # Initialize result buffers
        k = self.config.k_value
        self._top_values = np.zeros((self._total_rows, k), dtype=np.float32)
        self._top_indices = np.zeros((self._total_rows, k), dtype=np.int32)
        self._attention_weights = np.zeros((self._total_rows, k), dtype=np.float32)

        self._output.debug(f"Processing {self._total_rows} rows with k={k}")

        # Start processing first row
        self._process_next_row()

    def _handle_topk_sort(self, event: Event) -> None:
        """Handle single row TopK+SoftMax"""
        row_idx = event.get_data('row_idx')
        scores = event.get_data('scores')

        # Call functional component
        values, indices, weights, latency = self._stage.process_row(scores)

        self.get_statistic("total_latency").add_data(latency)
        self.get_statistic("rows_processed").add_data(1)

        # Track energy: Top-K comparisons and registers
        num_elements = len(scores)
        k = self.config.k_value
        num_comparisons = num_elements  # Simplified: O(N) comparisons per row
        num_registers = k
        self._energy_tracker.add_topk(num_comparisons, num_registers)

        # Track energy: SoftMax on k elements
        self._energy_tracker.add_softmax(k)

        # Store results
        self._top_values[row_idx] = values
        self._top_indices[row_idx] = indices
        self._attention_weights[row_idx] = weights

        # Schedule completion (only when simulation exists)
        if self._sim:
            complete_event = Event(
                event_type=EventType.TOPK_COMPLETE,
                data={'row_idx': row_idx, 'latency': latency},
                source=self.name
            )
            self._sim.schedule_event(
                delay=latency,
                event=complete_event,
                component=self,
                handler=self._on_row_complete
            )
        # Note: When no simulation, _process_next_row handles iteration

    def _process_next_row(self) -> None:
        """Process next row (iterative when no simulation)"""
        # Use iterative approach when no simulation to avoid recursion
        while self._current_row_idx < self._total_rows:
            scores = self._input_similarity[self._current_row_idx]

            # Create sort event
            sort_event = Event(
                event_type=EventType.TOPK_SORT,
                data={'row_idx': self._current_row_idx, 'scores': scores},
                source=self.name
            )

            self._handle_topk_sort(sort_event)

            # If simulation exists, it will handle the callback chain
            if self._sim:
                return

            # No simulation: advance to next row iteratively
            self._current_row_idx += 1

        # All rows complete
        self._on_all_rows_complete()

    def _on_row_complete(self, event: Event) -> None:
        """Called when a row completes"""
        self._current_row_idx += 1
        self._process_next_row()

    def _on_all_rows_complete(self) -> None:
        """Called when all rows are complete"""
        self._output.debug(f"All {self._total_rows} rows complete")

        # Calculate timing and energy using Power × Cycles model
        self._calculate_normalization_metrics()

        # Send completion event
        complete_event = Event(
            event_type=EventType.NORMALIZATION_COMPLETE,
            data={
                'top_values': self._top_values,
                'top_indices': self._top_indices,
                'attention_weights': self._attention_weights,
                'num_rows': self._total_rows,
                'k': self.config.k_value,
            },
            source=self.name
        )

        self.get_statistic("events_sent").add_data(1)

        if self._output_link:
            self._output_link.send(complete_event)

        if self._on_complete:
            self._on_complete(complete_event)

    # Convenience methods
    def process_similarity(self, similarity_matrix: np.ndarray,
                           callback: Optional[Callable] = None) -> None:
        """
        Process similarity matrix (event-driven).

        Args:
            similarity_matrix: Similarity scores (N, N)
            callback: Called when complete
        """
        if callback:
            self._on_complete = callback

        event = Event(
            event_type=EventType.ASSOCIATION_COMPLETE,
            data={'similarity_matrix': similarity_matrix},
            source="direct"
        )
        self._handle_association_complete(event)

    def get_results(self) -> Dict[str, np.ndarray]:
        """Get selection results"""
        return {
            'top_values': self._top_values,
            'top_indices': self._top_indices,
            'attention_weights': self._attention_weights,
        }

    def clear(self) -> None:
        """Clear state"""
        self._current_row_idx = 0
        self._total_rows = 0
        self._top_values = None
        self._top_indices = None
        self._attention_weights = None
        self._input_similarity = None

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
        # Normalization is only pipelined in FULL_PIPELINE mode
        self._pipelined = (mode == PipelineMode.FULL_PIPELINE)

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
        return f"SelectionWrapper({self.name}, k={self.config.k_value})"
