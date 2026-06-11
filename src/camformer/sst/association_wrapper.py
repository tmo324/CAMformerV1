"""
Association Stage Event-Driven Wrapper

Wraps the functional AssociationStage with event-driven interface.
Uses tiled BA-CAM architecture and Power × Cycles energy model.
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
from camformer.pipeline.association import AssociationStage, AssociationConfig
from .energy_model import EnergyTracker, EnergyConfig
from .power_model import PowerCyclesModel, MODULES


class AssociationWrapper(Component):
    """
    Event-driven wrapper for Association stage.

    Uses tiled BA-CAM architecture (64 tiles × 16×64) with Power × Cycles energy model.

    Events handled:
    - KEY_LOAD: Load keys into BA-CAM
    - QUERY_LOAD: Process query through BA-CAM

    Events emitted:
    - ASSOCIATION_COMPLETE: Similarity scores ready
    """

    def __init__(self, name: str, config: Optional[AssociationConfig] = None,
                 simulation: Optional['Simulation'] = None,
                 energy_config: Optional[EnergyConfig] = None,
                 pipeline_mode: PipelineMode = PipelineMode.REALISTIC,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        # Wrap the functional component
        self._stage = AssociationStage(f"{name}_func", config=config)
        self.config = self._stage.config

        # Simulation reference
        self._sim = simulation

        # Pipeline mode for timing calculation
        self._pipeline_mode = pipeline_mode
        self._pipelined = (pipeline_mode != PipelineMode.NO_PIPELINE)

        # Tiled architecture parameters
        self._num_tiles = 64
        self._rows_per_tile = 16
        self._rows_per_program = 4

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
        self._pending_queries = []
        self._current_query_idx = 0
        self._keys_loaded = False
        self._busy = False

        # Results buffer
        self._similarity_results = None

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

    def _calculate_association_metrics(self) -> None:
        """
        Calculate timing and energy for tiled association stage.

        Implements paper's tiled architecture:
        - 64 tiles × (4 program + 1 search + ADC + scale + Top-2)
        - Top-32 every 16 tiles
        - Pipelined or non-pipelined modes
        """
        self._stage_cycles = 0
        self._stage_energy_pj = 0.0
        self._module_energy = {}

        programs_per_tile = self._rows_per_tile // self._rows_per_program
        pipelined = self._pipelined

        for tile in range(self._num_tiles):
            # Program CAM (4 rows at a time)
            for _ in range(programs_per_tile):
                self._add_module_energy("key_sram", 1)
                self._add_module_energy("ba_cam", 1)
                self._stage_cycles += MODULES["ba_cam"].delay_cycles

            # Search
            self._add_module_energy("query_buffer", 1)
            self._add_module_energy("ba_cam", 1)
            self._stage_cycles += MODULES["ba_cam"].delay_cycles

            # ADC, Scaler, Top-2, Ptop Reg
            self._add_module_energy("adc", 1)
            self._add_module_energy("scaler", 1)
            self._add_module_energy("top2", 1)
            self._add_module_energy("ptop_reg", 1)

            if not pipelined:
                self._stage_cycles += MODULES["adc"].delay_cycles
                self._stage_cycles += MODULES["scaler"].delay_cycles
                self._stage_cycles += MODULES["top2"].delay_cycles
                self._stage_cycles += MODULES["ptop_reg"].delay_cycles

            # Top-32 every 16 tiles (after tile 32)
            if tile >= 32 and (tile + 1) % 16 == 0:
                self._add_module_energy("top32", 1)
                self._add_module_energy("ptop_reg", 1)
                if not pipelined:
                    self._stage_cycles += MODULES["top32"].delay_cycles
                    self._stage_cycles += MODULES["ptop_reg"].delay_cycles

            # DRAM prefetch (energy only, overlapped)
            self._add_module_energy("dram", 1)
            self._add_module_energy("value_sram", 1)

        # Pipeline drain
        if pipelined:
            self._stage_cycles += MODULES["adc"].delay_cycles
            self._stage_cycles += MODULES["scaler"].delay_cycles
            self._stage_cycles += MODULES["top2"].delay_cycles
            self._stage_cycles += MODULES["ptop_reg"].delay_cycles

        # Update statistics
        self.get_statistic("total_latency").add_data(self._stage_cycles)
        self.get_statistic("total_energy").add_data(self._stage_energy_pj)

    def _setup_impl(self) -> None:
        """Component setup"""
        self._stage.setup()
        self._output.info(f"AssociationWrapper initialized")

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
        """
        Handle incoming events.

        Args:
            event: Incoming event
        """
        self.get_statistic("events_received").add_data(1)

        if event.event_type == EventType.KEY_LOAD:
            self._handle_key_load(event)
        elif event.event_type == EventType.QUERY_LOAD:
            self._handle_query_load(event)
        elif event.event_type == EventType.CAM_SEARCH:
            self._handle_cam_search(event)
        else:
            self._output.warning(f"Unknown event type: {event.event_type}")

    def _handle_key_load(self, event: Event) -> None:
        """Handle key loading event"""
        keys = event.get_data('keys')

        # Call functional component
        latency = self._stage.load_keys(keys)

        self._keys_loaded = True
        self.get_statistic("total_latency").add_data(latency)

        # Track energy: CAM write for all key bits
        num_keys, key_dim = keys.shape
        total_bits = num_keys * key_dim
        self._energy_tracker.add_cam_write(total_bits)

        self._output.debug(f"Keys loaded, latency={latency} cycles")

        # Schedule completion event
        if self._sim:
            complete_event = Event(
                event_type=EventType.DATA_READY,
                data={'type': 'keys_loaded', 'latency': latency},
                source=self.name
            )
            self._sim.schedule_event(
                delay=latency,
                event=complete_event,
                component=self,
                handler=self._on_keys_loaded
            )

    def _on_keys_loaded(self, event: Event) -> None:
        """Called when keys are loaded"""
        self._output.debug("Keys loading complete")

    def _handle_query_load(self, event: Event) -> None:
        """Handle query loading event"""
        queries = event.get_data('queries')

        if not self._keys_loaded:
            self._output.error("Keys not loaded, cannot process queries")
            return

        # Store queries for processing
        self._pending_queries = queries
        self._current_query_idx = 0
        self._similarity_results = np.zeros(
            (len(queries), self._stage._num_keys),
            dtype=np.float32
        )

        # Start processing first query
        self._process_next_query()

    def _handle_cam_search(self, event: Event) -> None:
        """Handle single CAM search event"""
        query = event.get_data('query')
        query_idx = event.get_data('query_idx', 0)

        # Call functional component
        hamming, similarity, latency = self._stage.process_query(query)

        self.get_statistic("total_latency").add_data(latency)

        # Track energy: CAM search for this query
        num_keys = len(similarity)
        query_dim = len(query)
        search_bits = num_keys * query_dim  # Each query searches all keys
        self._energy_tracker.add_cam_search(search_bits)

        # ADC conversions for each matchline
        self._energy_tracker.add_adc(num_keys)

        # Store result
        if self._similarity_results is not None and query_idx < len(self._similarity_results):
            self._similarity_results[query_idx] = similarity

        # Schedule completion (only when simulation exists)
        if self._sim:
            complete_event = Event(
                event_type=EventType.CAM_SEARCH_COMPLETE,
                data={
                    'query_idx': query_idx,
                    'similarity': similarity,
                    'hamming': hamming,
                    'latency': latency
                },
                source=self.name
            )
            self._sim.schedule_event(
                delay=latency,
                event=complete_event,
                component=self,
                handler=self._on_query_complete
            )
        # Note: When no simulation, _process_next_query handles iteration

    def _process_next_query(self) -> None:
        """Process next query in queue (iterative when no simulation)"""
        # Use iterative approach when no simulation to avoid recursion
        while self._current_query_idx < len(self._pending_queries):
            query = self._pending_queries[self._current_query_idx]

            # Create search event
            search_event = Event(
                event_type=EventType.CAM_SEARCH,
                data={'query': query, 'query_idx': self._current_query_idx},
                source=self.name
            )

            # Handle immediately (internal event)
            self._handle_cam_search(search_event)

            # If simulation exists, it will handle the callback chain
            if self._sim:
                return

            # No simulation: advance to next query iteratively
            self._current_query_idx += 1

        # All queries done
        self._on_all_queries_complete()

    def _on_query_complete(self, event: Event) -> None:
        """Called when a query completes"""
        self._current_query_idx += 1

        # Process next query
        self._process_next_query()

    def _on_all_queries_complete(self) -> None:
        """Called when all queries are complete"""
        self._output.debug(f"All {len(self._pending_queries)} queries complete")

        # Calculate timing and energy using tiled architecture model
        self._calculate_association_metrics()

        # Send completion event to output link
        complete_event = Event(
            event_type=EventType.ASSOCIATION_COMPLETE,
            data={
                'similarity_matrix': self._similarity_results,
                'num_queries': len(self._pending_queries),
            },
            source=self.name
        )

        self.get_statistic("events_sent").add_data(1)

        if self._output_link:
            self._output_link.send(complete_event)

        if self._on_complete:
            self._on_complete(complete_event)

    # Convenience methods for direct invocation
    def load_keys(self, keys: np.ndarray) -> int:
        """Direct key loading (creates and handles event internally)"""
        event = Event(
            event_type=EventType.KEY_LOAD,
            data={'keys': keys},
            source="direct"
        )
        self._handle_key_load(event)
        return self._stage._ba_cam.get_statistics().get_summary().get(
            'total_cycles', {'value': 0}
        )['value']

    def process_queries(self, queries: np.ndarray,
                        callback: Optional[Callable] = None) -> None:
        """
        Process queries (event-driven).

        Args:
            queries: Query matrix (N, d_k)
            callback: Called when complete with similarity matrix
        """
        if callback:
            self._on_complete = callback

        event = Event(
            event_type=EventType.QUERY_LOAD,
            data={'queries': queries},
            source="direct"
        )
        self._handle_query_load(event)

    def get_results(self) -> Optional[np.ndarray]:
        """Get similarity results (after processing)"""
        return self._similarity_results

    def clear(self) -> None:
        """Clear state"""
        self._stage.clear()
        self._pending_queries = []
        self._current_query_idx = 0
        self._keys_loaded = False
        self._similarity_results = None

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
        return f"AssociationWrapper({self.name})"
