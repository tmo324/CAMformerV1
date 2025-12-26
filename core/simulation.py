"""
CAMformer Simulation Engine

Main simulation engine that manages components, events, and time advancement.
"""

import heapq
from enum import Enum
from typing import Optional, Callable, Dict, List

from .event import Event
from .component import Component
from .statistics import Statistics
from .output import Output


class SimulationState(Enum):
    """Simulation execution states"""
    INIT = "init"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


class ScheduledEvent:
    """Event scheduled for future execution"""

    def __init__(self, time: int, priority: int, event: Event,
                 component: Component, handler: Callable):
        self.time = time  # Simulation time in cycles
        self.priority = priority  # Event priority (lower = higher priority)
        self.event = event
        self.component = component
        self.handler = handler

    def __lt__(self, other):
        """For heapq ordering: time first, then priority"""
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


class Simulation:
    """
    Main CAMformer simulation engine.

    Manages discrete-event simulation with:
    - Component lifecycle management
    - Event scheduling and delivery
    - Time advancement
    - Statistics collection
    - Clock management
    """

    def __init__(self, name: str = "CAMformer_Simulation"):
        self.name = name
        self.state = SimulationState.INIT

        # Core simulation data
        self._current_time = 0
        self._end_time = None
        self._event_queue: List[ScheduledEvent] = []
        self._components: Dict[str, Component] = {}
        self._links: Dict[str, 'Link'] = {}

        # Statistics and output
        self._statistics = Statistics()
        self._output = Output("Simulation", 3, 0)

        # Simulation control
        self._running = False
        self._paused = False

        # Event counters
        self._total_events = 0
        self._events_processed = 0

    def add_component(self, name: str, component: Component) -> Component:
        """Add a component to the simulation"""
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists")

        self._components[name] = component
        component._simulation = self
        component._name = name

        self._output.verbose(f"Added component: {name}")
        return component

    def get_component(self, name: str) -> Optional[Component]:
        """Get a component by name"""
        return self._components.get(name)

    def get_all_components(self) -> Dict[str, Component]:
        """Get all components"""
        return self._components.copy()

    def add_link(self, name: str, link: 'Link') -> 'Link':
        """Add a link to the simulation"""
        if name in self._links:
            raise ValueError(f"Link '{name}' already exists")

        self._links[name] = link
        link._simulation = self
        link._name = name

        self._output.verbose(f"Added link: {name}")
        return link

    def schedule_event(self, delay: int, event: Event, component: Component,
                       handler: Callable, priority: int = 0) -> None:
        """
        Schedule an event for future execution.

        Args:
            delay: Delay in simulation cycles
            event: Event to schedule
            component: Target component
            handler: Handler function to call
            priority: Event priority (lower = higher priority)
        """
        execution_time = self._current_time + delay
        scheduled_event = ScheduledEvent(
            time=execution_time,
            priority=priority,
            event=event,
            component=component,
            handler=handler
        )

        heapq.heappush(self._event_queue, scheduled_event)
        self._total_events += 1
        self._output.debug(f"Scheduled event for time {execution_time}")

    def run(self, end_time: Optional[int] = None) -> None:
        """
        Run the simulation.

        Args:
            end_time: Maximum simulation time in cycles (None for unlimited)
        """
        if self.state not in (SimulationState.INIT, SimulationState.PAUSED):
            raise RuntimeError("Simulation already running or completed")

        self._end_time = end_time
        self.state = SimulationState.RUNNING
        self._running = True
        self._paused = False

        self._output.info(f"Starting simulation: {self.name}")

        # Initialize all components
        self._initialize_components()

        # Main simulation loop
        self._simulation_loop()

        # Finalize all components
        self._finalize_components()

        self.state = SimulationState.FINISHED
        self._running = False

        self._output.info(f"Simulation completed at time {self._current_time}")
        self._output.info(f"Total events processed: {self._events_processed}")

    def run_cycles(self, cycles: int) -> None:
        """
        Run simulation for specified number of cycles.

        Args:
            cycles: Number of cycles to run
        """
        self.run(end_time=self._current_time + cycles)

    def pause(self) -> None:
        """Pause the simulation"""
        if self.state == SimulationState.RUNNING:
            self._paused = True
            self.state = SimulationState.PAUSED
            self._output.verbose("Simulation paused")

    def resume(self) -> None:
        """Resume the simulation"""
        if self.state == SimulationState.PAUSED:
            self._paused = False
            self.state = SimulationState.RUNNING
            self._output.verbose("Simulation resumed")

    def stop(self) -> None:
        """Stop the simulation"""
        self._running = False
        self.state = SimulationState.FINISHED
        self._output.verbose("Simulation stopped")

    def get_current_time(self) -> int:
        """Get current simulation time in cycles"""
        return self._current_time

    def get_statistics(self) -> Statistics:
        """Get statistics collector"""
        return self._statistics

    def get_output(self) -> Output:
        """Get output logger"""
        return self._output

    def _initialize_components(self) -> None:
        """Initialize all components"""
        for name, component in self._components.items():
            try:
                component.setup()
                self._output.verbose(f"Initialized component: {name}")
            except Exception as e:
                self._output.fatal(f"Failed to initialize component {name}: {e}")
                raise

    def _finalize_components(self) -> None:
        """Finalize all components"""
        for name, component in self._components.items():
            try:
                component.finish()
                self._output.verbose(f"Finalized component: {name}")
            except Exception as e:
                self._output.fatal(f"Failed to finalize component {name}: {e}")

    def _simulation_loop(self) -> None:
        """Main simulation event loop"""
        while self._running and self._event_queue:
            # Check if we've reached the end time
            if self._end_time is not None and self._current_time >= self._end_time:
                break

            # Get next event
            scheduled_event = heapq.heappop(self._event_queue)

            # Advance time
            self._current_time = scheduled_event.time

            # Check for pause
            if self._paused:
                # Re-add event and wait
                heapq.heappush(self._event_queue, scheduled_event)
                break

            if not self._running:
                break

            # Execute the event
            try:
                scheduled_event.handler(scheduled_event.event)
                self._events_processed += 1
                self._output.debug(f"Executed event at time {self._current_time}")
            except Exception as e:
                self._output.fatal(f"Error executing event: {e}")
                self.state = SimulationState.ERROR
                raise

    def get_simulation_stats(self) -> Dict:
        """Get simulation-level statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'current_time_cycles': self._current_time,
            'total_events': self._total_events,
            'events_processed': self._events_processed,
            'pending_events': len(self._event_queue),
            'num_components': len(self._components)
        }

    def print_summary(self) -> None:
        """Print simulation summary"""
        stats = self.get_simulation_stats()
        print("\n" + "=" * 60)
        print(f"Simulation Summary: {self.name}")
        print("=" * 60)
        print(f"  State: {stats['state']}")
        print(f"  Total cycles: {stats['current_time_cycles']:,}")
        print(f"  Events processed: {stats['events_processed']:,}")
        print(f"  Components: {stats['num_components']}")
        print("=" * 60)

        # Print component statistics
        for name, component in self._components.items():
            print(f"\n  Component: {name}")
            comp_stats = component.get_statistics().get_summary()
            for stat_name, stat_info in comp_stats.items():
                value = stat_info['value']
                unit = stat_info.get('unit', '')
                if isinstance(value, float):
                    print(f"    {stat_name}: {value:.4f} {unit}")
                else:
                    print(f"    {stat_name}: {value} {unit}")

    def __str__(self) -> str:
        return f"Simulation({self.name}, state={self.state.value}, time={self._current_time})"

    def __repr__(self) -> str:
        return self.__str__()
