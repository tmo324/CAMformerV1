"""
CAMformer Link System

Inter-component communication links.
"""

from typing import Any, Callable, Optional, TYPE_CHECKING
from collections import deque

from .event import Event
from .clock import TimeConverter

if TYPE_CHECKING:
    from .component import BaseComponent
    from .simulation import Simulation


class Link:
    """
    Communication link between components.

    Links provide:
    - Event delivery between components
    - Optional latency modeling
    - Buffering for pipelined communication
    """

    def __init__(self, latency_cycles: int = 0, buffer_size: int = 16):
        """
        Initialize link.

        Args:
            latency_cycles: Delay in cycles for event delivery
            buffer_size: Maximum events in buffer (0 = unlimited)
        """
        self._latency_cycles = latency_cycles
        self._buffer_size = buffer_size

        # Link endpoints
        self._owner: Optional['BaseComponent'] = None
        self._target: Optional['BaseComponent'] = None
        self._handler: Optional[Callable[[Event], None]] = None

        # Simulation reference
        self._simulation: Optional['Simulation'] = None
        self._name: str = ""

        # Time converter
        self._time_converter: Optional[TimeConverter] = None

        # Buffer for events
        self._buffer: deque = deque(maxlen=buffer_size if buffer_size > 0 else None)

        # Statistics
        self._events_sent = 0
        self._events_received = 0

    @property
    def latency_cycles(self) -> int:
        """Get link latency in cycles"""
        return self._latency_cycles

    @latency_cycles.setter
    def latency_cycles(self, value: int) -> None:
        """Set link latency in cycles"""
        self._latency_cycles = max(0, value)

    def set_target(self, target: 'BaseComponent', handler: Callable[[Event], None]) -> None:
        """
        Set link target component and handler.

        Args:
            target: Target component
            handler: Handler function for received events
        """
        self._target = target
        self._handler = handler

    def set_default_time_base(self, time_converter: TimeConverter) -> None:
        """Set time converter for latency calculations"""
        self._time_converter = time_converter

    def send(self, event: Event) -> bool:
        """
        Send event through link.

        Args:
            event: Event to send

        Returns:
            True if event was sent, False if buffer full
        """
        # Check buffer capacity
        if self._buffer_size > 0 and len(self._buffer) >= self._buffer_size:
            return False

        self._events_sent += 1

        # If simulation is available, schedule with latency
        if self._simulation and self._handler and self._latency_cycles > 0:
            self._simulation.schedule_event(
                delay=self._latency_cycles,
                event=event,
                component=self._target,
                handler=self._handler
            )
        elif self._handler:
            # Immediate delivery (no latency or no simulation)
            self._buffer.append(event)
            self._handler(event)
            self._events_received += 1

        return True

    def receive(self) -> Optional[Event]:
        """
        Receive next event from buffer.

        Returns:
            Event if available, None otherwise
        """
        if self._buffer:
            self._events_received += 1
            return self._buffer.popleft()
        return None

    def peek(self) -> Optional[Event]:
        """
        Peek at next event without removing it.

        Returns:
            Event if available, None otherwise
        """
        if self._buffer:
            return self._buffer[0]
        return None

    def has_events(self) -> bool:
        """Check if buffer has events"""
        return len(self._buffer) > 0

    def clear(self) -> None:
        """Clear event buffer"""
        self._buffer.clear()

    def get_buffer_occupancy(self) -> int:
        """Get current buffer occupancy"""
        return len(self._buffer)

    def get_statistics(self) -> dict:
        """Get link statistics"""
        return {
            'events_sent': self._events_sent,
            'events_received': self._events_received,
            'buffer_occupancy': len(self._buffer),
            'latency_cycles': self._latency_cycles
        }

    def __str__(self) -> str:
        return f"Link({self._name}, latency={self._latency_cycles})"

    def __repr__(self) -> str:
        return self.__str__()
