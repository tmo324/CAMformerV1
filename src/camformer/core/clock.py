"""
CAMformer Clock System

Clock and timing management for simulation.
"""

from typing import Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .component import BaseComponent


@dataclass
class TimeConverter:
    """Convert between different time units"""
    frequency_hz: float
    period_ns: float
    period_ps: float

    @classmethod
    def from_frequency(cls, frequency_hz: float) -> 'TimeConverter':
        """Create converter from frequency in Hz"""
        period_ns = 1e9 / frequency_hz
        period_ps = 1e12 / frequency_hz
        return cls(frequency_hz, period_ns, period_ps)

    @classmethod
    def from_period_ns(cls, period_ns: float) -> 'TimeConverter':
        """Create converter from period in nanoseconds"""
        frequency_hz = 1e9 / period_ns
        period_ps = period_ns * 1000
        return cls(frequency_hz, period_ns, period_ps)

    def cycles_to_ns(self, cycles: int) -> float:
        """Convert cycles to nanoseconds"""
        return cycles * self.period_ns

    def cycles_to_us(self, cycles: int) -> float:
        """Convert cycles to microseconds"""
        return cycles * self.period_ns / 1000

    def cycles_to_ms(self, cycles: int) -> float:
        """Convert cycles to milliseconds"""
        return cycles * self.period_ns / 1e6

    def ns_to_cycles(self, ns: float) -> int:
        """Convert nanoseconds to cycles"""
        return int(ns / self.period_ns)

    def us_to_cycles(self, us: float) -> int:
        """Convert microseconds to cycles"""
        return int(us * 1000 / self.period_ns)


class Clock:
    """
    Clock generator for simulation components.

    Attributes:
        frequency: Clock frequency in Hz
        handler: Function to call on each clock tick
        owner: Component that owns this clock
    """

    def __init__(self, frequency: float, handler: Callable,
                 owner: Optional['BaseComponent'] = None):
        """
        Initialize clock.

        Args:
            frequency: Clock frequency in Hz (e.g., 500e6 for 500 MHz)
            handler: Handler function called on each tick
            owner: Owning component
        """
        self.frequency = frequency
        self.handler = handler
        self.owner = owner
        self.enabled = True
        self.tick_count = 0

        # Create time converter
        self.time_converter = TimeConverter.from_frequency(frequency)

    @property
    def period_ns(self) -> float:
        """Get clock period in nanoseconds"""
        return self.time_converter.period_ns

    @property
    def period_cycles(self) -> int:
        """Period is always 1 cycle"""
        return 1

    def tick(self) -> None:
        """Execute one clock tick"""
        if self.enabled:
            self.tick_count += 1
            if self.handler:
                self.handler()

    def enable(self) -> None:
        """Enable the clock"""
        self.enabled = True

    def disable(self) -> None:
        """Disable the clock"""
        self.enabled = False

    def reset(self) -> None:
        """Reset tick counter"""
        self.tick_count = 0

    def __str__(self) -> str:
        freq_mhz = self.frequency / 1e6
        return f"Clock({freq_mhz:.0f}MHz, ticks={self.tick_count})"


# Standard clock frequencies for CAMformer
class ClockFrequencies:
    """Standard clock frequencies used in CAMformer"""

    # Main system clock (500 MHz from paper)
    SYSTEM_CLOCK_HZ = 500e6

    # BA-CAM clock (matches system)
    CAM_CLOCK_HZ = 500e6

    # ADC clock (6-bit SAR ADC)
    ADC_CLOCK_HZ = 500e6

    # Memory clock
    MEMORY_CLOCK_HZ = 500e6

    @classmethod
    def get_period_ns(cls, freq_hz: float) -> float:
        """Get period in nanoseconds for given frequency"""
        return 1e9 / freq_hz
