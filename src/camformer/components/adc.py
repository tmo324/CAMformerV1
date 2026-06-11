"""
ADC (Analog-to-Digital Converter) Component

Converts matchline voltages from BA-CAM to digital Hamming distance values.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.component import Component
from camformer.core.statistics import CounterStatistic, AccumulatorStatistic


@dataclass
class ADCConfig:
    """Configuration for ADC"""

    # ADC parameters
    resolution_bits: int = 6        # ADC resolution (covers up to 64 distance levels)
    num_channels: int = 64          # Number of parallel ADC channels

    # Timing (cycles at 500 MHz)
    conversion_latency_cycles: int = 2  # Latency per conversion

    # Power parameters
    energy_per_conversion_pj: float = 0.5  # Energy per conversion in pJ

    # Area (28nm)
    area_per_channel_um2: float = 100.0  # Area per ADC channel

    # Voltage reference
    vref_high: float = 0.9
    vref_low: float = 0.0

    def get_total_area_mm2(self) -> float:
        """Get total ADC area"""
        area_um2 = self.num_channels * self.area_per_channel_um2
        return area_um2 * 1e-6


class ADC(Component):
    """
    Multi-channel ADC for matchline voltage conversion.

    Converts analog matchline voltages to digital Hamming distance values.
    Supports parallel conversion for efficient pipeline operation.
    """

    def __init__(self, name: str, config: Optional[ADCConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or ADCConfig()

        # Quantization levels
        self._num_levels = 2 ** self.config.resolution_bits
        self._lsb = (self.config.vref_high - self.config.vref_low) / self._num_levels

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics tracking"""
        self.add_statistic("conversions", CounterStatistic(
            "conversions", "Total ADC conversions", "ops"
        ))
        self.add_statistic("batches", CounterStatistic(
            "batches", "Number of conversion batches", "batches"
        ))
        self.add_statistic("energy", AccumulatorStatistic(
            "energy", "Total ADC energy", "pJ"
        ))
        self.add_statistic("cycles", AccumulatorStatistic(
            "cycles", "Total ADC cycles", "cycles"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"ADC initialized: {self.config.num_channels} channels, "
                         f"{self.config.resolution_bits}-bit")

    def convert(self, voltages: np.ndarray) -> tuple:
        """
        Convert analog voltages to digital codes.

        Args:
            voltages: Array of analog voltages to convert

        Returns:
            Tuple of (digital_codes, latency_cycles)
        """
        num_samples = len(voltages)

        # Quantize voltages
        # Clip to valid range
        clipped = np.clip(voltages, self.config.vref_low, self.config.vref_high)

        # Convert to digital codes
        normalized = (clipped - self.config.vref_low) / (
            self.config.vref_high - self.config.vref_low
        )
        digital_codes = np.floor(normalized * (self._num_levels - 1)).astype(np.int32)

        # Calculate timing (parallel channels)
        num_batches = int(np.ceil(num_samples / self.config.num_channels))
        latency = num_batches * self.config.conversion_latency_cycles

        # Update statistics
        self.get_statistic("conversions").add_data(num_samples)
        self.get_statistic("batches").add_data(num_batches)
        self.get_statistic("energy").add_data(
            num_samples * self.config.energy_per_conversion_pj
        )
        self.get_statistic("cycles").add_data(latency)

        self._output.debug(f"ADC converted {num_samples} samples in {latency} cycles")

        return digital_codes, latency

    def convert_to_hamming(self, voltages: np.ndarray, max_distance: int) -> tuple:
        """
        Convert matchline voltages directly to Hamming distances.

        Higher voltage = lower Hamming distance (more similar)

        Args:
            voltages: Matchline voltages
            max_distance: Maximum possible Hamming distance

        Returns:
            Tuple of (hamming_distances, latency_cycles)
        """
        digital_codes, latency = self.convert(voltages)

        # Convert digital codes to Hamming distances
        # Higher voltage code = lower distance
        max_code = self._num_levels - 1
        hamming_distances = np.round(
            (1.0 - digital_codes / max_code) * max_distance
        ).astype(np.int32)

        return hamming_distances, latency

    def get_area_mm2(self) -> float:
        """Get ADC area"""
        return self.config.get_total_area_mm2()

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics"""
        return {
            'num_channels': self.config.num_channels,
            'resolution_bits': self.config.resolution_bits,
            'area_mm2': self.get_area_mm2(),
        }

    def __str__(self) -> str:
        return f"ADC({self.name}, {self.config.num_channels}ch, {self.config.resolution_bits}b)"
