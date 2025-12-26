"""
CAMformer Statistics System

Statistics collection for area, power, energy, and performance tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class StatisticType(Enum):
    """Types of statistics"""
    COUNTER = "counter"
    ACCUMULATOR = "accumulator"
    HISTOGRAM = "histogram"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


@dataclass
class StatisticInfo:
    """Information about a statistic"""
    name: str
    type: StatisticType
    description: str
    unit: str = ""
    enabled: bool = True


class Statistic(ABC):
    """Base class for all statistics"""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        self.name = name
        self.description = description
        self.unit = unit
        self.enabled = True
        self._data_points = 0

    @property
    def data_points(self) -> int:
        """Get number of data points"""
        return self._data_points

    @abstractmethod
    def add_data(self, value: Any) -> None:
        """Add a data point"""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """Get current statistic value"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset statistic to initial state"""
        pass

    def enable(self) -> None:
        """Enable statistic collection"""
        self.enabled = True

    def disable(self) -> None:
        """Disable statistic collection"""
        self.enabled = False

    @abstractmethod
    def _get_type(self) -> StatisticType:
        """Get statistic type"""
        pass


class CounterStatistic(Statistic):
    """Counter statistic - counts occurrences"""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, description, unit)
        self._count = 0

    def add_data(self, value: Union[int, float] = 1) -> None:
        """Add to counter"""
        if self.enabled:
            self._count += value
            self._data_points += 1

    def get_value(self) -> Union[int, float]:
        """Get current count"""
        return self._count

    def reset(self) -> None:
        """Reset counter"""
        self._count = 0
        self._data_points = 0

    def _get_type(self) -> StatisticType:
        return StatisticType.COUNTER


class AccumulatorStatistic(Statistic):
    """Accumulator statistic - sums values (e.g., total energy)"""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, description, unit)
        self._sum = 0.0
        self._count = 0

    def add_data(self, value: Union[int, float]) -> None:
        """Add value to accumulator"""
        if self.enabled:
            self._sum += value
            self._count += 1
            self._data_points += 1

    def get_value(self) -> float:
        """Get current sum"""
        return self._sum

    def get_average(self) -> float:
        """Get average value"""
        return self._sum / max(1, self._count)

    def get_count(self) -> int:
        """Get count of additions"""
        return self._count

    def reset(self) -> None:
        """Reset accumulator"""
        self._sum = 0.0
        self._count = 0
        self._data_points = 0

    def _get_type(self) -> StatisticType:
        return StatisticType.ACCUMULATOR


class AverageStatistic(Statistic):
    """Average statistic - running average of values"""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, description, unit)
        self._sum = 0.0
        self._count = 0

    def add_data(self, value: Union[int, float]) -> None:
        """Add value to average"""
        if self.enabled:
            self._sum += value
            self._count += 1
            self._data_points += 1

    def get_value(self) -> float:
        """Get current average"""
        return self._sum / max(1, self._count)

    def get_sum(self) -> float:
        """Get total sum"""
        return self._sum

    def get_count(self) -> int:
        """Get count"""
        return self._count

    def reset(self) -> None:
        """Reset average"""
        self._sum = 0.0
        self._count = 0
        self._data_points = 0

    def _get_type(self) -> StatisticType:
        return StatisticType.AVERAGE


class HistogramStatistic(Statistic):
    """Histogram statistic - distribution of values"""

    def __init__(self, name: str, description: str = "", unit: str = "",
                 bins: int = 10, min_val: float = 0.0, max_val: float = 100.0):
        super().__init__(name, description, unit)
        self._bins = bins
        self._min_val = min_val
        self._max_val = max_val
        self._bin_counts = [0] * bins
        self._count = 0

    def add_data(self, value: Union[int, float]) -> None:
        """Add value to histogram"""
        if self.enabled:
            self._count += 1
            self._data_points += 1

            # Determine bin
            if value < self._min_val:
                bin_idx = 0
            elif value >= self._max_val:
                bin_idx = self._bins - 1
            else:
                bin_idx = int((value - self._min_val) /
                              (self._max_val - self._min_val) * self._bins)
                bin_idx = min(bin_idx, self._bins - 1)

            self._bin_counts[bin_idx] += 1

    def get_value(self) -> Dict[str, Any]:
        """Get histogram data"""
        return {
            'bins': self._bin_counts,
            'min': self._min_val,
            'max': self._max_val,
            'count': self._count
        }

    def get_bin_counts(self) -> List[int]:
        """Get bin counts"""
        return self._bin_counts.copy()

    def reset(self) -> None:
        """Reset histogram"""
        self._bin_counts = [0] * self._bins
        self._count = 0
        self._data_points = 0

    def _get_type(self) -> StatisticType:
        return StatisticType.HISTOGRAM


class MinMaxStatistic(Statistic):
    """Min/Max statistic - tracks minimum and maximum values"""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, description, unit)
        self._min_val = float('inf')
        self._max_val = float('-inf')
        self._count = 0

    def add_data(self, value: Union[int, float]) -> None:
        """Add value to min/max"""
        if self.enabled:
            self._count += 1
            self._data_points += 1
            self._min_val = min(self._min_val, value)
            self._max_val = max(self._max_val, value)

    def get_value(self) -> Dict[str, Any]:
        """Get min/max values"""
        return {
            'min': self._min_val if self._min_val != float('inf') else None,
            'max': self._max_val if self._max_val != float('-inf') else None
        }

    def get_min(self) -> Optional[float]:
        """Get minimum value"""
        return self._min_val if self._min_val != float('inf') else None

    def get_max(self) -> Optional[float]:
        """Get maximum value"""
        return self._max_val if self._max_val != float('-inf') else None

    def reset(self) -> None:
        """Reset min/max"""
        self._min_val = float('inf')
        self._max_val = float('-inf')
        self._count = 0
        self._data_points = 0

    def _get_type(self) -> StatisticType:
        return StatisticType.MIN


class Statistics:
    """Statistics collection manager"""

    def __init__(self):
        self._statistics: Dict[str, Statistic] = {}
        self._enabled = True

    def add_statistic(self, name: str, stat: Statistic) -> Statistic:
        """Add a statistic"""
        self._statistics[name] = stat
        return stat

    def get_statistic(self, name: str) -> Optional[Statistic]:
        """Get a statistic by name"""
        return self._statistics.get(name)

    def get_all_statistics(self) -> Dict[str, Statistic]:
        """Get all statistics"""
        return self._statistics.copy()

    def enable_all(self) -> None:
        """Enable all statistics"""
        self._enabled = True
        for stat in self._statistics.values():
            stat.enable()

    def disable_all(self) -> None:
        """Disable all statistics"""
        self._enabled = False
        for stat in self._statistics.values():
            stat.disable()

    def reset_all(self) -> None:
        """Reset all statistics"""
        for stat in self._statistics.values():
            stat.reset()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all statistics"""
        summary = {}
        for name, stat in self._statistics.items():
            if stat.enabled:
                summary[name] = {
                    'type': stat._get_type().value,
                    'value': stat.get_value(),
                    'data_points': stat.data_points,
                    'unit': stat.unit
                }
        return summary

    def print_summary(self) -> None:
        """Print formatted summary of statistics"""
        print("\n" + "=" * 60)
        print("Statistics Summary")
        print("=" * 60)
        for name, stat in self._statistics.items():
            if stat.enabled:
                value = stat.get_value()
                unit = f" {stat.unit}" if stat.unit else ""
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}{unit}")
                else:
                    print(f"  {name}: {value}{unit}")
        print("=" * 60)
