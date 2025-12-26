"""
CAMformer Event System

Event classes for discrete-event simulation.
"""

from enum import Enum, auto
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class EventType(Enum):
    """Types of events in CAMformer simulation"""
    # General events
    CLOCK_TICK = auto()
    DATA_READY = auto()

    # BA-CAM events
    CAM_PROGRAM = auto()
    CAM_SEARCH = auto()
    CAM_SEARCH_COMPLETE = auto()
    ADC_CONVERT = auto()
    ADC_COMPLETE = auto()

    # Pipeline events
    QUERY_LOAD = auto()
    KEY_LOAD = auto()
    VALUE_LOAD = auto()

    # Stage events
    ASSOCIATION_START = auto()
    ASSOCIATION_COMPLETE = auto()
    NORMALIZATION_START = auto()
    NORMALIZATION_COMPLETE = auto()
    CONTEXTUALIZATION_START = auto()
    CONTEXTUALIZATION_COMPLETE = auto()

    # Top-K events
    TOPK_SORT = auto()
    TOPK_COMPLETE = auto()

    # SoftMax events
    SOFTMAX_START = auto()
    SOFTMAX_COMPLETE = auto()

    # MAC events
    MAC_COMPUTE = auto()
    MAC_COMPLETE = auto()

    # Memory events
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    MEMORY_COMPLETE = auto()
    DMA_PREFETCH = auto()
    DMA_COMPLETE = auto()

    # System events
    ATTENTION_START = auto()
    ATTENTION_COMPLETE = auto()
    LAYER_START = auto()
    LAYER_COMPLETE = auto()


@dataclass
class Event:
    """
    Base event class for CAMformer simulation.

    Attributes:
        event_type: Type of event
        data: Event-specific data payload
        source: Source component name
        target: Target component name
        priority: Event priority (lower = higher priority)
    """
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    target: str = ""
    priority: int = 0

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data value by key"""
        return self.data.get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """Set data value"""
        self.data[key] = value

    def __str__(self) -> str:
        return f"Event({self.event_type.name}, src={self.source}, tgt={self.target})"

    def __repr__(self) -> str:
        return self.__str__()


# Convenience event constructors
def create_cam_search_event(query_data: Any, source: str = "", target: str = "") -> Event:
    """Create a CAM search event"""
    return Event(
        event_type=EventType.CAM_SEARCH,
        data={'query': query_data},
        source=source,
        target=target
    )


def create_data_ready_event(data: Any, source: str = "", target: str = "") -> Event:
    """Create a data ready event"""
    return Event(
        event_type=EventType.DATA_READY,
        data={'payload': data},
        source=source,
        target=target
    )


def create_attention_event(q_idx: int, layer: int, head: int,
                           source: str = "", target: str = "") -> Event:
    """Create an attention computation event"""
    return Event(
        event_type=EventType.ATTENTION_START,
        data={'query_idx': q_idx, 'layer': layer, 'head': head},
        source=source,
        target=target
    )
