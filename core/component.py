"""
CAMformer Component System

Base classes for simulation components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .event import Event
from .link import Link
from .clock import Clock, TimeConverter
from .statistics import Statistic, Statistics
from .output import Output


@dataclass
class ComponentId:
    """Unique component identifier"""
    id: int
    name: str

    def __str__(self) -> str:
        return f"ComponentId({self.id}, {self.name})"


class Params(dict):
    """Parameter dictionary with type conversion helpers"""

    def get_int(self, key: str, default: int = 0) -> int:
        """Get parameter as integer"""
        return int(self.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get parameter as float"""
        return float(self.get(key, default))

    def get_string(self, key: str, default: str = "") -> str:
        """Get parameter as string"""
        return str(self.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get parameter as boolean"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')


class BaseComponent(ABC):
    """
    Abstract base class for all CAMformer components.

    Provides common functionality for:
    - Parameter management
    - Link management
    - Statistics collection
    - Output logging
    - Clock registration
    """

    # Class-level component counter
    _component_counter = 0

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize component.

        Args:
            name: Component name
            params: Component parameters
        """
        BaseComponent._component_counter += 1
        self._id = ComponentId(BaseComponent._component_counter, name)
        self._params = Params(params or {})
        self._simulation = None
        self._name = name

        # Component state
        self._initialized = False
        self._finalized = False

        # Links and clocks
        self._links: Dict[str, Link] = {}
        self._clocks: List[Clock] = []

        # Statistics
        self._statistics = Statistics()

        # Output
        self._output = Output(self._name, 3, 0)

        # Subcomponents
        self._subcomponents: Dict[str, 'BaseComponent'] = {}

        # Parent component
        self._parent: Optional['BaseComponent'] = None

    @property
    def id(self) -> ComponentId:
        """Get component ID"""
        return self._id

    @property
    def name(self) -> str:
        """Get component name"""
        return self._name

    @property
    def params(self) -> Params:
        """Get component parameters"""
        return self._params

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value"""
        return self._params.get(key, default)

    def get_param_int(self, key: str, default: int = 0) -> int:
        """Get a parameter as integer"""
        return self._params.get_int(key, default)

    def get_param_float(self, key: str, default: float = 0.0) -> float:
        """Get a parameter as float"""
        return self._params.get_float(key, default)

    def get_param_string(self, key: str, default: str = "") -> str:
        """Get a parameter as string"""
        return self._params.get_string(key, default)

    def get_param_bool(self, key: str, default: bool = False) -> bool:
        """Get a parameter as boolean"""
        return self._params.get_bool(key, default)

    def add_link(self, name: str, link: Link) -> Link:
        """Add a link to this component"""
        self._links[name] = link
        link._owner = self
        return link

    def get_link(self, name: str) -> Optional[Link]:
        """Get a link by name"""
        return self._links.get(name)

    def register_clock(self, frequency: float, handler: callable,
                       register_links: bool = True) -> Clock:
        """
        Register a clock handler.

        Args:
            frequency: Clock frequency in Hz
            handler: Handler function to call on each clock tick
            register_links: Whether to register this clock with all links

        Returns:
            Clock object
        """
        clock = Clock(frequency, handler, self)
        self._clocks.append(clock)

        if register_links:
            for link in self._links.values():
                link.set_default_time_base(clock.time_converter)

        return clock

    def add_statistic(self, name: str, stat: Statistic) -> Statistic:
        """Add a statistic to this component"""
        return self._statistics.add_statistic(name, stat)

    def get_statistic(self, name: str) -> Optional[Statistic]:
        """Get a statistic by name"""
        return self._statistics.get_statistic(name)

    def get_statistics(self) -> Statistics:
        """Get statistics manager"""
        return self._statistics

    def set_subcomponent(self, slot: str, component: 'BaseComponent') -> 'BaseComponent':
        """Set a subcomponent"""
        self._subcomponents[slot] = component
        component._parent = self
        return component

    def get_subcomponent(self, slot: str) -> Optional['BaseComponent']:
        """Get a subcomponent by slot name"""
        return self._subcomponents.get(slot)

    def setup(self) -> None:
        """Initialize the component (called before simulation starts)"""
        if self._initialized:
            return

        self._output.verbose(f"Setting up component: {self._name}")

        # Initialize subcomponents
        for subcomp in self._subcomponents.values():
            subcomp.setup()

        # Call component-specific setup
        self._setup_impl()

        self._initialized = True

    def finish(self) -> None:
        """Finalize the component (called after simulation ends)"""
        if self._finalized:
            return

        self._output.verbose(f"Finishing component: {self._name}")

        # Call component-specific finish
        self._finish_impl()

        # Finalize subcomponents
        for subcomp in self._subcomponents.values():
            subcomp.finish()

        self._finalized = True

    @abstractmethod
    def _setup_impl(self) -> None:
        """Component-specific setup implementation"""
        pass

    def _finish_impl(self) -> None:
        """Component-specific finish implementation (optional)"""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def __repr__(self) -> str:
        return self.__str__()


class Component(BaseComponent):
    """
    Main component class for CAMformer simulations.

    This is the primary class that most simulation components should inherit from.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

    def _setup_impl(self) -> None:
        """Default setup implementation"""
        pass

    def _finish_impl(self) -> None:
        """Default finish implementation"""
        pass
