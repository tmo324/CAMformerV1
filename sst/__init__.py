"""
CAMformer SST Event-Driven Simulation

Event-driven wrappers for discrete-event simulation using Python SST framework.
"""

from .association_wrapper import AssociationWrapper
from .selection_wrapper import SelectionWrapper
from .contextualization_wrapper import ContextualizationWrapper
from .camformer_sst import CAMformerSST, CAMformerSSTConfig
from .simulation_runner import SimulationRunner
from .energy_model import EnergyConfig, EnergyBreakdown, EnergyTracker, estimate_attention_energy

# Re-export technology nodes for convenience
from core.technology import (
    TechnologyNode, TECH_28NM, TECH_16NM, TECH_7NM,
    DEFAULT_TECH, get_technology, print_technology_summary
)

__all__ = [
    'AssociationWrapper',
    'SelectionWrapper',
    'ContextualizationWrapper',
    'CAMformerSST',
    'CAMformerSSTConfig',
    'SimulationRunner',
    'EnergyConfig',
    'EnergyBreakdown',
    'EnergyTracker',
    'estimate_attention_energy',
    # Technology nodes
    'TechnologyNode',
    'TECH_28NM',
    'TECH_16NM',
    'TECH_7NM',
    'DEFAULT_TECH',
    'get_technology',
    'print_technology_summary',
]
