"""Headless generators for the CAMformer paper figures."""

from .fig06_bimm_energy import generate_figure as generate_fig06
from .fig08_area_energy_breakdown import generate_figure as generate_fig08
from .fig10_pareto_front import generate_figure as generate_fig10

__all__ = ["generate_fig06", "generate_fig08", "generate_fig10"]
