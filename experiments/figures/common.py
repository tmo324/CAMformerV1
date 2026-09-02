"""Shared paths, data loading, and plotting setup for paper figures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES_CSV = REPO_ROOT / "data" / "inputs" / "hardware_modules.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"


@dataclass(frozen=True)
class RawModule:
    """One unscaled row from the paper's module-characterization table."""

    name: str
    delay_cycles: int
    power_mw: float
    area_um2: float
    source: str
    node_nm: int
    number: int
    note: str


def load_raw_modules(path: Path = MODULES_CSV) -> dict[str, RawModule]:
    """Load the raw module table without applying technology scaling."""
    modules: dict[str, RawModule] = {}
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            module = RawModule(
                name=row["Module"],
                delay_cycles=int(row["Delay (cycles)"]),
                power_mw=float(row["Power (mW)"]),
                area_um2=float(row["Area (um^2)"]),
                source=row["Source"],
                node_nm=int(row["Node (nm)"]),
                number=int(row["Number"]),
                note=row["Note"],
            )
            modules[module.name] = module
    return modules


def apply_paper_style() -> str:
    """Use the paper font when available and a metrically similar fallback."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif")
    selected = next(name for name in candidates if name in available)
    plt.rcParams.update({"font.family": selected})
    return selected


def prepare_output(path: Path | str) -> Path:
    """Normalize an output path and create its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
