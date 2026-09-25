import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from camformer.core.paper_hardware import PaperHardwareModel, PipelineMode, validate_against_paper
from camformer.cli.sweep import run_topk_sweep, run_tile_sweep
from experiments.figures.fig06_bimm_energy import compute_bimm_energy
from experiments.figures.fig08_area_energy_breakdown import compute_breakdowns
from experiments.figures.fig10_pareto_front import compute_pareto_points

assert validate_against_paper()
model = PaperHardwareModel()
data = {
    'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'generated': datetime.now(timezone.utc).isoformat(),
    'baseline': model.run_attention(PipelineMode.REALISTIC),
    'topk': run_topk_sweep(),
    'tile': run_tile_sweep(),
    'bimm': compute_bimm_energy(),
    'breakdowns': compute_breakdowns(),
    'pareto': compute_pareto_points(),
    'source': 'https://github.com/tmo324/CAMformerV1',
}
path = Path(__file__).resolve().parents[1] / 'p00_data/results.json'
path.write_text(json.dumps(data, indent=2, default=lambda a: a.tolist()) + '\n')
print(path)
