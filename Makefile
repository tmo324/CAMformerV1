SHELL := /bin/bash
PYTHON ?= python3

NB_DIR := experiments/notebooks
FIG_NB := Simulate.ipynb SensitivityStudyBACAM.ipynb

.PHONY: install install-test install-figures test check paper figures clean

install:
	$(PYTHON) -m pip install -e .

install-test:
	$(PYTHON) -m pip install -e ".[test]"

install-figures:
	$(PYTHON) -m pip install -e ".[figures]"

test:
	$(PYTHON) tests/test_camformer.py
	$(PYTHON) tests/test_sst.py

check:
	$(PYTHON) -m compileall -q src tests experiments
	$(MAKE) test

paper:
	$(PYTHON) experiments/compare_with_paper.py
	$(PYTHON) experiments/sensitivity_study.py

# Regenerate the paper figures (Fig 6/8/10) by executing the plotter notebooks
# headless (see experiments/run_notebook.py — no Jupyter kernel needed).
# Requires `make install-figures` first. The figures (pareto_front.png,
# area_energy_breakdown.png, bimm_energy.png) are written into $(NB_DIR)/ as a
# side effect of the notebooks' savefig calls; the .ipynb files are not modified.
figures:
	$(foreach nb,$(FIG_NB),$(PYTHON) experiments/run_notebook.py $(NB_DIR)/$(nb) &&) true
	@echo "Regenerated in $(NB_DIR)/: pareto_front.png, area_energy_breakdown.png, bimm_energy.png"

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
