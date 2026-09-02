SHELL := /bin/bash
PYTHON ?= python3
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

.PHONY: install install-test test check validate sensitivity figures paper rtl release clean

install:
	$(PYTHON) -m pip install -e .

install-test:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m compileall -q src tests experiments
	$(MAKE) test

validate:
	$(PYTHON) -m camformer.cli.validate

sensitivity:
	$(PYTHON) -m camformer.cli.sweep --output-dir results/figures

# Regenerate the three plotted paper figures without a notebook or display.
figures:
	$(PYTHON) -m experiments.figures.generate_all

paper: validate sensitivity figures

rtl:
	bash rtl/tests/run_checks.sh

release: check paper rtl

clean:
	rm -rf build/ dist/ src/*.egg-info/ *.egg-info/ __pycache__/ .pytest_cache/ .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
