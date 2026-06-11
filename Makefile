SHELL := /bin/bash
PYTHON ?= python3

.PHONY: install install-test test check paper clean

install:
	$(PYTHON) -m pip install -e .

install-test:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) tests/test_camformer.py
	$(PYTHON) tests/test_sst.py

check:
	$(PYTHON) -m compileall -q src tests experiments
	$(MAKE) test

paper:
	$(PYTHON) compare_with_paper.py
	$(PYTHON) sensitivity_study.py

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
