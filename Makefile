SHELL := /bin/bash
PYTHON ?= python3

.PHONY: install install-test test paper clean

install:
	$(PYTHON) -m pip install -e .

install-test:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m unittest discover -s tests -v || $(PYTHON) -m unittest test_camformer.py test_sst.py -v

paper:
	$(PYTHON) compare_with_paper.py
	$(PYTHON) sensitivity_study.py

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
