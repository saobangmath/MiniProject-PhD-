.PHONY: install venv jupyterlab
SHELL := /bin/bash

VENV := .venv
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

create-venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created at $(VENV). Activate with: source $(VENV)/bin/activate"

install:
	python3 -m pip install -r requirements.txt
	@echo "Dependencies installed. Activate with: source $(VENV)/bin/activate"

jupyterlab:
	PYTHONPATH="$(CURDIR):$$PYTHONPATH" jupyter lab