.PHONY: install venv

VENV := .venv

create-venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created at $(VENV). Activate with: source $(VENV)/bin/activate"

install:
	python3 -m pip install -r requirements.txt
	@echo "Dependencies installed. Activate with: source $(VENV)/bin/activate"

jupyterlab: 
	jupyter lab