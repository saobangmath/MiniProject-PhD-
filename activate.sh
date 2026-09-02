#!/usr/bin/env bash

VENV=".venv"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Activating virtual environment ${VENV}"

source "${ROOT}/${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"