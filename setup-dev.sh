#!/bin/bash

# setup the virtual environment
python3 -m venv .venv
source ".venv/bin/activate"

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools

# install the project in editable mode
pip install -r requirements-dev.txt
pip install -e .

# # install the git hook scripts
pre-commit install
