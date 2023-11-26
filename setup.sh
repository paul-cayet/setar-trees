#!/bin/bash

# setup the virtual environment
python3 -m venv .venv
source ".venv/bin/activate"

# install the project in editable mode
pip install -r requirements.txt
pip install .
