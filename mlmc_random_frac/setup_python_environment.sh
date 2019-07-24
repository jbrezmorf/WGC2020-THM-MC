#!/bin/bash
echo "Make sure you called 'source load_modules.sh' before!"
echo "Creating python environment.."
python3.6 -m venv env

source env/bin/activate
python --version
which python
which pip
pip install --upgrade pip
pip -V

pip install gmsh-sdk

#pip freeze
deactivate

