#!/bin/bash
echo "Make sure you called 'source load_modules.sh' before!"
echo "Creating python environment.."
python3 -m venv env

source env/bin/activate
python --version
which python
which pip
pip install --upgrade pip
pip -V

# install packages
pip install pyyaml attrs numpy ruamel.yaml matplotlib
# our packages
pip install bgem

#pip freeze
deactivate

