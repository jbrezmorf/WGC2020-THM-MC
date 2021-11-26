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

# pip install wheel # error then installing bih 
pip install pyyaml attrs numpy ruamel.yaml
pip install bgem

#pip freeze
deactivate

