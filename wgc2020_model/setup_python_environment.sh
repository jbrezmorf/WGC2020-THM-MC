#!/bin/bash
echo "Creating python environment.."
python3 -m venv --system-site-packages venv

source venv/bin/activate
python --version
which python
which pip
pip install --upgrade pip
pip -V

# pip install wheel # error then installing bih
# pip install ruamel.yaml h5py memoization matplotlib
#pip install pyyaml attrs numpy ruamel.yaml
#pip install -e ../bgem
#pip install -e ../MLMC

#pip freeze
deactivate

