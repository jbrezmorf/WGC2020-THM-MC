#!/bin/bash
echo "Creating python environment.."
python3 -m venv --system-site-packages venv

if [ ! -f venv/bin/activate ]
then
    echo "Virtual environment 'venv' not created."
    exit 1
fi

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

