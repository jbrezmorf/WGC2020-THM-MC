# Model of an artificial Enhanced Geothermal System (EGS)

This repository contains a stochastic model of an EGS, developed for the Wolrd Geothermal Congress 2020+1.
The main model is in the directory `wgc2020_model`. 
The model uses Monte Carlo method for random generation of fracture network.
For each fracture network, the simulation of EGS is run and the quantities of interest (temperature, power) are collected.
The result is then in form of distribution characteristics of the quantities of interest: mean, standard deviation, probability density function.

This provides a much higher quality of information in comparison to single simulation,
since the fracture network and hydrogeological parameters are mostly estimated with high level of uncertainty.

The model uses two main softwares:
- GeoMop http://geomop.github.io/, two of its libraries in particular:
  - MLMC https://github.com/GeoMop/MLMC
  - BGEM https://github.com/GeoMop/bgem
- Flow123d http://flow123d.github.io/


## Single sample model
Each sample takes several computation steps:
1. random fracture network sample generation (by BGEM)
2. creation of compatible 1D-2D-3D computational mesh (by BGEM)
3. computational mesh optimization by HealMesh algorithm (by BGEM)
4. hydraulic stimulation model (poroelastic model in Flow123d)
5. modification of hydraulic parameters based on mechanical changes (by BGEM)
6. heat transfer model with modified parameters (thermo-hydraulic model in Flow123d)
7. heat transfer model with original parameters as a reference (thermo-hydraulic model in Flow123d)


## Installation
For running the simulation one has to install Python 3 and MLMC package.
Sample computation then requires running BGEM and Flow123d inside Geomop 2.0.0 Docker container (download the image at http://geomop.github.io/).
This obviously requires having Docker (https://www.docker.com/) or Singularity (https://apptainer.org/) installed for running the container.

For installing MLMC in virtual environment one can use the enclosed script `setup_python_envinroment.sh`.


## Running the simulation
Model can be run locally by script `run_process_local.sh` or on cluster (Metacentrum in CZ tested) by script `run_process_pbs.sh`.
In case of cluster run, one has to specify the computational resources in file `config_pbs.yaml`, the PBS script itself is then created automatically.
In both cases, the main configuration is loaded from file `config.yaml`.
