## `01_hm.yaml`
The hydro-mechanical model of the EGS. One day of stimulation. Result is a GMSH file with the output fields:
- `displacement` - the displacement vector [m]
- `stress` - the stress tensor [Pa]
- `cross_section_updated` - the cross section for all elements (scalar). 
  Crucial input for the subsequent TH model. Constante one [-] for the bulk 3d elements, variable [m] for the 2d fracture elements. 
- `region_id` - the region ID in the input mesh

The soultion for the fine mesh takes about 50 iterations of the nonlinear solver and about 30min using 3 processors.

## `02_th.yaml`
The thermo-hydraulic model of the EGS. Simulation of 30 years of the geothermal heat exchanger with steady Darcy flow throuch the opened system of the fractures. The results are: 

- The average temperature on the surface of the wells (boundary of the bulk and the fracture elements). Currently approximated by the singel observe point, Flow123d has to be modified yet to get average temperature over the set of boundary regions.
- The power of the exachanger. The power is the sum of the energy fluxes through the regions `left_fr_left_well`, `left_well`, `right_fr_right_well`, `right_well` extracted from the `energy_balance.yaml` file.


## `plot_power.sh` 
Plot the change of the temperature and power with time. TODO: Should be done in Python as part of the extractionn script.

## `parse_temp.py`
Extraction of the time series of the temperatures from the observe point output. TODO: Should be replaced by the extraction from the new auxiliary output of the temperature averages.

## `three_frac_symmetric.msh`
Fine mesh, wells approximated by 12 side polygons, three fixed fractures, about 25 000 elements.
## `three_frac_symmetric_5000el.msh`
Coarse mesh, wells approximated by 6 side polygons, about 5000 elements.







# Running with Singularity on Charon

## Building Flow123d

1. clone repositories
> git clone `<Flow123d>`\
> git clone `<WGC2020>`
2. run Singularity container (replace "../../workspace/" with your own workspace path - has to include both flow123d and wgc2020 directories)
> singularity shell -B ../../workspace/:/../../workspace docker://flow123d/flow-dev-gnu-rel:3.1.0
3. go to flow123d repository, checkout your branch and build it inside container
4. remember the path to flow123d binary, e.g. workspace/flow123d/bin/flow123d

## Running Python script:
0. update 2021-12: no need to load python modules anymore - use python inside singularity
1. go to WG2020 repository (and your model subdir)
2. for the first time, create virtual environment with necessary packages (possibly add new ones)
> ./setup_python_environment.sh
3. activate env
> source env/bin/activate
4. Possibly edit config.yaml and set correct relative path to flow123d build
5. run process (you are inside singularity and python env)
> python3 process.py
