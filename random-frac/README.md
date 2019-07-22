## `01_hm_tmpl.yaml`
The hydro-mechanical model of the EGS. One day of stimulation. Result is a GMSH file with the output fields:
- `displacement` - the displacement vector [m]
- `stress` - the stress tensor [Pa]
- `cross_section_updated` - the cross section for all elements (scalar). 
  Crucial input for the subsequent TH model. Constante one [-] for the bulk 3d elements, variable [m] for the 2d fracture elements. 
- `region_id` - the region ID in the input mesh
- `displacement_divergence` - div U, can be used to modify bulk conductivity 

The soultion for the fine mesh takes about 50 iterations of the nonlinear solver and about 30min using 3 processors.

## `02_th.yaml`
The thermo-hydraulic model of the EGS. Simulation of 30 years of the geothermal heat exchanger with steady Darcy flow throuch the opened system of the fractures. The results are: 

- The average temperature on the surface of the wells (boundary of the bulk and the fracture elements). Currently approximated by the singel observe point, Flow123d has to be modified yet to get average temperature over the set of boundary regions.
- The power of the exachanger. The power is the sum of the energy fluxes through the regions `left_fr_left_well`, `left_well`, `right_fr_right_well`, `right_well` extracted from the `energy_balance.yaml` file.

## `process.py`
Script to compute a single sample:
- generate set of random fractures
- mesh the domain with fractures
- heal the mesh: contract shor edges, remove thin 3d elements
- run HM simulation
- create modified cross-section and conductivity fields
- run TH simulation
- extract temperature and power time series
- plot them


