## `01_hm.yaml`
Hydro-mechanical model of the EGS. One day of stimulation. Result is a GMSH file with the output fields:
- `displacement` - the displacement vector [m]
- `stress` - the stress tensor [Pa]
- `cross_section_updated` - the cross section for all elements (scalar). 
  Crucial input for the subsequent TH model. Constante one [-] for the bulk 3d elements, variable [m] for the 2d fracture elements. 
- `region_id` - the region ID in the input mesh

The soultion for the fine mesh takes about 50 iterations of the nonlinear solver and about 30min using 3 processors.

## `02_th.yaml`

  
