#!/usr/bin/env python3
import sys
import os

WGC_DIR = "/storage/liberec3-tul/home/martin_spetlik/WGC"
WORK_DIR = os.path.join(WGC_DIR, 'mlmc_frac')

sys.path.append(os.path.join(WGC_DIR, 'MLMC/src'))
sys.path.append(WORK_DIR)
import yaml
import numpy as np
import gmsh_io


def load_config_dict():
    with open(os.path.join(WORK_DIR, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def prepare_th_input(sample_dir):
    """
    Prepare FieldFE input file for the TH simulation.
    """

    # #we have to read region names from the input mesh
    # input_mesh = gmsh_io.GmshIO(config_dict['hm_params']['mesh'])
    #
    # is_bc_region = {}
    # for name, (id, _) in input_mesh.physical.items():
    #     unquoted_name = name.strip("\"'")
    #     is_bc_region[id] = (unquoted_name[0] == '.')

    config_dict = load_config_dict()

    # read mesh and mechanical output data
    mesh = gmsh_io.GmshIO(os.path.join(sample_dir, 'output_hm/mechanics.msh'))

    n_bulk = len(mesh.elements)
    ele_ids = np.zeros(n_bulk, dtype=int)
    for i, id_bulk in zip(range(n_bulk), mesh.elements.items()):
        ele_ids[i] = id_bulk[0]

    init_fr_cs = float(config_dict['hm_params']['fr_cross_section'])
    init_fr_K = float(config_dict['hm_params']['fr_conductivity'])
    init_bulk_K = float(config_dict['hm_params']['bulk_conductivity'])

    field_cs = mesh.element_data['cross_section_updated'][1]

    K = np.zeros((n_bulk, 1), dtype=float)
    cs = np.zeros((n_bulk, 1), dtype=float)
    for i, valcs in zip(range(n_bulk), field_cs[1].values()):
        cs_el = valcs[0]
        cs[i, 0] = cs_el
        if cs_el != 1.0:  # if cross_section == 1, i.e. 3d bulk
            K[i, 0] = init_fr_K * (cs_el * cs_el) / (init_fr_cs * init_fr_cs)
        else:
            K[i, 0] = init_bulk_K

    # mesh.write_fields('output_hm/th_input.msh', ele_ids, {'conductivity': K})
    th_input_file = os.path.join(sample_dir, 'output_hm/th_input.msh')
    with open(th_input_file, "w") as fout:
        mesh.write_ascii(fout)
        mesh.write_element_data(fout, ele_ids, 'conductivity', K)
        mesh.write_element_data(fout, ele_ids, 'cross_section_updated', cs)

    # create field for K (copy cs)
    # posun dat K do casu 0
    # read original K = oK (define in config yaml)
    # read original cs = ocs (define in config yaml)
    # compute K = oK * (cs/ocs)^2
    # write K
    #
    # posun dat cs do casu 0
    # write cs
    #
    # mesh.element_data.

    # @attr.s(auto_attribs=True)
    # class ValueDesctription:
    #     time: float
    #     position: str
    #     quantity: str
    #     unit: str


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_th_input(sys.argv[1])

