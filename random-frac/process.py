import sys
sys.path.append('../dfn/src')
sys.path.append('../MLMC/src')

import os
import shutil
import subprocess
import yaml
import attr
import numpy as np
# import matplotlib.pyplot as plt

from gmsh_api import gmsh
from gmsh_api import options
import fracture

@attr.s(auto_attribs=True)
class ValueDesctription:
    time: float
    position: str
    quantity: str
    unit: str


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


def prepare_mesh(config_dict):
    geom = config_dict["geometry"]
    dimensions = geom["box_dimensions"]
    well_z0, well_z1 = geom["well_openning"]
    well_r = geom["well_effective_radius"]
    well_dist = geom["well_distance"]

    # generate fracture set
    volume = np.product(dimensions)
    pop = fracture.Population(volume)
    pop.initialize(geom["fracture_stats"])
    pop.set_sample_range([1, well_dist], max_sample_size=geom["n_frac_limit"])
    print("total mean size: ", pop.mean_size())
    fractures = pop.sample()

    factory = gmsh.GeometryOCC("three_frac_symmetric", verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 1e-5
    gopt.ToleranceBoolean = 1e-3
    # gopt.MatchMeshTolerance = 1e-1

    # Main box
    box = factory.box(dimensions).set_region("box")
    side_z = factory.rectangle([dimensions[0], dimensions[1]])
    side_y = factory.rectangle([dimensions[0], dimensions[2]])
    side_x = factory.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box = box.get_boundary().copy()

    # two vertical cut-off wells, just permeable part
    left_center = [-well_dist/2, 0, 0]
    right_center = [+well_dist/2, 0, 0]
    left_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
        .translate([0, 0, well_z0]).translate(left_center)
    right_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
        .translate([0, 0, well_z0]).translate(right_center)
    b_right_well = right_well.get_boundary()
    b_left_well = left_well.get_boundary()

    print("n fractures:", len(fractures))
    fractures = factory.make_fractures(fractures, factory.rectangle())
    fractures_group = factory.group(*fractures)

    # drilled box and its boundary
    box_drilled = box.cut(left_well, right_well)

    # fractures, fragmented, fractures boundary
    fractures_group = fractures_group.intersect(box_drilled.copy())
    box_fr, fractures_fr = factory.fragment(box_drilled, fractures_group)
    b_box_fr = box_fr.get_boundary()
    b_left_r = b_box_fr.select_by_intersect(b_left_well).set_region(".left_well")
    b_right_r = b_box_fr.select_by_intersect(b_right_well).set_region(".right_well")

    box_all = []
    for name, side_tool in sides.items():
        isec = b_box_fr.select_by_intersect(side_tool)
        box_all.append(isec.modify_regions("." + name))
    box_all.extend([box_fr, b_left_r, b_right_r])

    b_fractures = factory.group(*fractures_fr.get_boundary_per_region())
    b_fractures_box = b_fractures.select_by_intersect(b_box).modify_regions("{}_box")
    b_fr_left_well = b_fractures.select_by_intersect(b_left_well).modify_regions("{}_left_well")
    b_fr_right_well = b_fractures.select_by_intersect(b_right_well).modify_regions("{}_right_well")
    b_fractures = factory.group(b_fr_left_well, b_fr_right_well, b_fractures_box)
    mesh_groups = [*box_all, fractures_fr, b_fractures]

    factory.keep_only(*mesh_groups)
    factory.remove_duplicate_entities()
    factory.write_brep()

    min_el_size = well_r / 20
    fracture_el_size = np.max(dimensions) / 20
    max_el_size = np.max(dimensions) / 10

    fractures_fr.set_mesh_step(200)
    # fracture_el_size = field.constant(100, 10000)
    # frac_el_size_only = field.restrict(fracture_el_size, fractures_fr, add_boundary=True)
    # field.set_mesh_step_field(frac_el_size_only)

    mesh = options.Mesh()
    mesh.ToleranceInitialDelaunay = 0.0001
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 1
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCurvePoints = 12

    factory.make_mesh(mesh_groups)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)

    factory.show()


def compute_hm(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    substitute_placeholders('01_hm_tmpl.yaml', '01_hm.yaml', config_dict['hm_params'])
    arguments = config_dict['flow_executable'].copy()
    arguments.extend(['--output_dir', 'output_hm', '01_hm.yaml'])
    print("Running: ", " ".join(arguments))
    subprocess.call(arguments)

def prepare_th_input(config_dict):
    """
    Prepare FieldFE input file for the TH simulation.
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    pass

def compute_th(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    substitute_placeholders('02_th_tmpl.yaml', '02_th.yaml', config_dict['th_params'])
    arguments = config_dict['flow_executable'].copy()
    arguments.extend(['--output_dir', 'output_th', '02_th.yaml'])
    print("Running: ", " ".join(arguments))
    subprocess.call(arguments)

def get_result_description():
    """
    :return:
    """
    end_time = 30
    values = [ [ValueDesctription(time=t, position="extraction_well", quantity="power", unit="MW"),
                ValueDesctription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
                ] for t in np.linspace(0, end_time, 0.1)]
    power_series, temp_series = zip(*values)
    return power_series + temp_series


def extract_time_series(yaml_stream, regions, extract):
    """

    :param yaml_stream:
    :param regions:
    :return: times list, list: for every region the array of value series
    """
    data = yaml.safe_load(yaml_stream)['data']
    times = set()
    reg_series = {reg: [] for reg in regions}

    for time_data in data:
        region = time_data['region']
        if region in reg_series:
            times.add(time_data['time'])
            power_in_time = extract(time_data)
            reg_series[region].append(power_in_time)
    times = list(times)
    times.sort()
    series = [np.array(region_series) for region_series in reg_series.values()]
    return np.array(times), series

def extract_results(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    : return
    """
    abs_zero_temp = 273.15
    year_sec = 60 * 60 * 24 *365
    bc_regions = ['.left_fr_left_well', '.left_well', '.right_fr_right_well', '.right_well']
    out_regions = bc_regions[2:]
    with open("output_th/energy_balance.yaml", "r") as f:
        power_times, reg_powers = extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
    power_series = -sum(reg_powers)

    with open("output_th/Heat_AdvectionDiffusion_region_stat.yaml", "r") as f:
        temp_times, reg_temps = extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
    with open("output_th/water_balance.yaml", "r") as f:
        flux_times, reg_fluxes = extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
    sum_flux = sum(reg_fluxes)
    avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux

    fig, ax1 = plt.subplots()
    temp_color = 'red'
    ax1.set_xlabel('time [y]')
    ax1.set_ylabel('Temperature [C deg]', color=temp_color)
    ax1.plot(temp_times[1:] / year_sec, avg_temp[1:] - abs_zero_temp, color=temp_color)
    ax1.tick_params(axis='y', labelcolor=temp_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    pow_color = 'blue'
    ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
    ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
    ax2.tick_params(axis='y', labelcolor=pow_color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()



def sample(tag, config_dict):
    # Setup dir
    root_dir = os.getcwd()
    sample_dir = os.path.abspath("samples/{}".format(tag))
    try:
        shutil.rmtree(sample_dir)
    except FileNotFoundError:
        pass
    os.mkdir(sample_dir)
    os.chdir(sample_dir)

    prepare_mesh(config_dict)
    #compute_hm(config_dict)
    #prepare_th_input(config_dict)
    #compute_th(config_dict)
    #extract_results(config_dict)

    #os.chdir(root_dir)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    sample(0, config_dict)