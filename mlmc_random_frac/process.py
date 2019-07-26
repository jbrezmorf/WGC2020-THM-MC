import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../MLMC/src'))
sys.path.append(os.path.join(script_dir, '../dfn/src'))

import shutil
import subprocess
import yaml
import attr
import numpy as np
import collections
import gmsh_io
# import matplotlib.pyplot as plt

import fracture

# TODO:
# - enforce creation of empty physical groups, or creation of empty regions in the flow input
# - speedup mechanics

@attr.s(auto_attribs=True)
class ValueDescription:
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

def to_polar(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if z > 0:
        phi += np.pi
    return (phi, rho)

def plot_fr_orientation(fractures):
    family_dict = collections.defaultdict(list)
    for fr in fractures:
        x, y, z = fracture.FisherOrientation.rotate(np.array([0,0,1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
        family_dict[fr.region].append([
            to_polar(z, y, x),
            to_polar(z, x, -y),
            to_polar(y, x, z)
            ])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    for name, data in family_dict.items():
        # data shape = (N, 3, 2)
        data = np.array(data)
        for i, ax in enumerate(axes):
            phi = data[:, i, 0]
            r = data[:, i, 1]
            c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
    axes[0].set_title("X-view, Z-north")
    axes[1].set_title("Y-view, Z-north")
    axes[2].set_title("Z-view, Y-north")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    fig.legend(loc = 1)
    fig.savefig("fracture_orientation.pdf")
    plt.close(fig)
    #plt.show()

def generate_fractures(config_dict):
    geom = config_dict["geometry"]
    dimensions = geom["box_dimensions"]
    well_z0, well_z1 = geom["well_openning"]
    well_length = well_z1 - well_z0
    well_r = geom["well_effective_radius"]
    well_dist = geom["well_distance"]

    # generate fracture set
    fracture_box = [1.5 * well_dist, 1.5 * well_length, 1.5 * well_length]
    volume = np.product(fracture_box)
    pop = fracture.Population(volume)
    pop.initialize(geom["fracture_stats"])
    pop.set_sample_range([1, well_dist], max_sample_size=geom["n_frac_limit"])
    print("total mean size: ", pop.mean_size())
    pos_gen = fracture.UniformBoxPosition(fracture_box)
    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    #fracture.fr_intersect(fractures)

    for fr in fractures:
        fr.region = "fr"
    used_families = set((f.region for f in fractures))
    for model in ["hm_params", "th_params"]:
        model_dict = config_dict[model]
        model_dict["fracture_regions"] = list(used_families)
        model_dict["left_well_fracture_regions"] = [".{}_left_well".format(f) for f in used_families]
        model_dict["right_well_fracture_regions"] = [".{}_right_well".format(f) for f in used_families]
    return fractures


def create_fractures_rectangles(gmsh_geom, fractures, base_shape: 'ObjectSet'):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.centre) \
            .set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def create_fractures_polygons(gmsh_geom, fractures):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    frac_obj = fracture.Fractures(fractures)
    frac_obj.snap_vertices_and_edges()
    shapes = []
    for fr, square in zip(fractures, frac_obj.squares):
        shape = gmsh_geom.make_polygon(square).set_region(fr.region)
        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def make_mesh(config_dict, fractures, mesh_name, mesh_file):
    fracture_mesh_step = 10
    geom = config_dict["geometry"]
    dimensions = geom["box_dimensions"]
    well_z0, well_z1 = geom["well_openning"]
    well_length = well_z1 - well_z0
    well_r = geom["well_effective_radius"]
    well_dist = geom["well_distance"]

    from gmsh_api import gmsh
    from gmsh_api import options
    factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
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
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
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
    fractures = create_fractures_rectangles(factory, fractures, factory.rectangle())
    #fractures = create_fractures_polygons(factory, fractures)
    fractures_group = factory.group(*fractures)
    #fractures_group = fractures_group.remove_small_mass(fracture_mesh_step * fracture_mesh_step / 10)

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
    fractures_fr.set_mesh_step(fracture_mesh_step)

    factory.keep_only(*mesh_groups)
    factory.remove_duplicate_entities()
    factory.write_brep()

    min_el_size = fracture_mesh_step / 10
    fracture_el_size = np.max(dimensions) / 20
    max_el_size = np.max(dimensions) / 8


    # fracture_el_size = field.constant(100, 10000)
    # frac_el_size_only = field.restrict(fracture_el_size, fractures_fr, add_boundary=True)
    # field.set_mesh_step_field(frac_el_size_only)

    mesh = options.Mesh()
    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    #mesh.Algorithm = options.Algorithm2d.Delaunay
    #mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    #mesh.Algorithm3D = options.Algorithm3d.Frontal
    #mesh.Algorithm3D = options.Algorithm3d.Delaunay
    #mesh.ToleranceInitialDelaunay = 0.01
    #mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 1
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2


    #factory.make_mesh(mesh_groups, dim=2)
    factory.make_mesh(mesh_groups)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(mesh_name + ".msh2", mesh_file)


def prepare_mesh(config_dict, fractures):
    mesh_name = config_dict["mesh_name"]
    mesh_file = mesh_name + ".msh"
    if not os.path.isfile(mesh_file):
        make_mesh(config_dict, fractures, mesh_name, mesh_file)

    mesh_healed = mesh_name + "_healed.msh"
    if not os.path.isfile(mesh_healed):
        import heal_mesh
        hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
        hm.heal_mesh(tol_edge_ratio=0.02, tol_flat_ratio=0.02)
        hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
        hm.write()
        assert hm.healed_mesh_name == mesh_healed
    return mesh_healed

def check_conv_reasons(log_fname):
    with open(log_fname, "r") as f:
        for line in f:
            tokens = line.split(" ")
            try:
                i = tokens.index('convergence')
                if tokens[i + 1] == 'reason':
                    value = tokens[i + 2].rstrip(",")
                    conv_reason = int(value)
                    if conv_reason < 0:
                        print("Failed to converge: ", conv_reason)
                        return False
            except ValueError:
                continue
    return True

def call_flow(config_dict, param_key, result_files):
    """
    Redirect sstdout and sterr, return true on succesfull run.
    :param arguments:
    :return:
    """

    params = config_dict[param_key]
    fname = params["in_file"]
    substitute_placeholders(fname + '_tmpl.yaml', fname + '.yaml', params)
    arguments = config_dict["_aux_flow_path"].copy()
    output_dir = "output_" + fname
    config_dict[param_key]["output_dir"] = output_dir
    if all([os.path.isfile(os.path.join(output_dir, f)) for f in result_files]):
        status = True
    else:
        arguments.extend(['--output_dir', output_dir, fname + ".yaml"])
        print("Running: ", " ".join(arguments))
        with open(fname + "_stdout", "w") as stdout:
            with open(fname + "_stderr", "w") as stderr:
                completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
        print("Exit status: ", completed.returncode)
        status = completed.returncode == 0
    conv_check = check_conv_reasons(os.path.join(output_dir, "flow123.0.log"))
    return status # and conv_check



def prepare_th_input(config_dict):
    """
    Prepare FieldFE input file for the TH simulation.
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    # pass
    # we have to read region names from the input mesh
    # input_mesh = gmsh_io.GmshIO(config_dict['hm_params']['mesh'])
    #
    # is_bc_region = {}
    # for name, (id, _) in input_mesh.physical.items():
    #     unquoted_name = name.strip("\"'")
    #     is_bc_region[id] = (unquoted_name[0] == '.')

    # read mesh and mechanichal output data
    mechanics_output = os.path.join(config_dict['hm_params']["output_dir"], 'mechanics.msh')
    mesh = gmsh_io.GmshIO(mechanics_output)

    n_bulk = len(mesh.elements)
    ele_ids = np.array(list(mesh.elements.keys()), dtype=float)

    init_fr_cs = float(config_dict['hm_params']['fr_cross_section'])
    init_fr_K = float(config_dict['hm_params']['fr_conductivity'])
    init_bulk_K = float(config_dict['hm_params']['bulk_conductivity'])
    min_fr_cross_section = float(config_dict['th_params']['min_fr_cross_section'])

    time_idx = 1
    time, field_cs = mesh.element_data['cross_section_updated'][time_idx]


    cs = np.maximum(np.array([v[0] for v in field_cs.values()]), min_fr_cross_section)

    K = np.where(
        cs == 1.0,      # condition
        init_bulk_K,    # true array
        init_fr_K * (cs / init_fr_cs) ** 2
    )

    # mesh.write_fields('output_hm/th_input.msh', ele_ids, {'conductivity': K})
    th_input_file = 'th_input.msh'
    with open(th_input_file, "w") as fout:
        mesh.write_ascii(fout)
        mesh.write_element_data(fout, ele_ids, 'conductivity', K[:, None])
        mesh.write_element_data(fout, ele_ids, 'cross_section_updated', cs[:, None])

    # create field for K (copy cs)
    # posun dat K do casu 0
    # read original K = oK (define in config yaml)
    # read original cs = ocs (define in config yaml)
    # compute K = oK * (cs/ocs)^2
    # write K

    # posun dat cs do casu 0
    # write cs

    # mesh.element_data.



def get_result_description():
    """
    :return:
    """
    end_time = 30
    values = [ [ValueDescription(time=t, position="extraction_well", quantity="power", unit="MW"),
                ValueDescription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
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
    series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
    return np.array(times), series


def extract_results(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    : return
    """
    bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
    out_regions = bc_regions[2:]
    output_dir = config_dict["th_params"]["output_dir"]
    with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as f:
        power_times, reg_powers = extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
        power_series = -sum(reg_powers)

    with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as f:
        temp_times, reg_temps = extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
    with open(os.path.join(output_dir, "water_balance.yaml"), "r") as f:
        flux_times, reg_fluxes = extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
    sum_flux = sum(reg_fluxes)
    avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
    print("temp: ", avg_temp)
    return temp_times, avg_temp, power_times, power_series


def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
    abs_zero_temp = 273.15
    year_sec = 60 * 60 * 24 * 365

    import matplotlib.pyplot as plt
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


def setup_dir(config_dict, clean=False):
    for f in config_dict["copy_files"]:
        shutil.copyfile(os.path.join(script_dir, f), os.path.join(".", f))
    flow_exec = config_dict["flow_executable"].copy()
    config_dict["_aux_flow_path"] = flow_exec


def sample(config_dict):

    setup_dir(config_dict, clean=True)
    #setup_dir(tag, config_dict)

    fractures = generate_fractures(config_dict)
    # plot_fr_orientation(fractures)
    healed_mesh = prepare_mesh(config_dict, fractures)
    healed_mesh_bn = os.path.basename(healed_mesh)
    config_dict["hm_params"]["mesh"] = healed_mesh_bn
    config_dict["th_params"]["mesh"] = healed_mesh_bn

    hm_succeed = call_flow(config_dict, 'hm_params', result_files=["mechanics.msh"])
    th_succeed = False
    if hm_succeed:
        prepare_th_input(config_dict)
        th_succeed = call_flow(config_dict, 'th_params', result_files=["energy_balance.yaml"])
        if th_succeed:
            series = extract_results(config_dict)
            plot_exchanger_evolution(*series)



if __name__ == "__main__":
    sample_dir = sys.argv[1]
    os.chdir(sample_dir)
    np.random.seed()

    with open(os.path.join(script_dir, "config.yaml"), "r") as f:
        config_dict = yaml.safe_load(f)
    sample(config_dict)
