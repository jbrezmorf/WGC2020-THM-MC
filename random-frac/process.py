import sys
sys.path.append('../dfn/src')
sys.path.append('../MLMC/src')

import os
import shutil
import subprocess
import yaml
import attr
import numpy as np
import collections
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


def prepare_mesh(config_dict, fractures):
    geom = config_dict["geometry"]
    dimensions = geom["box_dimensions"]
    well_z0, well_z1 = geom["well_openning"]
    well_length = well_z1 - well_z0
    well_r = geom["well_effective_radius"]
    well_dist = geom["well_distance"]
    mesh_name = config_dict["mesh_name"]
    mesh_file = mesh_name + ".msh"
    fracture_mesh_step = 20

    if os.path.isfile(mesh_file):
        return mesh_file


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
    fractures_group = fractures_group.remove_small_mass(fracture_mesh_step * fracture_mesh_step / 10)

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
    #fractures_fr.set_mesh_step(fracture_mesh_step)

    factory.keep_only(*mesh_groups)
    factory.remove_duplicate_entities()
    factory.write_brep()

    min_el_size = fracture_mesh_step / 100
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

    healed_mesh = heal_mesh(mesh_file)
    #factory.show()
    return healed_mesh


def tri_measure(nodes):
    return np.linalg.norm(np.cross(nodes[2] - nodes[0], nodes[1] - nodes[0]))/2


def smooth_grad_error_indicator_2d(nodes):
    edges = [(0,1), (1,2), (2,0)]
    e_lens = [np.linalg.norm(nodes[i]- nodes[j]) for i,j in edges]
    i_min_edge = np.argmin(e_lens)
    prod = max(1e-300, (np.prod(e_lens)) ** (2.0 / 3.0))
    quality = 4 / np.sqrt(3) * np.abs(tri_measure(nodes)) / prod
    return quality, edges[i_min_edge]


def tet_measure(nodes):
    return np.linalg.det(nodes[1:, :] - nodes[0, :]) / 6


def smooth_grad_error_indicator_3d(nodes):
    vtxs_faces = [[0,1,2], [0,1,3], [0, 2,3], [1,2,3]]
    faces = [tri_measure(nodes[face_vtxs]) for face_vtxs in vtxs_faces]
    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    e_lens = [np.linalg.norm(nodes[i]- nodes[j]) for i,j in edges]
    e_faces = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
    sum_pairs = max(1e-300, np.sum([faces[i] * faces[j] * elen ** 2 for (i,j),elen in zip(e_faces, e_lens)]))
    regular = (2.0 * np.sqrt(2.0 / 3.0) / 9.0)
    quality = np.abs(tet_measure(nodes)) * (np.sum(faces) / sum_pairs) ** (3.0/4.0) / regular
    i_min_edge = np.argmin(e_lens)
    return quality, edges[i_min_edge]


def check_element(mesh, eid):
    """
    Check element, possibly contract its shortest edge.
    :param mesh:
    :param eid:
    :return: List of still existing but changed elements.
    """
    quality_tol = 0.001
    type, tags, node_ids = mesh.elements[eid]
    nodes = np.array([mesh.nodes[n] for n in node_ids])
    if len(nodes) < 3:
        return []
    merge_nodes = None
    if len(nodes) == 4:
        quality, min_edge_nodes = smooth_grad_error_indicator_3d(nodes)
        #print(eid, "3dq: ", quality)
        if quality < quality_tol:
            loc_node_a, loc_node_b = min_edge_nodes
            merge_nodes = (node_ids[loc_node_a], node_ids[loc_node_b])
    elif len(nodes) == 3:
        quality, min_edge_nodes = smooth_grad_error_indicator_2d(nodes)
        #print(eid, "2dq: ", quality)
        if quality < quality_tol:
            loc_node_a, loc_node_b = min_edge_nodes
            merge_nodes = (node_ids[loc_node_a], node_ids[loc_node_b])

    if merge_nodes is None:
        return []

    print("eid: {} q{}d: {} nodes: {}".format(eid, len(nodes) -1, quality, merge_nodes))
    node_a, node_b = merge_nodes
    els_a = set(mesh.node_els[node_a])
    els_b = set(mesh.node_els[node_b])
    # remove elements containing the edge (a,b)
    for i_edge_el in  els_a & els_b:
        if i_edge_el in mesh.elements:
            del mesh.elements[i_edge_el]
    # substitute node a for the node b
    for i_b_el in els_b:
        if i_b_el in mesh.elements:
            type, tags, node_ids = mesh.elements[i_b_el]
            node_ids = [node_a if n == node_b else n for n in node_ids]
            mesh.elements[i_b_el] = (type, tags, node_ids)
    # average nodes, remove the second one
    node_avg = np.average([np.array(mesh.nodes[n]) for n in merge_nodes], axis=0)
    #print("node avg: ", node_avg)
    mesh.nodes[node_a] = node_avg
    del mesh.nodes[node_b]

    # merge node element lists
    mesh.node_els[node_a] = list(els_a | els_b)
    del mesh.node_els[node_b]

    return mesh.node_els[node_a]

def heal_mesh(mesh_file):
    """
    Detect elements with bad quality according to the flow123d quality measure (poor one).
    Contract their shortest edge.
    - should only be used for elements with an edge shorter then others
    - not effective for thetrahedra between two close skew lines
      TODO: move two of vertices to get intersection and split the neigbouring thtrahedra
    TODO: use simpler quality measure
    Write healed mesh to the new file.
    :param mesh_file:
    :return: Name of the healed mesh file.
    """

    import gmsh_io
    mesh = gmsh_io.GmshIO(mesh_file)

    # make node -> element map
    mesh.node_els = collections.defaultdict(list)
    for eid, e in mesh.elements.items():
        type, tags, node_ids = e
        for n in node_ids:
            mesh.node_els[n].append(eid)
    #print("node els:", node_els[682])

    el_to_check = collections.deque(mesh.elements.keys())
    while el_to_check:
        eid = el_to_check.popleft()
        if eid in mesh.elements:
            el_to_check.extend(check_element(mesh, eid))


    base, ext = os.path.splitext(mesh_file)
    healed_name = base + "_healed.msh"
    with open(healed_name, "w") as f:
        mesh.write_ascii(f)
    return healed_name

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
        status = completed.returncode == 0
    print("Exit status: ", status)
    return status



def prepare_th_input(config_dict):
    """
    Prepare FieldFE input file for the TH simulation.
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    pass


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
    series = [np.array(region_series) for region_series in reg_series.values()]
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


def setup_dir(tag, config_dict, clean=False):
    sample_dir = os.path.abspath("samples/{}".format(tag))
    if clean:
        try:
           shutil.rmtree(sample_dir)
        except FileNotFoundError:
           pass
        os.mkdir(sample_dir)
    os.chdir(sample_dir)
    for f in config_dict["copy_files"]:
        shutil.copyfile("../../" + f, f)
    flow_exec = config_dict["flow_executable"].copy()
    flow_exec[0] = "../../" + flow_exec[0]
    config_dict["_aux_flow_path"] = flow_exec


def sample(tag, config_dict):
    root_dir = os.getcwd()
    #setup_dir(tag, config_dict, clean=True)
    setup_dir(tag, config_dict)

    fractures = generate_fractures(config_dict)
    # plot_fr_orientation(fractures)
    healed_mesh = prepare_mesh(config_dict, fractures)
    config_dict["hm_params"]["mesh"] = healed_mesh
    config_dict["th_params"]["mesh"] = healed_mesh

    hm_succeed = call_flow(config_dict, 'hm_params', result_files=["mechanics.msh"])
    th_succeed = False
    if hm_succeed:
        prepare_th_input(config_dict)
        th_succeed = call_flow(config_dict, 'th_params', result_files=["energy_balance.yaml"])
        if th_succeed:
            series = extract_results(config_dict)
            plot_exchanger_evolution(*series)
    os.chdir(root_dir)


if __name__ == "__main__":
    np.random.seed(1)
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    sample(0, config_dict)