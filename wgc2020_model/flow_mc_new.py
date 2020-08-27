import os
import subprocess
import numpy as np
import itertools
import collections
import shutil
import yaml
from typing import List

from bgem.gmsh import gmsh
from bgem.gmsh import gmsh_io
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import field as gmsh_field
from bgem.gmsh import heal_mesh

from mlmc.level_simulation import LevelSimulation
from mlmc.sim.simulation import Simulation
from mlmc.sim.simulation import QuantitySpec

import fracture
import matplotlib.pyplot as plt

def force_mkdir(path, force=False):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recursively if it already exists.
    :param path: path to directory
    :param force: if dir already exists then remove it and create new one
    :return: None
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)

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

class Flow123d_WGC2020(Simulation):

    zero_temperature_offset = 273.15

    def __init__(self, config, clean):
        super(Flow123d_WGC2020, self).__init__(config)

        # TODO: how should I know, that these variables must be set here ?
        self.need_workspace = True
        self.work_dir = config["work_dir"]
        self.clean = clean

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float])-> LevelSimulation:
        """
        #TODO: I do not understand the parameters (what should I do for my random fractures test case ?)

        Overrides Simulation.level_instance
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.)
        :param fine_level_params: in this version, it is just fine simulation step
        :param coarse_level_params: in this version, it is just coarse simulation step
        :return: mlmc.LevelSimulation object, this object is serialized in SamplingPoolPbs and deserialized in PbsJob,
         so it allows pass simulation data from main process to PBS process
        """

        config = self._config.copy()

        # Set fine simulation common files directory
        # Files in the directory are used by each simulation at that level
        common_files_dir = os.path.join(self.work_dir, "common_files")
        force_mkdir(common_files_dir, force=self.clean)
        config["common_files_dir"] = common_files_dir

        # copy common files
        for f in config["copy_files"]:
            shutil.copyfile(os.path.join(config["script_dir"], f), os.path.join(common_files_dir, f))


        #TODO: what is the role of task_size
        # in one-level method it does not matter [by JB]
        return LevelSimulation(config_dict=config,
                               # task_size=len(fine_mesh_data['points']),
                               task_size=config["task_size"],
                               calculate=Flow123d_WGC2020.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True  # If True, a sample directory is created
                               )

    @staticmethod
    def calculate(config_dict, seed):
        """
        TODO: what is sample_workspace? where is calculate called? Where is the seed in orig flow_mc.py ?
        Overrides Simulation.calculate. The program is currently in <work_dir>/<sample_id_dir> directory.
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config_dict: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        # does all the data preparation, passing
        # running simulation
        # extracting results

        print("=========================== RUNNING CALCULATION ===========================")

        # Set random seed, seed is calculated from sample id, so it is not user defined
        np.random.seed(seed)

        # collect only
        if config_dict["collect_only"]:
            return Flow123d_WGC2020.collect_results(config_dict)

        mesh_repo = config_dict.get('mesh_repository', None)
        # p = [1,2,3]
        # a = p[5]

        print("Creating mesh...")
        if mesh_repo:
            healed_mesh = Flow123d_WGC2020.sample_mesh_repository(mesh_repo)
        else:
            fractures = Flow123d_WGC2020.generate_fractures(config_dict)
            # fractures = Flow123d_WGC2020.generate_fixed_fractures()
            Flow123d_WGC2020.plot_fr_orientation(fractures)
            healed_mesh = Flow123d_WGC2020.prepare_mesh(config_dict, fractures)

        Flow123d_WGC2020.read_physical_names(config_dict, healed_mesh)
        print("Creating mesh...finished")

        if config_dict["mesh_only"]:
            return Flow123d_WGC2020.empty_result()

        print("Running Flow123d - HM...")
        hm_succeed = Flow123d_WGC2020.call_flow(config_dict, 'hm_params', result_files=["mechanics.msh"])
        if not hm_succeed:
            raise Exception("HM model failed.")
        print("Running Flow123d - HM...finished")

        print("Running Flow123d - TH_ref...")
        th_succeed = Flow123d_WGC2020.call_flow(config_dict, 'th_params_ref', result_files=["energy_balance.yaml"])
        if not th_succeed:
            raise Exception("TH reference model failed.")
        print("Running Flow123d - TH_ref...finished")

        print("Preparing TH input...")
        Flow123d_WGC2020.prepare_th_input(config_dict)
        print("Preparing TH input...finished")
        print("Running Flow123d - TM...")
        th_succeed = Flow123d_WGC2020.call_flow(config_dict, 'th_params', result_files=["energy_balance.yaml"])
        if not th_succeed:
            raise Exception("TH model failed.")
        print("Running Flow123d - TH...finished")

        print("Finished computation")

        return Flow123d_WGC2020.collect_results(config_dict)

    @staticmethod
    def check_data(data, minimum, maximum):
        n_times = len(Flow123d_WGC2020.result_format()[0].times)
        if len(data) != n_times:
            raise Exception("Data not corresponding with time axis.")

        if np.isnan(np.sum(data)):
            raise Exception("NaN present in extracted data.")

        min = np.amin(data)
        if min < minimum:
            raise Exception("Data out of given range [min].")
        max = np.amax(data)
        if max > maximum:
            raise Exception("Data out of given range [max].")

    @staticmethod
    def collect_results(config_dict):
        config_dict["th_params"]["output_dir"] = "output_" + config_dict["th_params"]["in_file"]
        config_dict["th_params_ref"]["output_dir"] = "output_" + config_dict["th_params_ref"]["in_file"]
        filenames = ["energy_balance.yaml",
                     "water_balance.yaml",
                     "Heat_AdvectionDiffusion_region_stat.yaml"]
        result_files = list()
        result_files.extend([os.path.join(config_dict["th_params"]["output_dir"], f) for f in filenames])
        result_files.extend([os.path.join(config_dict["th_params_ref"]["output_dir"], f) for f in filenames])

        if all([os.path.isfile(f) for f in result_files]):
            print("Extracting results...")
            series = Flow123d_WGC2020.extract_results(config_dict)
            # Flow123d_WGC2020.plot_exchanger_evolution(*series)
            print("Extracting results...finished")
            (avg_temp, power), (avg_temp_ref, power_ref) = series

            Flow123d_WGC2020.check_data(avg_temp, config_dict["extract"]["temp_min"], config_dict["extract"]["temp_max"])
            Flow123d_WGC2020.check_data(power, config_dict["extract"]["power_min"], config_dict["extract"]["power_max"])
            Flow123d_WGC2020.check_data(avg_temp_ref, config_dict["extract"]["temp_min"], config_dict["extract"]["temp_max"])
            Flow123d_WGC2020.check_data(power_ref, config_dict["extract"]["power_min"], config_dict["extract"]["power_max"])

            # [fine, coarse] -> [fine_vector, fine_vector]
            return [[*avg_temp, *power, *avg_temp_ref, *power_ref],
                    [*avg_temp, *power, *avg_temp_ref, *power_ref]]
        else:
            raise Exception("Not all result files present.")

    @staticmethod
    def result_format()-> List[QuantitySpec]:
        """
        Overrides Simulation.result_format
        :return:
        """
        # create simple instance of QuantitySpec for each quantity we want to collect
        # the time vector of the data must be specified here!

        # TODO: define times according to output times of Flow123d
        # TODO: how should be units defined (and other members)?
        step = 1
        end_time = 31
        times = list(range(0, end_time, step))
        spec1 = QuantitySpec(name="avg_temp", unit="C", shape=(1, 1), times=times, locations=['.well'])
        spec2 = QuantitySpec(name="power", unit="J", shape=(1, 1), times=times, locations=['.well'])
        spec3 = QuantitySpec(name="avg_temp_ref", unit="C", shape=(1, 1), times=times, locations=['.well'])
        spec4 = QuantitySpec(name="power_ref", unit="J", shape=(1, 1), times=times, locations=['.well'])
        return [spec1, spec2, spec3, spec4]

    @staticmethod
    def empty_result():
        return [[np.random.normal()], [np.random.normal()]]




    @staticmethod
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

    @staticmethod
    def call_flow(config_dict, param_key, result_files):
        """
        Redirect sstdout and sterr, return true on succesfull run.
        :param arguments:
        :return:
        """

        params = config_dict[param_key]
        fname = params["in_file"]
        arguments = config_dict["_aux_flow_path"].copy()
        output_dir = "output_" + fname
        config_dict[param_key]["output_dir"] = output_dir
        if all([os.path.isfile(os.path.join(output_dir, f)) for f in result_files]):
            status = True
        else:
            substitute_placeholders(os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml'),
                                    fname + '.yaml',
                                    params)
            arguments.extend(['--no_profiler', '--output_dir', output_dir, fname + ".yaml"])
            print("Running: ", " ".join(arguments))
            with open(fname + "_stdout", "w") as stdout:
                with open(fname + "_stderr", "w") as stderr:
                    completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        conv_check = Flow123d_WGC2020.check_conv_reasons(os.path.join(output_dir, "flow123.0.log"))
        print("converged: ", conv_check)
        return status and conv_check










    @staticmethod
    def sample_mesh_repository(mesh_repository):
        """
        Select random mesh from the given mesh repository.
        """
        mesh_file = np.random.choice(os.listdir(mesh_repository))
        healed_mesh = "random_fractures_healed.msh"
        shutil.copyfile(os.path.join(mesh_repository, mesh_file), healed_mesh)
        # heal_ref_report = {'flow_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []},
        #                    'gamma_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []}}
        # with open("random_fractures_heal_stats.yaml", "w") as f:
        #     yaml.dump(heal_ref_report, f)
        return healed_mesh

    @staticmethod
    def read_physical_names(config_dict, mesh_file):
        """
        Read physical names to set fracture B.C. regions in Flow123d yaml later.
        :param config_dict:
        :param mesh_file:
        :return:
        """
        mesh_bn = os.path.basename(mesh_file)
        reader = gmsh_io.GmshIO()
        reader.filename = mesh_file
        regions = [*reader.read_physical_names()]

        # fracture regions
        reg_fr = [reg for reg in regions if reg.startswith("fr")]

        # split boundary fracture regions on left and right well
        reg_fr_left_well = [reg for reg in regions if ".fr" in reg and "left_well" in reg]
        reg_fr_right_well = [reg for reg in regions if ".fr" in reg and "right_well" in reg]
        for model in ["hm_params", "th_params", "th_params_ref"]:
            model_dict = config_dict[model]
            model_dict["mesh"] = mesh_bn
            model_dict["fracture_regions"] = reg_fr
            model_dict["left_well_fracture_regions"] = reg_fr_left_well
            model_dict["right_well_fracture_regions"] = reg_fr_right_well

    @staticmethod
    def create_fractures_shapes(gmsh_geom, fractures, base_shape: 'ObjectSet', max_mesh_step = 0):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = []
        for i, fr in enumerate(fractures):
            shape = base_shape.copy()
            fr_mesh_step = np.min([fr.rx, fr.ry]) / 1.25
            if max_mesh_step != 0:
                fr_mesh_step = np.min([fr_mesh_step, max_mesh_step])
            # print("fr: ", i, "tag: ", shape.dim_tags)
            if i % 50 == 0:
                print(i, " fracture shapes generated... ")
            shape = shape.scale([fr.rx, fr.ry, 1]) \
                .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
                .translate(fr.centre) \
                .set_region(fr.region) \
                .mesh_step(fr_mesh_step)
            shapes.append(shape)

        fracture_fragments = gmsh_geom.fragment(*shapes)
        return fracture_fragments

    @staticmethod
    def generate_fixed_fractures():
        fractures = []
        f1 = fracture.FractureShape(
            r=140,
            centre=np.array([-100, -5, 15]),
            rotation_axis=np.array([-0.17, -0.98, 0]),
            rotation_angle=0.3,
            shape_angle=4.2,
            aspect=1,
            region='fr_1')
        fractures.append(f1)

        f2 = fracture.FractureShape(
            r=140,
            centre=np.array([100, 5, -5]),
            rotation_axis=np.array([0.07, -0.99, 0]),
            rotation_angle=2.48,
            shape_angle=0.6,
            aspect=1,
            region='fr_2')
        fractures.append(f2)

        f3 = fracture.FractureShape(
            r=65,
            centre=np.array([0, 0, 120]),
            rotation_axis=np.array([0.5, 0.5, 0]),
            rotation_angle=1.31,
            shape_angle=1.56,
            aspect=1,
            region='fr_3')
        fractures.append(f3)

        f4 = fracture.FractureShape(
            r=65,
            centre=np.array([0, 0, -95]),
            rotation_axis=np.array([0.5, -0.5, 0]),
            rotation_angle=1.31,
            shape_angle=1.56,
            aspect=1,
            region='fr_4')
        fractures.append(f4)

        f5 = fracture.FractureShape(
            r=40,
            centre=np.array([0, 0, 10]),
            rotation_axis=np.array([0.5, -0.5, 0]),
            rotation_angle=0.1,
            shape_angle=0.1,
            aspect=1,
            region='fr_5')
        fractures.append(f5)

        return fractures
    @staticmethod
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
        connected_position = geom.get('connected_position_distr', False)
        if connected_position:
            eps = well_r / 2
            left_well_box = [-well_dist / 2 - eps, -eps, well_z0, -well_dist / 2 + eps, +eps, well_z1]
            right_well_box = [well_dist / 2 - eps, -eps, well_z0, well_dist / 2 + eps, +eps, well_z1]
            pos_gen = fracture.ConnectedPosition(
                confining_box=fracture_box,
                init_boxes=[left_well_box, right_well_box])
        else:
            pos_gen = fracture.UniformBoxPosition(fracture_box)
        fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
        # fracture.fr_intersect(fractures)

        for i, fr in enumerate(fractures):
            fr_name = "fr_" + str(i)
            fr.region = fr_name
        return fractures

    @staticmethod
    def to_polar(x, y, z):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        if z > 0:
            phi += np.pi
        return phi, rho

    @staticmethod
    def plot_fr_orientation(fractures):
        family_dict = collections.defaultdict(list)
        for fr in fractures:
            x, y, z = \
                fracture.FisherOrientation.rotate(np.array([0, 0, 1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
            family_dict[fr.region].append([
                Flow123d_WGC2020.to_polar(z, y, x),
                Flow123d_WGC2020.to_polar(z, x, -y),
                Flow123d_WGC2020.to_polar(y, x, z)
            ])

        fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
        rmax = 0
        for name, data in family_dict.items():
            # data shape = (N, 3, 2)
            data = np.array(data)
            for i, ax in enumerate(axes):
                phi = data[:, i, 0]
                r = data[:, i, 1]
                rmax = r.max()
                c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
        subtitle_pad = 1.3
        axes[0].set_title("X-view, Z-north", y=subtitle_pad)
        axes[1].set_title("Y-view, Z-north", y=subtitle_pad)
        axes[2].set_title("Z-view, Y-north", y=subtitle_pad)
        rticks = np.arange(rmax/4, rmax, rmax/4)
        rtickslables = [round(num, 2) for num in rticks]
        for ax in axes:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1)
            ax.set_rlabel_position(5)
            ax.set_rticks(rticks)
            ax.set_yticklabels(rtickslables, fontsize=6)

        # fig.legend(loc=1)
        plt.tight_layout()
        fig.savefig("fracture_orientation.pdf")
        plt.close(fig)
        # plt.show()

    @staticmethod
    def prepare_mesh(config_dict, fractures):
        mesh_name = config_dict["mesh_name"]
        mesh_file = mesh_name + ".msh"
        if not os.path.isfile(mesh_file):
            Flow123d_WGC2020.make_mesh(config_dict, fractures, mesh_name, mesh_file)

        mesh_healed = mesh_name + "_healed.msh"
        if not os.path.isfile(mesh_healed):
            hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
            hm.heal_mesh(gamma_tol=0.01)
            hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
            hm.write()
            assert hm.healed_mesh_name == mesh_healed
        return mesh_healed

    @staticmethod
    def make_mesh(config_dict, fractures, mesh_name, mesh_file):
        geom = config_dict["geometry"]
        fracture_mesh_step = geom['fracture_mesh_step']
        dimensions = geom["box_dimensions"]
        well_z0, well_z1 = geom["well_openning"]
        well_length = well_z1 - well_z0
        well_r = geom["well_effective_radius"]
        well_dist = geom["well_distance"]
        print("n fractures:", len(fractures))

        print("load gmsh api")
        factory = gmsh.GeometryOCC(mesh_name, verbose=True)
        gmsh_logger = factory.get_logger()
        gmsh_logger.start()
        gopt = gmsh_options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001
        # gopt.MatchMeshTolerance = 1e-1
        gopt.OCCFixSmallEdges = True
        gopt.OCCFixSmallFaces = True

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
        left_center = [-well_dist / 2, 0, 0]
        right_center = [+well_dist / 2, 0, 0]
        left_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
            .translate([0, 0, well_z0]).translate(left_center)
        right_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
            .translate([0, 0, well_z0]).translate(right_center)
        b_right_well = right_well.get_boundary()
        b_left_well = left_well.get_boundary()

        print("generating fracture shapes...")
        # fractures_shapes = \
        #     Flow123d_WGC2020.create_fractures_shapes(factory, fractures, factory.rectangle(), fracture_mesh_step)
        fractures_shapes = \
            Flow123d_WGC2020.create_fractures_shapes(factory, fractures, factory.disc(rx=0.5, ry=0.5), fracture_mesh_step)
        # fractures_tool = [fr.copy() for fr in fractures_shapes]
        fractures_shapes_group = factory.group(*fractures_shapes)
        # fractures_group = fractures_group.remove_small_mass(fracture_mesh_step * fracture_mesh_step / 10)

        print("cutting and fragmenting...")
        # drilled box and its boundary
        box_drilled = box.cut(left_well, right_well)

        # fractures, fragmented, fractures boundary
        print("cut fractures by box without wells...")
        fractures_shapes_group = fractures_shapes_group.intersect(box_drilled.copy())
        print("fragment fractures...")
        box_fr, fractures_fr = factory.fragment(box_drilled, fractures_shapes_group)

        # print("setting fractures mesh step...")
        # # set mesh size to fractures
        # # - trying out selecting fracture by fracture after fragmentation - very SLOW
        # # - solved by correctly passing mesh step in group/fragment/intersect functions
        # fractures_final = []
        # for fr in fractures_tool:
        #     frac = fractures_fr.select_by_intersect(fr)
        #     # suppose the regions on fracture dimtags are all the same (see create_fracture_shapes())
        #     rname = fr.regions[0].name
        #     assert rname in fr_mesh_step
        #     fr_step = np.min([fracture_mesh_step, fr_mesh_step[rname]])
        #     frac.mesh_step(fr_step)
        #     fractures_final.append(frac)
        # fractures_final_group = factory.group(*fractures_final)

        print("marking boundary regions...")
        b_box_fr = box_fr.get_boundary()
        b_left_r = b_box_fr.select_by_intersect(b_left_well).set_region(".left_well")
        b_right_r = b_box_fr.select_by_intersect(b_right_well).set_region(".right_well")

        box_all = []
        for name, side_tool in sides.items():
            isec = b_box_fr.select_by_intersect(side_tool)
            box_all.append(isec.modify_regions("." + name))
        box_all.extend([box_fr, b_left_r, b_right_r])

        # boundary of all fractures
        b_fractures = fractures_fr.get_boundary_per_region()
        b_fractures_group = factory.group(*b_fractures)
        # boundary of fractures on boundary of the box
        b_fractures_box = b_fractures_group.select_by_intersect(b_box).modify_regions("{}_box")
        # boundary of fractures on left well
        b_fr_left_well = b_fractures_group.select_by_intersect(b_left_well).modify_regions("{}_left_well")
        # boundary of fractures on right well
        b_fr_right_well = b_fractures_group.select_by_intersect(b_right_well).modify_regions("{}_right_well")
        b_fractures_group = factory.group(b_fr_left_well, b_fr_right_well, b_fractures_box)

        b_left_r.mesh_step(config_dict["geometry"]["well_effective_radius"] / 2)
        b_right_r.mesh_step(config_dict["geometry"]["well_effective_radius"] / 2)


        mesh_groups = [*box_all, fractures_fr, b_fractures_group]

        # print(fracture_mesh_step)
        # fractures_fr.mesh_step(fracture_mesh_step)
        # fracture_el_size = np.max(dimensions) / 20
        #
        # fracture_el_size = gmsh_field.constant(fracture_mesh_step, 10000)
        # frac_el_size_only = gmsh_field.restrict(fracture_el_size, fractures_fr, add_boundary=True)
        # gmsh_field.set_mesh_step_field(frac_el_size_only)

        print("meshing...")

        factory.keep_only(*mesh_groups)
        factory.remove_duplicate_entities()
        factory.write_brep()

        min_el_size = fracture_mesh_step / 10
        max_el_size = np.max(dimensions) / 8

        mesh = gmsh_options.Mesh()
        # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
        # mesh.Algorithm = options.Algorithm2d.Delaunay
        # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
        # mesh.Algorithm3D = options.Algorithm3d.Frontal
        # mesh.Algorithm3D = options.Algorithm3d.Delaunay

        # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
        mesh.Algorithm3D = gmsh_options.Algorithm3d.HXT

        mesh.ToleranceInitialDelaunay = 0.01
        # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
        mesh.CharacteristicLengthFromPoints = True
        mesh.CharacteristicLengthFromCurvature = True
        mesh.CharacteristicLengthExtendFromBoundary = 2
        mesh.CharacteristicLengthMin = min_el_size
        mesh.CharacteristicLengthMax = max_el_size
        mesh.MinimumCirclePoints = 6
        mesh.MinimumCurvePoints = 2

        # factory.make_mesh(mesh_groups, dim=2)
        factory.make_mesh(mesh_groups)

        gmsh_log_msgs = gmsh_logger.get()
        gmsh_logger.stop()
        Flow123d_WGC2020.check_gmsh_log(gmsh_log_msgs)

        factory.write_mesh(format=gmsh.MeshFormat.msh2)
        os.rename(mesh_name + ".msh2", mesh_file)
        # factory.show()

    @staticmethod
    def check_gmsh_log(lines):
        """
        Search for "No elements in volume" message -> could not mesh the volume -> empty mesh.
        # PLC Error:  A segment and a facet intersect at point (-119.217,65.5762,-40.8908).
        #   Segment: [70,2070] #-1 (243)
        #   Facet:   [3147,9829,13819] #482
        # Info    : failed to recover constrained lines/triangles
        # Info    : function failed
        # Info    : function failed
        # Error   : HXT 3D mesh failed
        # Error   : No elements in volume 1
        # Info    : Done meshing 3D (Wall 0.257168s, CPU 0.256s)
        # Info    : 13958 nodes 34061 elements
        # Error   : ------------------------------
        # Error   : Mesh generation error summary
        # Error   :     0 warnings
        # Error   :     2 errors
        # Error   : Check the full log for details
        # Error   : ------------------------------
        """
        empty_volume_error = "No elements in volume"
        res = [line for line in lines if empty_volume_error in line]
        if len(res) != 0:
            raise Exception("GMSH error - No elements in volume")



    @staticmethod
    def common_elements(node_ids, mesh, node_els, subset=False, max=1000):
        """
        Finds elements common to the given nodes.
        :param node_ids: Ids of the nodes for which we look for common elements.
        :param mesh:
        :param node_els: node -> element map
        :param subset: if true, it returns all the elements that are adjacent to at least one of the nodes
                       if false, it returns all the elements adjacent to all the nodes
        :param max:
        :return:
        """
        # Generates active elements common to given nodes.
        node_sets = [node_els[n] for n in node_ids]
        if subset:
            elements = list(set(itertools.chain.from_iterable(node_sets)))  # remove duplicities
        else:
            elements = set.intersection(*node_sets)

        if len(elements) > max:
            print("Too many connected elements:", len(elements), " > ", max)
            for eid in elements:
                type, tags, node_ids = mesh.elements[eid]
                print("  eid: ", eid, node_ids)

        def active(mesh, element_iterable):
            for eid in element_iterable:
                if eid in mesh.elements:
                    yield eid

        # return elements
        return active(mesh, elements)


    @staticmethod
    def find_fracture_neigh(mesh, fract_regions, n_levels=1):
        """
        Find neighboring elements in the bulk rock in the vicinity of the fractures.
        Creates several levels of neighbors.
        :param mesh: GmshIO mesh object
        :param fract_regions: list of physical names of the fracture regions
        :param n_levels: number of layers of elements from the fractures
        :return:
        """

        # make node -> element map
        node_els = collections.defaultdict(set)
        max_ele_id = 0
        for eid, e in mesh.elements.items():
            max_ele_id = max(max_ele_id, eid)
            type, tags, node_ids = e
            for n in node_ids:
                node_els[n].add(eid)

        print("max_ele_id = %d" % max_ele_id)

        # select ids of fracture regions
        fr_regs = fract_regions
        # fr_regs = []
        # for fr in fract_regions:
        #     rid, dim = mesh.physical['fr']
        #     assert dim == 2
        #     fr_regs.append(rid)

        # for n in node_els:
        #     if len(node_els[n]) > 1:
        #         print(node_els[n])

        visited_elements = np.zeros(shape=(max_ele_id + 1, 1), dtype=int)
        fracture_neighbors = []

        def find_neighbors(mesh, element, level, fracture_neighbors, visited_elements):
            """
            Auxiliary function which finds bulk neighbor elements to 'element' and
            saves them to list 'fracture_neighbors'.
            'visited_elements' keeps track of already investigated elements
            'level' is number of layer from the fractures in which we search
            """
            type, tags, node_ids = element
            ngh_elements = Flow123d_WGC2020.common_elements(node_ids, mesh, node_els, True)
            for ngh_eid in ngh_elements:
                if visited_elements[ngh_eid] > 0:
                    continue
                ngh_ele = mesh.elements[ngh_eid]
                ngh_type, ngh_tags, ngh_node_ids = ngh_ele
                if ngh_type == 4:  # if they are bulk elements and not already added
                    visited_elements[ngh_eid] = 1
                    fracture_neighbors.append((ngh_eid, level))  # add them

        # ele type: 1 - line, 2-triangle, 4-tetrahedron, 15-node
        # find the first layer of elements neighboring to fractures
        for eid, e in mesh.elements.items():
            type, tags, node_ids = e
            if type == 2:  # fracture elements
                visited_elements[eid] = 1
                if tags[0] not in fr_regs:  # is element in fracture region ?
                    continue
                find_neighbors(mesh, element=e, level=0, fracture_neighbors=fracture_neighbors,
                               visited_elements=visited_elements)

        # find next layers of elements from the first layer
        for i in range(1, n_levels):
            for eid, lev in fracture_neighbors:
                if lev < i:
                    e = mesh.elements[eid]
                    find_neighbors(mesh, element=e, level=i, fracture_neighbors=fracture_neighbors,
                                   visited_elements=visited_elements)

        return fracture_neighbors

    @staticmethod
    def prepare_th_input(config_dict):
        """
        Prepare FieldFE input file for the TH simulation.
        :param config_dict: Parsed config.yaml. see key comments there.
        """
        th_input_file = 'th_input.msh'
        if os.path.exists(th_input_file):
            return

        # pass
        # we have to read region names from the input mesh
        # get fracture regions ids
        orig_mesh_reader = gmsh_io.GmshIO()
        orig_mesh_reader.filename = config_dict["th_params"]["mesh"]
        orig_mesh_reader.read_physical_names()
        fr_regs = orig_mesh_reader.get_reg_ids_by_physical_names(config_dict["th_params"]["fracture_regions"], 2)

        # input_mesh = gmsh_io.GmshIO(config_dict['hm_params']['mesh'])
        #
        # is_bc_region = {}
        # for name, (id, _) in input_mesh.physical.items():
        #     unquoted_name = name.strip("\"'")
        #     is_bc_region[id] = (unquoted_name[0] == '.')

        # read mesh and mechanichal output data
        mechanics_output = os.path.join(config_dict['hm_params']["output_dir"], 'mechanics.msh')
        # mechanics_output = 'output_01_hm/mechanics.msh'

        mesh = gmsh_io.GmshIO(mechanics_output)
        # map eid to the element position in the array
        # TODO: use extract_mesh
        ele_ids_map = dict()
        for i, eid in enumerate(mesh.elements.keys()):
            ele_ids_map[eid] = i

        init_fr_cs = float(config_dict['hm_params']['fr_cross_section'])
        init_fr_K = float(config_dict['hm_params']['fr_conductivity'])
        init_bulk_K = float(config_dict['hm_params']['bulk_conductivity'])
        min_fr_cross_section = float(config_dict['th_params']['min_fr_cross_section'])
        max_fr_cross_section = float(config_dict['th_params']['max_fr_cross_section'])

        # read cross-section from HM model
        time_idx = 1
        time, field_cs = mesh.element_data['cross_section_updated'][time_idx]
        # cut small and large values of cross-section
        cs = np.maximum(np.array([v[0] for v in field_cs.values()]), min_fr_cross_section)
        cs = np.minimum(cs, max_fr_cross_section)

        # IMPORTANT - we suppose mesh nodes continuous, so we can find neighboring elements
        fr_indices = mesh.get_elements_of_regions(fr_regs)
        # find bulk elements neighboring to the fractures
        fr_n_levels = config_dict["th_params"]["increased_bulk_cond_levels"]
        fracture_neighbors = Flow123d_WGC2020.find_fracture_neigh(mesh, fr_regs, n_levels=fr_n_levels)

        # create and fill conductivity field
        # set all values to initial bulk conductivity
        K = init_bulk_K * np.ones(shape=(len(ele_ids_map),1))
        # increase conductivity in fractures due to power law
        for eid in fr_indices:
            i = ele_ids_map[eid]
            K[i] = init_fr_K * (cs[i] / init_fr_cs) ** 2
        # increase the bulk conductivity in the vicinity of the fractures
        level_factor = [10**(fr_n_levels - i) for i in range(fr_n_levels)]
        for eid, lev in fracture_neighbors:
            assert lev < len(level_factor)
            # if eid >= len(ele_ids_map):
            #     print("eid {}, n {}", eid, len(ele_ids_map))
            # assert eid < len(ele_ids_map)
            assert ele_ids_map[eid] < len(K)
            K[ele_ids_map[eid]] = init_bulk_K * level_factor[lev]

        # get cs and K on fracture elements only
        cs_fr = np.array([cs[ele_ids_map[i]] for i in fr_indices])
        k_fr = np.array([K[ele_ids_map[i]] for i in fr_indices])

        # compute cs and K statistics and write it to a file
        fr_param = {}
        avg = float(np.average(cs_fr))
        median = float(np.median(cs_fr))
        interquantile = float(1.5 * (np.quantile(cs_fr, 0.75) - np.quantile(cs_fr, 0.25)))
        fr_param["fr_cross_section"] = {"avg": avg, "median": median, "interquantile": interquantile}

        avg = float(np.average(k_fr))
        median = float(np.median(k_fr))
        interquantile = float(1.5 * (np.quantile(k_fr, 0.75) - np.quantile(k_fr, 0.25)))
        fr_param["fr_conductivity"] = {"avg": avg, "median": median, "interquantile": interquantile}

        with open('fr_param_output.yaml', 'w') as outfile:
            yaml.dump(fr_param, outfile, default_flow_style=False)

        # mesh.write_fields('output_hm/th_input.msh', ele_ids, {'conductivity': K})
        th_input_file = 'th_input.msh'
        ele_ids = np.array(list(mesh.elements.keys()), dtype=float)
        with open(th_input_file, "w") as fout:
            mesh.write_ascii(fout)
            mesh.write_element_data(fout, ele_ids, 'conductivity', K)
            mesh.write_element_data(fout, ele_ids, 'cross_section_updated', cs[:, None])














    # @staticmethod
    # def get_result_description():
    #     """
    #     :return:
    #     """
    #     end_time = 30
    #     values = [ [ValueDescription(time=t, position="extraction_well", quantity="power", unit="MW"),
    #                 ValueDescription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
    #                 ] for t in np.linspace(0, end_time, 0.1)]
    #     power_series, temp_series = zip(*values)
    #     return power_series + temp_series


    @staticmethod
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

    @staticmethod
    def extract_results(config_dict):
        """
        :param config_dict: Parsed config.yaml. see key comments there.
        : return
        """
        bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
        out_regions = bc_regions[2:]

        th_res = Flow123d_WGC2020.extract_th_results(config_dict["th_params"]["output_dir"], out_regions, bc_regions)
        th_res_ref = Flow123d_WGC2020.extract_th_results(config_dict["th_params_ref"]["output_dir"], out_regions, bc_regions)

        return th_res, th_res_ref

    @staticmethod
    def extract_th_results(output_dir, out_regions, bc_regions):
        with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as f:
            power_times, reg_powers = Flow123d_WGC2020.extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
            power_series = -sum(reg_powers)

        with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as f:
            temp_times, reg_temps = Flow123d_WGC2020.extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
        with open(os.path.join(output_dir, "water_balance.yaml"), "r") as f:
            flux_times, reg_fluxes = Flow123d_WGC2020.extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
        sum_flux = sum(reg_fluxes)

        reg_temps = reg_temps - Flow123d_WGC2020.zero_temperature_offset

        avg_temp_flux = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
        return avg_temp_flux, power_series

    @staticmethod
    def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
        year_sec = 60 * 60 * 24 * 365

        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        temp_color = 'red'
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Temperature [C deg]', color=temp_color)
        ax1.plot(temp_times[1:] / year_sec, avg_temp[1:], color=temp_color)
        ax1.tick_params(axis='y', labelcolor=temp_color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        pow_color = 'blue'
        ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
        ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
        ax2.tick_params(axis='y', labelcolor=pow_color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
