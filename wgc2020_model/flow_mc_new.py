import os
import os.path
import subprocess
import numpy as np
import json
import glob
import shutil
import copy
import yaml
from typing import List
from mlmc.level_simulation import LevelSimulation
from mlmc.tool import gmsh_io
from mlmc.sim.simulation import Simulation
from mlmc.sim.simulation import QuantitySpec
from mlmc.random import correlated_field as cf

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
                               task_size=1,
                               calculate=Flow123d_WGC2020.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True  # If True, a sample directory is created
                               )

    @staticmethod
    def calculate(config_dict, sample_workspace=None):
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

        mesh_repo = config_dict.get('mesh_repository', None)

        if mesh_repo:
            healed_mesh = Flow123d_WGC2020.sample_mesh_repository(mesh_repo)
            Flow123d_WGC2020.config_fracture_regions(config_dict, config_dict["fracture_regions"])
        else:
            # plot_fr_orientation(fractures)
            fractures = generate_fractures(config_dict)
            healed_mesh = prepare_mesh(config_dict, fractures)

        healed_mesh_bn = os.path.basename(healed_mesh)
        config_dict["hm_params"]["mesh"] = healed_mesh_bn
        config_dict["th_params"]["mesh"] = healed_mesh_bn
        config_dict["th_params_ref"]["mesh"] = healed_mesh_bn

        hm_succeed = Flow123d_WGC2020.call_flow(config_dict, 'hm_params', result_files=["mechanics.msh"])
        th_succeed = Flow123d_WGC2020.call_flow(config_dict, 'th_params_ref', result_files=["energy_balance.yaml"])
        th_succeed = False
        # if hm_succeed:
        #     prepare_th_input(config_dict)
        #     th_succeed = call_flow(config_dict, 'th_params', result_files=["energy_balance.yaml"])

            # if th_succeed:
            #    series = extract_results(config_dict)
            #    plot_exchanger_evolution(*series)
        print("Finished")

        # TODO: extract results, pass as tuple (fine, coarse) -> (fine, fine)
        result = (np.array([np.random.normal()]), np.array([np.random.normal()]))
        return result

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
        times = [1]
        spec1 = QuantitySpec(name="avg_temp_flux", unit="t/m/m/s", shape=(1, 1), times=times, locations=['.well'])
        spec2 = QuantitySpec(name="power", unit="J", shape=(1, 1), times=times, locations=['.well'])
        return [spec1, spec2]







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
        substitute_placeholders(os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml'),
                                fname + '.yaml',
                                params)
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
        conv_check = Flow123d_WGC2020.check_conv_reasons(os.path.join(output_dir, "flow123.0.log"))
        print("converged: ", conv_check)
        return status  # and conv_check

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
    def config_fracture_regions(config, used_families):
        for model in ["hm_params", "th_params", "th_params_ref"]:
            model_dict = config[model]
            model_dict["fracture_regions"] = list(used_families)
            # model_dict["left_well_fracture_regions"] = [".{}_left_well".format(f) for f in used_families]
            # model_dict["right_well_fracture_regions"] = [".{}_right_well".format(f) for f in used_families]
            model_dict["left_well_fracture_regions"] = [".left_fr_left_well"]
            model_dict["right_well_fracture_regions"] = [".right_fr_right_well"]
