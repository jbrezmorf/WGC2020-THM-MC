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

        self.need_workspace = True
        self.work_dir = config["work_dir"]
        self.clean = clean


    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float])-> LevelSimulation:
        """
        Overrides Simulation.level_instance
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.)
        :param fine_level_params: in this version, it is just fine simulation step
        :param coarse_level_params: in this version, it is just coarse simulation step
        :return: mlmc.LevelSimulation object, this object is serialized in SamplingPoolPbs and deserialized in PbsJob,
         so it allows pass simulation data from main process to PBS process
        """

        # Set fine simulation common files directory
        # Files in the directory are used by each simulation at that level
        common_files_dir = os.path.join(self.work_dir, "common_files")
        force_mkdir(common_files_dir, force=self.clean)

        # Simulation config
        # Configuration is used in mlmc.tool.pbs_job.PbsJob instance which is run from PBS process
        # It is part of LevelSimulation which is serialized and then deserialized in mlmc.tool.pbs_job.PbsJob
        config = dict()
        config["common_files_dir"] = common_files_dir

        return LevelSimulation(config_dict=config,
                               # task_size=len(fine_mesh_data['points']),
                               task_size=1000,
                               calculate=Flow123d_WGC2020.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True  # If True, a sample directory is created
                               )

    @staticmethod
    def calculate(config_dict, sample_workspace=None):
        """
        Overrides Simulation.calculate
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        # does all the data preparation, passing
        # running simulation
        # extracting results

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
        times = [1]
        spec1 = QuantitySpec(name="avg_temp_flux", unit="t/m/m/s", shape=(1, 1), times=times, locations=['.well'])
        spec2 = QuantitySpec(name="power", unit="J", shape=(1, 1), times=times, locations=['.well'])
        return [spec1, spec2]