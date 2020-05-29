import os
import sys
import shutil
import ruamel.yaml as yaml
import numpy as np

from mlmc.random import correlated_field as cf
from mlmc.tool import flow_mc
from mlmc.moments import Legendre
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool, ThreadPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.tool import base_process
# from mlmc.tool.flow_mc import FlowSim
from flow_mc_new import Flow123d_WGC2020
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate


class WGC2020_Process(base_process.Process):

    def __init__(self):
        #TODO: separate constructor and run call
        #TODO: should there be different config for Process and Simulation ?
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as f:
            self.config_dict = yaml.safe_load(f)
        super(WGC2020_Process, self).__init__()

    def run(self, renew=False):
        """
        Run MLMC
        :param renew: If True then rerun failed samples with same sample id
        :return: None
        """

        # Create working directory if necessary
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)
        self.config_dict["work_dir"] = self.work_dir
        self.config_dict["script_dir"] = os.getcwd()

        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
        sampler = self.setup_config(n_levels=1, clean=True)
        # Schedule samples
        n_samples = self.config_dict["n_samples"]
        self.generate_jobs(sampler, n_samples=n_samples, renew=renew)

        self.all_collect([sampler])  # Check if all samples are finished
        self.calculate_moments(sampler)  # Simple moment check


    def setup_config(self, n_levels, clean):
        """
        # TODO: specify, what should be done here.
        - creation of Simulation
        - creation of Sampler
        - hdf file ?
        - why step_range must be here ?


        Simulation dependent configuration
        :param step_range: Simulation's step range, length of them is number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.sampler instance
        """
        self.set_environment_variables()

        sampling_pool = self.create_sampling_pool()

        # Create simulation factory
        simulation_factory = Flow123d_WGC2020(config=self.config_dict, clean=clean)

        # Create HDF sample storage, possibly remove old one
        hdf_file = os.path.join(self.work_dir, "wgc2020_mlmc.hdf5")
        if self.clean:
            # Remove HFD5 file
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
        sample_storage = SampleStorageHDF(
            file_path=hdf_file,
            append=self.append)

        # Create sampler, it manages sample scheduling and so on
        step_range = [1]   # auxiliary variable, not used
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          step_range=step_range)

        return sampler

    def set_environment_variables(self):
        run_on_metacentrum = self.config_dict["run_on_metacentrum"]

        #TODO: where these variables come from, why and how set them ?
        if run_on_metacentrum:
            # Charon
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.config_dict["_aux_flow_path"] = self.config_dict["metacentrum"]["flow_executable"].copy()
            self.config_dict["_aux_gmsh_path"] = self.config_dict["metacentrum"]["gmsh_executable"].copy()
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.config_dict["_aux_flow_path"] = self.config_dict["local"]["flow_executable"].copy()
            self.config_dict["_aux_gmsh_path"] = self.config_dict["local"]["gmsh_executable"].copy()

    def create_sampling_pool(self):

        if self.config_dict["run_on_metacentrum"]:
            return self.create_pbs_sampling_pool()
        elif self.config_dict["local"]["np"] > 1:
            # Simulations run in different processes
            ProcessPool(n_processes=self.config_dict["local"]["np"], work_dir=self.work_dir)
        else:
            return OneProcessPool(work_dir=self.work_dir)

    def create_pbs_sampling_pool(self):
        """
        Initialize object for PBS execution
        :return: None
        """
        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(job_weight=20000000, work_dir=self.work_dir, clean=self.clean)

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='128mb',
            queue='charon_2h',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            # pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc',
            python='python3',
            env_setting=['cd {work_dir}',
                         'module load python36-modules-gcc',
                         'source env/bin/activate',
                         'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',
                         'module use /storage/praha1/home/jan-hybs/modules',
                         'module load python36-modules-gcc',
                         'module load flow123d',
                         'module list']
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def generate_jobs(self, sampler, n_samples=None, renew=False):
        """
        Why this overrides base? What does the base method?

        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """
        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            if n_samples is not None:
                sampler.set_initial_n_samples(n_samples)
            sampler.schedule_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)

    def calculate_moments(self, sampler):
        """
        Calculate moments through the mlmc.QuantityEstimate
        :param sampler: sampler (mlmc.Sampler)
        :return: None
        """
        # Simple moment evaluation
        moments_fn = self.set_moments(sampler.sample_storage)

        q_estimator = QuantityEstimate(sample_storage=sampler.sample_storage, moments_fn=moments_fn,
                                       sim_steps=self.step_range)

        print("collected samples ", sampler._n_scheduled_samples)
        means, vars = q_estimator.estimate_moments(moments_fn)
        print("means ", means)
        print("vars ", vars)

        # The first moment is in any case 1 and its variance is 0
        assert means[0] == 1
        # assert np.isclose(means[1], 0, atol=1e-2)
        assert vars[0] == 0

    def set_moments(self, sample_storage):
        n_moments = 5
        true_domain = QuantityEstimate.estimate_domain(sample_storage, quantile=0.01)
        return Legendre(n_moments, true_domain)


if __name__ == "__main__":
    pr = WGC2020_Process()
