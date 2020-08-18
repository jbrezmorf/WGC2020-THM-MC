import gmsh
import os
import sys
import shutil
import ruamel.yaml as yaml

from flow_mc_new import Flow123d_WGC2020

# from mlmc.random import correlated_field as cf
# from mlmc.tool import flow_mc
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool, ThreadPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.sim.simulation import QuantitySpec
from mlmc.tool import process_base
import mlmc.moments as moments
from mlmc.quantity_estimate import QuantityEstimate

import numpy as np
import matplotlib.pyplot as plt

class WGC2020_Process(process_base.ProcessBase):

    def __init__(self):
        #TODO: separate constructor and run call
        #TODO: should there be different config for Process and Simulation ?
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as f:
            self.config_dict = yaml.safe_load(f)
        self.config_dict["config_pbs"] = os.path.join(os.getcwd(), "config_PBS.yaml")
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
        # if n_samples > 1:
        #     self.calculate_moments(sampler)  # Simple moment check

        if self.config_dict["mesh_only"]:
            return

        self.get_some_results(sampler.sample_storage)


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
        # the length of level_parameters must correspond to number of MLMC levels, at least 1 !!!
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=[1])

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
            #if self.config_dict["collect_only"]:
                #return OneProcessPool(work_dir=self.work_dir)
            #else:
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
        sampling_pool = SamplingPoolPBS(job_weight=1, work_dir=self.work_dir, clean=self.clean)

        with open(self.config_dict["config_pbs"], "r") as f:
            pbs_config = yaml.safe_load(f)

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

    def calculate_moments(self, storage: SampleStorageHDF, qspec: QuantitySpec):
        """
        Calculate moments of given quantity for all times.
        :param storage: Sample HDF storage
        :param qspec: quantity given by QuantitySpec
        :return: moments means estimates, their variances; tuple of 2 np.arrays of length 3
        """
        n_moments = 3
        means = []
        vars = []
        for time_id in range(len(qspec.times)):
            true_domain = QuantityEstimate.estimate_domain(storage, qspec, time_id, quantile=0.01)
            # moments_fn = moments.Legendre(n_moments, true_domain)
            # moments_fn = moments.Monomial(n_moments, true_domain)

            # compute mean in real values (without transform to ref domain)
            # mean_moment_fn = moments.Monomial.factory(2, domain=true_domain, ref_domain=true_domain, safe_eval=False)
            # q_estimator = QuantityEstimate(sample_storage=storage, sim_steps=self.step_range,
            #                                qspec=qspec, time_id=time_id)
            # m, v = q_estimator.estimate_moments(mean_moment_fn)
            #
            # # The first moment is in any case 1 and its variance is 0
            # assert m[0] == 1
            # assert v[0] == 0

            # compute variance in real values (center by computed mean)
            # mean_moment_fn = moments.Monomial.factory(3, center=m[1])
            mean_moment_fn = moments.Monomial.factory(3, domain=true_domain, ref_domain=true_domain, safe_eval=False)
            q_estimator = QuantityEstimate(sample_storage=storage, sim_steps=self.step_range,
                                           qspec=qspec, time_id=time_id)
            mm, vv = q_estimator.estimate_moments(mean_moment_fn)

            # The first moment is in any case 1 and its variance is 0
            assert np.isclose(mm[0], 1, atol=1e-10)
            assert vv[0] == 0
            # assert np.isclose(mm[1], 0, atol=1e-10)

            # means.append([1, m[1], mm[2]])
            # vars.append([0, v[1], vv[2]])
            means.append(mm)
            vars.append(vv)
            # print("t = ", qspec.times[time_id], " means ", mm[1], mm[2])
            # print("t = ", qspec.times[time_id], " vars ", vv[1], vv[2])

        return np.array(means), np.array(vars)

    def get_some_results(self, sample_storage):
        # Open HDF sample storage
        # hdf_file = os.path.join(self.work_dir, "wgc2020_mlmc.hdf5")
        # sample_storage = SampleStorageHDF(file_path=hdf_file, append=True)

        print("N levels: ", sample_storage.n_levels())
        avg_temp_vals, avg_temp\
            = sample_storage.load_collected_values("0", "avg_temp", fine_res=True)
        power_vals, power\
            = sample_storage.load_collected_values("0", "power", fine_res=True)

        self.plot_histogram(avg_temp, avg_temp_vals, 30)
        self.plot_histogram(power, power_vals, 30)

        plot_dict = dict()
        plot_dict["ylabel"] = "Temperature [$^\circ$C]"
        plot_dict["file_name"] = "temp_comparison"
        plot_dict["color"] = "red"
        plot_dict["color_ref"] = "orange"
        self.plot_comparison(sample_storage, "avg_temp", "avg_temp_ref", plot_dict)

        plot_dict = dict()
        plot_dict["ylabel"] = "Power [MW]"
        plot_dict["file_name"] = "power_comparison"
        plot_dict["color"] = "blue"
        plot_dict["color_ref"] = "forestgreen"
        self.plot_comparison(sample_storage, "power", "power_ref", plot_dict)
        # TODO: problems:
        # quantity_estimate.get_level_results does not support array quantities
        # it only gets the first (i.e. temp at time 0)
        #
        # moments are mapped to interval [0,1]
        # 1. moment (mean) can be mapped by Moments.inv_linear() function
        # what about higher moments (need to get std)
        #
        # SampleStorageHDF.sample_pairs get the data
        # I implemented load_collected_values() which selects the quantities by name at all times

        return

    def plot_histogram(self, quantity: QuantitySpec, data, bins):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(quantity.name + " [" + quantity.unit + "]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        n_samples, ntimes = data.shape
        times = quantity.times
        t_min = 0
        t_max = len(times)-1
        # t_min = times[0]
        # t_max = int(times[-1])
        # for t in [0, int((t_min + t_max)/2), t_max]:
        for t in [0, int(t_max / 2), t_max]:
            print(t)
            label = "t = " + str(t) + " y"
            ax1.hist(data[:, t], alpha=0.5, bins=bins, label=label) #edgecolor='white')

        ns = "N = " + str(n_samples)
        ax1.set_title(ns)
        ax1.legend()
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".png"))
        # plt.show()

    def plot_comparison(self, sample_storage, quantity_name, quantity_ref_name, plot_dict):
        """
        Plots comparison of quantity values and some reference values in time.
        Computes first two moment means and its variances.
        :param sample_storage: Sample HDF storage
        :param quantity_name: string, name corresponds to its QuantitySpec
        :param quantity_ref_name: string, name corresponds to its QuantitySpec
        :param plot_dict: dictionary with several plot parameters (labels, colors...)
        :return:
        """
        quant_vals, quant_spec \
            = sample_storage.load_collected_values("0", quantity_name, fine_res=True)
        quant_ref_vals, quant_ref_spec \
            = sample_storage.load_collected_values("0", quantity_ref_name, fine_res=True)

        moments = self.calculate_moments(sample_storage, quant_spec)
        moments_ref = self.calculate_moments(sample_storage, quant_ref_spec)

        assert len(quant_spec.times) == len(quant_ref_spec.times)
        times = quant_spec.times

        # Plot temperature
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel(plot_dict["ylabel"], color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        self.plot_param(moments, ax1, times, plot_dict["color"], 'stimulated')
        self.plot_param(moments_ref, ax1, times, plot_dict["color_ref"], 'unmodified')

        # ax1.legend(["stimulated - mean", "stimulated - std", "reference - mean", "reference - std"])
        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, plot_dict["file_name"] + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, plot_dict["file_name"] + ".png"))
        # plt.show()

    def plot_param(self, moments, ax, X, col, legend):
        """
        Get all result values by given param name, e.g. param_name = "temp" - return all temperatures...
        :param moments: moments means estimates, their variances; tuple of 2 np.arrays of length at least 3
        :param ax: plot axis
        :param X: x-axis data (time)
        :param col: color
        :param legend: legend label
        :return: void
        """
        mom_means, mom_vars = moments
        N = len(mom_means)

        # mom_means, mom_vars = mom_means[1:, :], mom_vars[1:, :] # omit time=0
        q_mean = mom_means[:, 1]
        q_mean_err = np.sqrt(mom_vars[:, 1])
        q_var = (mom_means[:, 2] - q_mean ** 2) * N / (N-1)
        # print("    means: ", mom_means[-1, :])
        # print("    vars: ", mom_vars[-1, :])
        # print("    alt var: ", mom_vars[-1, 1]*N)
        # print("    qvar : ", q_var[-1])
        # print("    qvar_err : ", q_var[-1])
        #q_var = q_mean_err * N
        q_std = np.sqrt(q_var)
        q_std_err = np.sqrt(q_var + np.sqrt(mom_vars[:, 2] * N / (N-1)))

        ax.fill_between(X, q_mean - q_mean_err, q_mean + q_mean_err,
                         color=col, alpha=1, label=legend + " - mean")
        ax.fill_between(X, q_mean - q_std, q_mean + q_std,
                         color=col, alpha=0.2, label=legend + " - std")
        ax.fill_between(X, q_mean - q_std_err, q_mean - q_std,
                         color=col, alpha=0.4, label = None)
        ax.fill_between(X, q_mean + q_std, q_mean + q_std_err,
                         color=col, alpha=0.4, label = None)

if __name__ == "__main__":
    pr = WGC2020_Process()
