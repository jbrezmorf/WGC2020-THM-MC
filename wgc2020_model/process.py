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
from mlmc.tool import process_base
import mlmc.moments as moments

from mlmc.estimator import Estimate
from mlmc.quantity_estimate import estimate_mean
from mlmc.quantity_estimate import moment

from mlmc.quantity import make_root_quantity
from mlmc.quantity import Quantity
from mlmc.quantity_spec import QuantitySpec
from mlmc.quantity_spec import ChunkSpec

import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title


class WGC2020_Process(process_base.ProcessBase):

    def __init__(self):
        #TODO: separate constructor and run call
        #TODO: should there be different config for Process and Simulation ?
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as f:
            self.config_dict = yaml.safe_load(f)
        self.config_dict["config_pbs"] = os.path.join(os.getcwd(), "config_PBS.yaml")
        super(WGC2020_Process, self).__init__()

        # Create simulation factory
        self.simulation_factory = None

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

        self.simulation_factory = Flow123d_WGC2020(config=self.config_dict, clean=self.clean)

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

        # Create HDF sample storage, possibly remove old one
        hdf_file = os.path.join(self.work_dir, "wgc2020_mlmc.hdf5")
        if self.clean:
            # Remove HFD5 file
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
        sample_storage = SampleStorageHDF(
            file_path=hdf_file)

        # Create sampler, it manages sample scheduling and so on
        # the length of level_parameters must correspond to number of MLMC levels, at least 1 !!!
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool,
                          sim_factory=self.simulation_factory, level_parameters=[[1]])

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
        debug = self.config_dict["debug"]
        if self.config_dict["run_on_metacentrum"]:
            #if self.config_dict["collect_only"]:
                #return OneProcessPool(work_dir=self.work_dir)
            #else:
            return self.create_pbs_sampling_pool(debug)
        elif self.config_dict["local"]["np"] > 1:
            # Simulations run in different processes
            ProcessPool(n_processes=self.config_dict["local"]["np"], work_dir=self.work_dir)
        else:
            return OneProcessPool(work_dir=self.work_dir, debug=debug)

    def create_pbs_sampling_pool(self, debug):
        """
        Initialize object for PBS execution
        :return: None
        """
        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(job_weight=1, work_dir=self.work_dir, clean=self.clean, debug=debug)

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

    def calculate_moments(self, qspec: QuantitySpec, quantity: Quantity, storage: SampleStorageHDF):
        """
        Calculate moments of given quantity for all times.
        :param storage: Sample HDF storage
        :param qspec: quantity given by QuantitySpec
        :return: moments means estimates, their variances; tuple of 2 np.arrays of length 3
        """
        n_moments = 3
        means = []
        vars = []
        for time in qspec.times:
            true_domain = Estimate.estimate_domain(quantity[time], storage)
            mean_moment_fn = moments.Monomial(3, domain=true_domain, ref_domain=true_domain, safe_eval=False)
            q_estimator = Estimate(quantity=quantity[time], sample_storage=storage, moments_fn=mean_moment_fn)
            mm, vv = q_estimator.estimate_moments(mean_moment_fn)

            # The first moment is in any case 1 and its variance is 0
            assert np.isclose(mm[0], 1, atol=1e-10)
            assert vv[0] == 0

            means.append(mm)
            vars.append(vv)
            # print("t = ", qspec.times[time_id], " means ", mm[1], mm[2])
            # print("t = ", qspec.times[time_id], " vars ", vv[1], vv[2])

        return np.array(means), np.array(vars)

    def get_some_results(self, sample_storage):

        q_spec_list = self.simulation_factory.result_format()
        quantity_storage = make_root_quantity(sample_storage, q_specs=q_spec_list)
        print("N_collected: ", quantity_storage.n_collected())
        print("Level ids: ", quantity_storage.level_ids())

        level_id = 0
        n_samples = sample_storage.get_n_collected()[level_id]
        ch_spec = ChunkSpec(level_id=level_id, n_samples=n_samples)

        self.get_variant_results(quantity_storage, sample_storage, "avg_temp_03", "power_03")
        self.get_variant_results(quantity_storage, sample_storage, "avg_temp_04", "power_04")

        n_fracture_ele_spec = next(q for q in q_spec_list if q.name == "n_fracture_elements")
        n_fracture_ele_data = quantity_storage["n_fracture_elements"].samples(ch_spec)
        n_contact_ele_spec = next(q for q in q_spec_list if q.name == "n_contact_elements")
        n_contact_ele_data = quantity_storage["n_contact_elements"].samples(ch_spec)

        # [:, 0] due to times
        self.plot_fractures_histogram(n_fracture_ele_spec, n_fracture_ele_data, 30)
        self.plot_fractures_histogram(n_contact_ele_spec, n_contact_ele_data, 30)

        n_contact_ele_percentage = np.divide(n_contact_ele_data, n_fracture_ele_data) * 100
        n_contact_ele_spec.name = n_contact_ele_spec.name + "_ratio"
        n_contact_ele_spec.unit = "%"
        self.plot_fractures_histogram(n_contact_ele_spec, n_contact_ele_percentage, 30)

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

    def get_variant_results(self, quantity_storage, sample_storage, avg_temp_name, power_name):
        avg_temp_name_ref = "avg_temp_02"
        power_name_ref = "power_02"

        q_spec_list = self.simulation_factory.result_format()
        level_id = 0
        n_samples = quantity_storage.n_collected()[0]
        ch_spec = ChunkSpec(level_id=level_id, n_samples=n_samples)

        # get quantity specs and samples for histograms
        avg_temp_spec = next(q for q in q_spec_list if q.name == avg_temp_name)
        avg_temp_data = quantity_storage[avg_temp_name].samples(ch_spec)
        power_spec = next(q for q in q_spec_list if q.name == power_name)
        power_data = quantity_storage[power_name].samples(ch_spec)

        self.plot_histogram(avg_temp_spec, avg_temp_data, 30)
        self.plot_histogram(power_spec, power_data, 30)

        # get reference quantity specs
        avg_temp_ref_spec = next(q for q in q_spec_list if q.name == avg_temp_name_ref)
        # avg_temp_ref_data = quantity_storage[avg_temp_name_ref].samples(ch_spec)
        power_ref_spec = next(q for q in q_spec_list if q.name == power_name_ref)
        # power_ref_data = quantity_storage[power_name_ref].samples(ch_spec)

        plot_dict = dict()
        plot_dict["ylabel"] = "Temperature [$^\circ$C]"
        plot_dict["file_name"] = avg_temp_name + "_comparison"
        plot_dict["color"] = "red"
        plot_dict["color_ref"] = "orange"
        self.plot_comparison(quantity_storage, sample_storage,
                             avg_temp_spec, avg_temp_ref_spec,
                             plot_dict)

        plot_dict = dict()
        plot_dict["ylabel"] = "Power [MW]"
        plot_dict["file_name"] = power_name + "_comparison"
        plot_dict["color"] = "blue"
        plot_dict["color_ref"] = "forestgreen"
        self.plot_comparison(quantity_storage, sample_storage,
                             power_spec, power_ref_spec,
                             plot_dict)

    def plot_fractures_histogram(self, quantity: QuantitySpec, data, bins):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(quantity.name + " [" + quantity.unit + "]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        const_time, n_samples, scalar = data.shape
        ax1.hist(data[0, :, 0], alpha=0.5, bins=bins)  # edgecolor='white')

        ns = "N = " + str(n_samples)
        ax1.set_title(ns)
        # ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".png"))
        # plt.show()

    def plot_histogram(self, quantity: QuantitySpec, data, bins):
        fig, ax1 = plt.subplots()
        if quantity.unit == 'C':
            ax1.set_xlabel(quantity.name + " [$^\circ$C]")
        else:
            ax1.set_xlabel(quantity.name + " [" + quantity.unit + "]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        ntimes, n_samples, loc = data.shape
        times = quantity.times
        t_min = 0
        t_max = len(times)-1
        # t_min = times[0]
        # t_max = int(times[-1])
        # for t in [0, int((t_min + t_max)/2), t_max]:
        for t in [0, int(t_max / 2), t_max]:
            print(t)
            label = "t = " + str(t) + " y"
            ax1.hist(data[t, :, 0], alpha=0.5, bins=bins, label=label) #edgecolor='white')

        ns = "N = " + str(n_samples)
        ax1.set_title(ns)
        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, quantity.name + ".png"))
        # plt.show()

    def plot_comparison(self, quantity_storage: Quantity, sample_storage: SampleStorageHDF,
                        quant_spec: QuantitySpec, quant_ref_spec: QuantitySpec, plot_dict):
        """
        Plots comparison of quantity values and some reference values in time.
        Computes first two moment means and its variances.
        :param quantity_storage: root quantity storage
        :param sample_storage: Sample HDF storage
        :param quant_spec: selected quantity of interest QuantitySpec
        :param quant_ref_spec: reference quantity of interest QuantitySpec
        :param plot_dict: dictionary with several plot parameters (labels, colors...)
        :return:
        """
        q_moments = self.calculate_moments(quant_spec, quantity_storage[quant_spec.name], sample_storage)
        q_moments_ref = self.calculate_moments(quant_ref_spec, quantity_storage[quant_ref_spec.name], sample_storage)

        assert len(quant_spec.times) == len(quant_ref_spec.times)
        times = quant_spec.times

        # Plot temperature
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel(plot_dict["ylabel"], color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        self.plot_param(q_moments, ax1, times, plot_dict["color"], 'stimulated')
        self.plot_param(q_moments_ref, ax1, times, plot_dict["color_ref"], 'unmodified')

        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, plot_dict["file_name"] + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, plot_dict["file_name"] + ".png"))
        # plt.show()

    def plot_param(self, moments_in, ax, X, col, legend):
        """
        Get all result values by given param name, e.g. param_name = "temp" - return all temperatures...
        :param moments_in: moments means estimates, their variances; tuple of 2 np.arrays of length at least 3
        :param ax: plot axis
        :param X: x-axis data (time)
        :param col: color
        :param legend: legend label
        :return: void
        """
        mom_means, mom_vars = moments_in
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
        # q_var = q_mean_err * N
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
