import os
import ruamel.yaml as yaml
import numpy as np
import matplotlib.pyplot as plt

from flow_mc_new import Flow123d_WGC2020

import mlmc as mlmc
from mlmc.quantity.quantity_spec import QuantitySpec, ChunkSpec
from mlmc.quantity.quantity import Quantity
from mlmc.quantity.quantity import make_root_quantity
import mlmc.tool.process_base


class WGC2020_Process(mlmc.tool.process_base.ProcessBase):

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
        self.config_dict["work_dir"] = os.path.abspath(self.work_dir)
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
        hdf_file = os.path.join(self.config_dict["work_dir"], "wgc2020_mlmc.hdf5")
        if self.clean:
            # Remove HFD5 file
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
        sample_storage = mlmc.sample_storage_hdf.SampleStorageHDF(file_path=hdf_file)

        # Create sampler, it manages sample scheduling and so on
        # the length of level_parameters must correspond to number of MLMC levels, at least 1 !!!
        sampler = mlmc.sampler.Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool,
                                      sim_factory=simulation_factory, level_parameters=[1])

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
            mlmc.sampling_pool.ProcessPool(n_processes=self.config_dict["local"]["np"], work_dir=self.work_dir)
        else:
            return mlmc.sampling_pool.OneProcessPool(work_dir=self.work_dir, debug=self.config_dict["debug"])

    def create_pbs_sampling_pool(self):
        """
        Initialize object for PBS execution
        :return: None
        """
        # Create PBS sampling pool
        sampling_pool = mlmc.sampling_pool_pbs.SamplingPoolPBS(job_weight=1, work_dir=self.work_dir, clean=self.clean)

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

    def calculate_moments(self, sample_storage, qspec: QuantitySpec, quantity: Quantity):
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
            q_value = quantity[qspec.times[time_id]]
            true_domain = mlmc.Estimate.estimate_domain(q_value, sample_storage, quantile=0.01)
            moments_fn = mlmc.Monomial(n_moments, domain=true_domain, ref_domain=true_domain, safe_eval=False)
            estimate_obj = mlmc.Estimate(q_value, sample_storage=sample_storage, moments_fn=moments_fn)
            mm, vv = estimate_obj.estimate_moments()

            # The first moment is in any case 1 and its variance is 0
            assert np.isclose(mm[0], 1, atol=1e-10)
            assert vv[0] == 0

            means.append(mm)
            vars.append(vv)
            # print("t = ", qspec.times[time_id], " means ", mm[1], mm[2])
            # print("t = ", qspec.times[time_id], " vars ", vv[1], vv[2])

        return np.array(means), np.array(vars)

    def get_quantity(self, result_format, root_quantity, quantity_name):

        iqspec, qspec = next(((i, q) for i, q in enumerate(result_format) if q.name == quantity_name), (0, None))
        assert qspec is not None

        quantity = root_quantity[quantity_name]
        return qspec, quantity


    def get_some_results(self, sample_storage):
        print("N levels: ", sample_storage.get_n_levels())
        print("N collected: ", sample_storage.get_n_collected()[0])
        assert sample_storage.get_n_collected()[0] > 0

        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing your real quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)

        n_fracture_ele_QS, n_fracture_ele_Q = self.get_quantity(result_format, root_quantity, "n_fracture_elements")
        n_contact_ele_QS, n_contact_ele_Q = self.get_quantity(result_format, root_quantity, "n_contact_elements")

        # [:, 0] due to times
        self.plot_fractures_histogram(sample_storage, n_fracture_ele_QS, n_fracture_ele_Q, 30)
        self.plot_fractures_histogram(sample_storage, n_contact_ele_QS, n_fracture_ele_Q, 30)

        # n_contact_ele_percentage_Q = np.divide(n_contact_ele_Q, n_fracture_ele_Q) * 100
        n_contact_ele_percentage_Q = n_contact_ele_Q / n_fracture_ele_Q * 100
        n_contact_ele_QS.name = n_contact_ele_QS.name + "_ratio"
        n_contact_ele_QS.unit = "%"
        self.plot_fractures_histogram(sample_storage, n_contact_ele_QS, n_contact_ele_percentage_Q, 30)

        self.get_variant_results(sample_storage, "avg_temp_03", "power_03")
        # self.get_variant_results(root_quantity, "avg_temp_04", "power_04")

        # TODO: problems:
        # quantity_estimate.get_level_results does not support array quantities
        # it only gets the first (i.e. temp at time 0)
        #
        # moments are mapped to interval [0,1]
        # 1. moment (mean) can be mapped by Moments.inv_linear() function
        # what about higher moments (need to get std)

        return

    def get_variant_results(self, sample_storage, avg_temp_qname, power_qname):

        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing your real quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)
        n_samples = sample_storage.get_n_collected()[0]
        chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=n_samples))

        avg_temp_QS, avg_temp_Q = self.get_quantity(result_format, root_quantity, avg_temp_qname)
        power_QS, power_Q = self.get_quantity(result_format, root_quantity, power_qname)

        avg_temp_qname_ref = "avg_temp_02"
        power_qname_ref = "power_02"
        avg_temp_ref_QS, avg_temp_ref_Q = self.get_quantity(result_format, root_quantity, avg_temp_qname_ref)
        power_ref_QS, power_ref_Q = self.get_quantity(result_format, root_quantity, power_qname_ref)

        # avg_temp_vals, avg_temp \
        #     = sample_storage.load_collected_values("0", avg_temp_quant, fine_res=True)
        # power_vals, power \
        #     = sample_storage.load_collected_values("0", power_quant, fine_res=True)

        self.plot_histogram(avg_temp_QS, avg_temp_Q, chunk_spec, 30)
        self.plot_histogram(power_QS, power_Q, chunk_spec, 30)

        plot_dict = dict()
        plot_dict["ylabel"] = "Temperature [$^\circ$C]"
        plot_dict["file_name"] = avg_temp_qname + "_comparison"
        plot_dict["color"] = "red"
        plot_dict["color_ref"] = "orange"
        self.plot_comparison(sample_storage, avg_temp_QS, avg_temp_Q, avg_temp_ref_QS, avg_temp_ref_Q,
                             chunk_spec, plot_dict)

        plot_dict = dict()
        plot_dict["ylabel"] = "Power [MW]"
        plot_dict["file_name"] = power_qname + "_comparison"
        plot_dict["color"] = "blue"
        plot_dict["color_ref"] = "forestgreen"
        self.plot_comparison(sample_storage, power_QS, power_Q, power_ref_QS, power_ref_Q,
                             chunk_spec, plot_dict)

    def plot_fractures_histogram(self, sample_storage, quant_spec: QuantitySpec, quantity: Quantity, bins):

        n_samples = sample_storage.get_n_collected()[0]
        chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=n_samples))
        # select by time interpolation and location
        data_quant = quantity.time_interpolation(0)['-']
        # get samples data
        data_chunk = data_quant.samples(chunk_spec)
        data = data_chunk[0, :].flatten()     # [0] time, [0] location

        fig, ax1 = plt.subplots()
        ax1.set_xlabel(quant_spec.name + " [" + quant_spec.unit + "]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        ax1.hist(data, alpha=0.5, bins=bins)  # edgecolor='white')

        ns = "N = " + str(n_samples)
        ax1.set_title(ns)
        # ax1.legend()
        fig.savefig(os.path.join(self.work_dir, quant_spec.name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, quant_spec.name + ".png"))
        # plt.show()

    def plot_histogram(self, quant_spec: QuantitySpec, quantity: Quantity, chunk_spec: ChunkSpec, bins):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(quant_spec.name + " [" + quant_spec.unit + "]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        data_chunk = quantity.samples(chunk_spec)
        data = data_chunk.squeeze()

        ntimes, n_samples = data.shape
        times = quant_spec.times
        t_min = 0
        t_max = len(times)-1
        # t_min = times[0]
        # t_max = int(times[-1])
        # for t in [0, int((t_min + t_max)/2), t_max]:
        for t in [0, int(t_max / 2), t_max]:
            # print(t)
            label = "t = " + str(t) + " y"
            ax1.hist(data[t, :], alpha=0.5, bins=bins, label=label) #edgecolor='white')

        ns = "N = " + str(n_samples)
        ax1.set_title(ns)
        ax1.legend()
        fig.savefig(os.path.join(self.work_dir, quant_spec.name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, quant_spec.name + ".png"))
        # plt.show()

    def plot_comparison(self, sample_storage,
                        quant_spec: QuantitySpec, quantity: Quantity,
                        quant_ref_spec: QuantitySpec, quantity_ref: Quantity,
                        chunk_spec: ChunkSpec, plot_dict):
        """
        Plots comparison of quantity values and some reference values in time.
        Computes first two moment means and its variances.
        :param sample_storage: Sample HDF storage
        :param quantity_name: string, name corresponds to its QuantitySpec
        :param quantity_ref_name: string, name corresponds to its QuantitySpec
        :param plot_dict: dictionary with several plot parameters (labels, colors...)
        :return:
        """
        data_chunk = quantity.samples(chunk_spec)
        data_ref_chunk = quantity_ref.samples(chunk_spec)
        data = data_chunk.squeeze()
        data_ref = data_ref_chunk.squeeze()

        # quant_vals, quant_spec \
        #     = sample_storage.load_collected_values("0", quantity_name, fine_res=True)
        # quant_ref_vals, quant_ref_spec \
        #     = sample_storage.load_collected_values("0", quantity_ref_name, fine_res=True)

        moments = self.calculate_moments(sample_storage, quant_spec, quantity)
        moments_ref = self.calculate_moments(sample_storage, quant_ref_spec, quantity_ref)

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
