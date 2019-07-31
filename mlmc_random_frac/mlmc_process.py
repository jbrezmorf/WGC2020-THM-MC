import sys
import os
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '../MLMC/src'))
sys.path.append(os.path.join(src_path, '../dfn/src'))

from mlmc import mlmc
from mlmc import base_process
from mlmc.estimate import Estimate, CompareLevels
from mlmc.moments import Legendre, Monomial

import pbs
from rand_frac_simulation import RandomFracSimulation


def load_config_dict(cfg):
    with open(cfg, "r") as f:
        return yaml.safe_load(f)


class Process(base_process.Process):
    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        self.flow123d = self.config_dict["flow_executable"]
        self.pbs_config = dict(
            job_weight=250000,  # max number of elements per job
            n_cores=1,
            n_nodes=1,
            select_flags=[],
            mem='8gb',
            queue='charon',
            qsub=None)
        if(self.config_dict["metacentrum"]):
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config.update(dict(
                qsub='/usr/bin/qsub',
                modules=[

                ]
            ))
        else:
            # pbs_config is necessary for the local run but is not used
            # as long as the pbs key is set to None
            self.pbs_config['qsub'] = None
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60

        self.mc_samples = self.config_dict["mc_samples"]

    def create_pbs_object(self, output_dir, clean):
        """
        Initialize object for PBS execution
        :param output_dir: Output directory
        :param clean: bool, if True remove existing files
        :return: None
        """
        pbs_work_dir = os.path.join(output_dir, "scripts")
        num_jobs = 0
        if os.path.isdir(pbs_work_dir):
            num_jobs = len([_ for _ in os.listdir(pbs_work_dir)])

        self.pbs_obj = pbs.Pbs(pbs_work_dir,
                               job_count=num_jobs,
                               qsub=self.pbs_config['qsub'],
                               clean=clean)
        self.pbs_obj.pbs_common_setting(**self.pbs_config)

    def setup_config(self, n_levels, clean):
        """
        Simulation dependent configuration
        :param n_levels: Number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.MLMC instance
        """
        # Set pbs config, flow123d, gmsh, ...
        self.config_dict = load_config_dict(os.path.join(src_path, "config.yaml"))



        self.set_environment_variables()
        output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))

        reuse_samples = self.config_dict.get('reuse_samples', None)
        if reuse_samples:
            clean = False
            self.config_dict['metacentrum'] = False
            self.config_dict['finish_sleep'] = 0

        # remove existing files
        if clean:
            self.rm_files(output_dir)

        if src_path != self.work_dir:
            for f in ["config.yaml", "01_hm_tmpl.yaml", "02_th_tmpl.yaml", "03_th_tmpl.yaml", "process.py"]:
                orig = os.path.join(src_path, f)
                work = os.path.join(self.work_dir, f)
                if clean or not os.path.isfile(work):
                    shutil.copy(orig, work)

        # Init pbs object
        self.create_pbs_object(output_dir, clean)
        # if (self.config_dict["metacentrum"]):
        #
        # else:
        #     self.pbs_obj = None

        simulation_config = {
            'env': dict(flow123d=self.flow123d, pbs=self.pbs_obj),  # The Environment.
            'output_dir': output_dir,
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
        }

        RandomFracSimulation.total_sim_id = 0

        self.options['output_dir'] = output_dir
        mlmc_obj = mlmc.MLMC(n_levels, RandomFracSimulation.factory(self.step_range, config=simulation_config, clean=clean),
                                  self.step_range, self.options)

        if clean or reuse_samples:
            # Create new execution of mlmc
            mlmc_obj.create_new_execution()
        else:
            # Use existing mlmc HDF file
            mlmc_obj.load_from_file()
        return mlmc_obj


    def run(self):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)
        self.n_moments = 10

        mlmc_list = []
        # Run one level Monte-Carlo method
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=True)

            # self.n_sample_estimate(mlmc)
            self.generate_jobs(mlmc, n_samples=[self.mc_samples])
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)



    def process(self):
        """
        Use collected data
        :return: None
        """
        assert os.path.isdir(self.work_dir)
        mlmc_est_list = []

        for nl in [1]:  # high resolution fields
            mlmc = self.setup_config(nl, clean=False)
            # Use wrapper object for working with collected data
            mlmc_est = Estimate(mlmc)
            mlmc_est_list.append(mlmc_est)

        # self.result_text(mlmc)
        # self.plot_density(mlmc)

        mlmc_est.mlmc.clean_select()

        th_02_model_params = ["power", "temp", "temp_min", "temp_max"]
        th_03_model_params = [n + "_ref" for n in th_02_model_params]
        print("02_th - stimulated EGS")
        good_mask = self.select_samples_with_good_values(mlmc_est, th_02_model_params)
        print("03_th - reference EGS")
        good_mask = self.select_samples_with_good_values(mlmc_est, th_03_model_params, good_mask=good_mask)
        #good_mask[np.random.choice(len(good_mask), int(len(good_mask)*0.93))] = False
        #print("N Random choice: ", sum(good_mask))
        mlmc_est.mlmc.subsample_by_indices(good_mask)

        # self.plot_temp_power(mlmc_est, th_02_model_params)
        # self.plot_temp_power(mlmc_est, th_03_model_params)

        #self.plot_histogram_const(mlmc_est, 'fr_cs_med', "fr_cs_median", "Median of $\delta_f$ [mm]", bins=30, scale=1000)
        #self.plot_histogram_const(mlmc_est, 'fr_cond_med', "fr_cond_median", "Median of $k_f$ [m.s$^{-1}$]", bins=30, scale=1)
        #self.plot_histogram_flux(mlmc_est, 'bc_flux', "total_flux_histogram", bins=30)

        self.plot_histogram(mlmc_est, 'temp', "temp_histogram", "Temperature [$^\circ$C]", bins=30)
        self.plot_histogram(mlmc_est, 'power', "power_histogram", "Power [MW]", bins=30)

        self.plot_temp_ref_comparison(mlmc_est)
        self.plot_power_ref_comparison(mlmc_est)

        # n_samples = int(mlmc_est.mlmc.n_samples)
        # print("N samples: ", n_samples)
        # temp_v_ele = np.zeros((2 * n_samples, 2 * n_samples), dtype=int)
        # print("good temperature interval: <{},{}>".format(MIN_T, MAX_T))
        # bad_temp_flag = np.zeros((len(n_bad_els),))
        # bad_ele_flag = np.zeros((len(n_bad_els),))
        # for i in range(0, n_samples):
        #     bad_temp_flag[i] = min(temp_min[i][0]) < MIN_T or max(temp_max[i][0]) > MAX_T
        #     bad_ele_flag[i] = n_bad_els[i][0][0] > 0
        #     ty = i if bad_temp_flag[i] else n_samples + i
        #     tx = i if bad_ele_flag[i] else n_samples + i
        #     temp_v_ele[ty, tx] = 1
        #     [np.sum(temp_v_ele[:n_samples, :n_samples]),
        #      np.sum(temp_v_ele[:n_samples, n_samples:2 * n_samples])],
        #     [np.sum(temp_v_ele[n_samples:2 * n_samples, :n_samples]),
        #      np.sum(temp_v_ele[n_samples:2 * n_samples, n_samples:2 * n_samples])]]
        #
        # print("temp[BAD, GOOD]' x mesh[BAD, GOOD]]\n{}\n{}".format(temp_v_ele_sum[0], temp_v_ele_sum[1]))
        #
        # # print(n_bad_els.shape)
        # n_bad_elements = np.zeros((len(n_bad_els),))
        # for i in range(len(n_bad_els)):
        #     n_bad_elements[i] = n_bad_els[i][0][0]
        #
        # print("min n_bad_element = ", min(n_bad_elements))
        # print("max n_bad_element = ", max(n_bad_elements))
        #
        # # compute EX and varX
        # bad_temp_ele_avg = 0    # EX of bad elements for bad temperature
        # good_temp_ele_avg = 0   # EX of bad elements for good temperature
        # for i in range(len(n_bad_els)):
        #     if bad_temp_flag[i]:
        #         bad_temp_ele_avg += n_bad_elements[i]
        #     else:
        #         good_temp_ele_avg += n_bad_elements[i]
        # if temp_v_ele_sum[0][0] != 0:
        #     bad_temp_ele_avg /= temp_v_ele_sum[0][0]
        # if temp_v_ele_sum[1][0] != 0:
        #     good_temp_ele_avg /= temp_v_ele_sum[1][0]
        # print("bad_temp EX: ", bad_temp_ele_avg)
        # print("good_temp EX: ", good_temp_ele_avg)
        #
        # bad_temp_ele_var = 0    # varX of bad elements for bad temperature
        # good_temp_ele_var = 0   # varX of bad elements for good temperature
        # for i in range(len(n_bad_els)):
        #     if bad_temp_flag[i]:
        #         bad_temp_ele_var += (n_bad_elements[i]-bad_temp_ele_avg)**2
        #     else:
        #         good_temp_ele_var += (n_bad_elements[i]-good_temp_ele_avg)**2
        # if temp_v_ele_sum[0][0] != 0:
        #     bad_temp_ele_var /= temp_v_ele_sum[0][0]
        # if temp_v_ele_sum[1][0] != 0:
        #     good_temp_ele_var /= temp_v_ele_sum[1][0]
        # print("bad_temp varX: ", bad_temp_ele_var)
        # print("good_temp varX: ", good_temp_ele_var)
        # print("bad_temp s: ", np.sqrt(bad_temp_ele_var))
        # print("good_temp s: ", np.sqrt(good_temp_ele_var))
        #
        # # print("temp_v_ele:\n", temp_v_ele)
        # temp_v_ele_sum = [
        #     [np.sum(temp_v_ele[:n_samples, :n_samples]),
        #      np.sum(temp_v_ele[:n_samples, n_samples:2 * n_samples])],
        #     [np.sum(temp_v_ele[n_samples:2 * n_samples, :n_samples]),
        #      np.sum(temp_v_ele[n_samples:2 * n_samples, n_samples:2 * n_samples])]]
        #
        # print("temp[BAD, GOOD]' x mesh[BAD, GOOD]]\n{}\n{}".format(temp_v_ele_sum[0], temp_v_ele_sum[1]))
        #
        # # print(n_bad_els.shape)
        # n_bad_elements = np.zeros((len(n_bad_els),))
        # for i in range(len(n_bad_els)):
        #     n_bad_elements[i] = n_bad_els[i][0][0]
        #
        # print("min n_bad_element = ", min(n_bad_elements))
        # print("max n_bad_element = ", max(n_bad_elements))
        #
        # # compute EX and varX
        # bad_temp_ele_avg = 0    # EX of bad elements for bad temperature
        # good_temp_ele_avg = 0   # EX of bad elements for good temperature
        # for i in range(len(n_bad_els)):
        #     if bad_temp_flag[i]:
        #         bad_temp_ele_avg += n_bad_elements[i]
        #     else:
        #         good_temp_ele_avg += n_bad_elements[i]
        # bad_temp_ele_avg /= temp_v_ele_sum[0][0]
        # good_temp_ele_avg /= temp_v_ele_sum[1][0]
        # print("bad_temp EX: ", bad_temp_ele_avg)
        # print("good_temp EX: ", good_temp_ele_avg)
        #
        # bad_temp_ele_var = 0    # varX of bad elements for bad temperature
        # good_temp_ele_var = 0   # varX of bad elements for good temperature
        # for i in range(len(n_bad_els)):
        #     if bad_temp_flag[i]:
        #         bad_temp_ele_var += (n_bad_elements[i]-bad_temp_ele_avg)**2
        #     else:
        #         good_temp_ele_var += (n_bad_elements[i]-good_temp_ele_avg)**2
        # bad_temp_ele_var /= temp_v_ele_sum[0][0]
        # good_temp_ele_var /= temp_v_ele_sum[1][0]
        # print("bad_temp varX: ", bad_temp_ele_var)
        # print("good_temp varX: ", good_temp_ele_var)
        # print("bad_temp s: ", np.sqrt(bad_temp_ele_var))
        # print("good_temp s: ", np.sqrt(good_temp_ele_var))
        #
        # # print("n_bad_els:\n", n_bad_els)
        #
        # # print("TEMP_MIN: ", temp_min)
        # # print("TEMP_MAX: ", temp_max)
        # #
        # # print("TEMP_MIN: ", min(temp_min[0][0]))
        # # print("TEMP_MAX: ", max(temp_max[1][0]))

        print("PROCESS FINISHED :)")

        # self.process_analysis(cl)

    def get_samples(self, mlmc_est, quantity):
        mlmc_est.mlmc.select_values(None, selected_param=quantity)
        q_array = mlmc_est.mlmc.levels[0].sample_values
        mlmc_est.mlmc.clean_select()
        return q_array[:,0,:]


    def select_samples_with_good_values(self, mlmc_est, result_params, good_mask=None):
        """
        Selects samples according to the temperature, power, mesh results.

        :param result_params : Dictionary of names of parameters {power :, temp: , temp_min: , temp_max: }
        :return: None
        """
        # determine samples with correct temperature
        power, temp, temp_min, temp_max = result_params
        n_bad_els = self.get_samples(mlmc_est, 'n_bad_els')
        temp_min = self.get_samples(mlmc_est, temp_min)
        temp_max = self.get_samples(mlmc_est, temp_max)
        temp = self.get_samples(mlmc_est, temp)
        power = self.get_samples(mlmc_est, power)

        abs_zero_temp = 273.15
        MIN_T = 273.15
        MAX_T = 470
        min_power= 0
        max_power= 50
        temp_not_nan = ~np.any(np.isnan(temp), axis=1)
        temp_good_min = np.min(temp_min[:, :], axis=1) > MIN_T
        temp_good_max = np.max(temp_max[:, :], axis=1) < MAX_T
        temp_good = np.logical_and(np.min(temp[:, :], axis=1) > MIN_T - 273.15,
                                   np.max(temp[:, :], axis=1) < MAX_T - 273.15)
        power_good = np.logical_and(np.min(power, axis=1) > min_power,
                                    np.max(power, axis=1) < max_power)
        no_bad_els = n_bad_els[:, 0] == 0
        N = mlmc_est.mlmc.n_samples[0]
        print("  N all samples: ", N)

        if good_mask is None:
            good_mask = np.full_like(power_good, True)
        good_mask = np.logical_and(good_mask, temp_not_nan)
        N_old, N = N, np.sum(good_mask)
        print("  N not nan: {} removed: {}".format(N, N_old - N))

        good_mask = np.logical_and(good_mask, temp_good_max)
        good_mask = np.logical_and(good_mask, temp_good_min)
        good_mask = np.logical_and(good_mask, temp_good)
        N_old, N = N, np.sum(good_mask)
        print("  N good temperature: {} removed: {}".format(N, N_old - N))

        good_mask = np.logical_and(good_mask, no_bad_els)
        N_old, N = N, np.sum(good_mask)
        print("  N good elements: {} removed: {}".format(N, N_old - N))

        good_mask = np.logical_and(good_mask, power_good)
        N_old, N = N, np.sum(good_mask)
        print("  N good power: {} removed: {}".format(N, N_old - N))

        good_indices = np.arange(0, len(good_mask))[good_mask]
        bad_indices = np.arange(0, len(good_mask))[~good_mask]
        assert not np.any(np.isnan(temp_max[good_indices]))
        assert not np.any(np.isnan(temp_min[good_indices]))
        assert not np.any(np.isnan(temp[good_indices]))

        return good_mask



    def plot_param(self, mlmc_est, ax, X, col, param_name, legend):
        """
        Get all result values by given param name, e.g. param_name = "temp" - return all temperatures...
        :param mlmc_est: Estimate instance
        :param param_name: Sample result param name
        :return: moments means, moments vars -> two numpy arrays
        """

        n_moments = 3
        mlmc_est.mlmc.clean_select()
        mlmc_est.mlmc.select_values(None, selected_param=param_name)
        print("plot param:", param_name)
        samples = mlmc_est.mlmc.levels[0].sample_values[:,0,:]
        print("    shape: ", samples.shape)
        #print(np.any(np.isnan(samples), axis=1))
        domain = Estimate.estimate_domain(mlmc_est.mlmc)
        domain_diff = domain[1] - domain[0]
        print("    domain: ", domain)
        moments_fn = Monomial(n_moments, domain, False, ref_domain=domain)
        N = mlmc_est.mlmc.n_samples[0]
        mom_means, mom_vars = mlmc_est.estimate_moments(moments_fn)

        mom_means, mom_vars = mom_means[1:, :], mom_vars[1:, :] # omit time=0
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


    def result_text(self, mlmc):
        for level in mlmc.levels:
            for f_sample, c_sample in level.collected_samples:
                print("Sample result data ", f_sample.result_data)
                print("Sample temp result", f_sample.result_data["temp"])

        mlmc.select_values(None, selected_param="temp")
        for l in mlmc.levels:
            print("Sample values ", l.sample_values)

    def plot_temp_power(self, mlmc_est, result_params):
        """
        Plot temperature and power
        :param mlmc_est: mlmc.Estimate instance
        :param result_params : Dictionary of names of parameters {power :, temp: , temp_min: , temp_max: }
        :return: None
        """
        times = np.average(self.get_samples(mlmc_est, 'value'), axis=0)[1:]

        # Plot temperature
        fig, ax1 = plt.subplots()
        temp_color = 'red'
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Temperature [$^\circ$C]', color=temp_color)
        ax1.tick_params(axis='y', labelcolor=temp_color)
        self.plot_param(mlmc_est, ax1, times, temp_color, result_params['temp'])

        # Plot power series
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        pow_color = 'blue'
        ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=pow_color)
        self.plot_param(mlmc_est, ax2, times, pow_color, result_params['power'])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("temp_power.pdf")
        plt.show()

    def plot_histogram(self, mlmc_est, quantity, name, ylabel, bins):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(ylabel)
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        times = np.average(self.get_samples(mlmc_est, 'value'), axis=0)[1:]
        t_min = 0
        t_max = int(times[-1])
        q = self.get_samples(mlmc_est, quantity)
        for t in [1, int((t_min + t_max)/2), t_max]:
            print(t)
            label = "t = " + str(t) + " y"
            ax1.hist(q[:, t], alpha=0.5, bins=bins, label=label) #edgecolor='white')

        ns = "N = " + str(mlmc_est.mlmc.n_samples[0])
        ax1.set_title(ns)
        ax1.legend()
        fig.savefig(os.path.join(self.work_dir, name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, name + ".png"))
        plt.show()

    def plot_histogram_const(self, mlmc_est, quantity, name, ylabel, bins=20, scale=1.0):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(ylabel)
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')

        q = self.get_samples(mlmc_est, quantity)

        ax1.hist(scale * q[:, 0], alpha=1.0, bins=bins) #edgecolor='white')

        ns = "N = " + str(mlmc_est.mlmc.n_samples[0])
        ax1.set_title(ns)
        ax1.legend()
        fig.savefig(os.path.join(self.work_dir, name + ".pdf"))
        fig.savefig(os.path.join(self.work_dir, name + ".png"))
        plt.show()

    def plot_histogram_flux(self, mlmc_est, quantity, name, bins=20):
        scale = -1000 # l.s^{-1}
        q_fr = self.get_samples(mlmc_est, quantity + "_fr")
        q_bulk = self.get_samples(mlmc_est, quantity + "_bulk")
        q_scaled = (q_fr[:, 0] + q_bulk[:, 0]) * scale

        q_fr_ref = self.get_samples(mlmc_est, quantity + "_fr_ref")
        q_bulk_ref = self.get_samples(mlmc_est, quantity + "_bulk_ref")
        q_ref_scaled = (q_fr_ref[:, 0] + q_bulk_ref[:, 0]) * scale

        # plot total flux
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel("Logarithm of total flux out [l$s^{-1}$]")
        ax1.set_ylabel('Frequency [-]')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.hist(np.log10(q_scaled), alpha=0.5, bins=bins, label="stimulated")
        ax1.hist(np.log10(q_ref_scaled), alpha=0.5, bins=bins, label="reference")
        # ax1.hist(np.log10(q_scaled), alpha=0.5, bins=bins, label="stimulated", density="true")
        # ax1.hist(np.log10(q_ref_scaled), alpha=0.5, bins=bins, label="reference", density="true")

        ns = "N = " + str(mlmc_est.mlmc.n_samples[0])
        ax1.set_title(ns)
        ax1.legend()
        fig1.savefig(os.path.join(self.work_dir, "total_" + name + ".pdf"))
        fig1.savefig(os.path.join(self.work_dir, "total_" + name + ".png"))
        plt.show()

        # plot flux ratio
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Ratio fr/bulk flux out [-]")
        ax2.set_ylabel('Frequency [-]')
        ax2.tick_params(axis='y', labelcolor='black')

        q_ratio = scale * q_fr[:, 0] / q_scaled
        q_ratio_ref = scale * q_fr_ref[:, 0] / q_ref_scaled
        ax2.hist(q_ratio, alpha=0.5, bins=bins, label="stimulated")
        ax2.hist(q_ratio_ref, alpha=0.5, bins=bins, label="reference")

        ns = "N = " + str(mlmc_est.mlmc.n_samples[0])
        ax2.set_title(ns)
        ax2.legend()
        fig2.savefig(os.path.join(self.work_dir, "ratio_" + name + ".pdf"))
        fig2.savefig(os.path.join(self.work_dir, "ratio_" + name + ".png"))
        plt.show()

    def plot_temp_ref_comparison(self, mlmc_est):
        """
        Plot temperature and power
        :param mlmc_est: mlmc.Estimate instance
        :param result_params : Dictionary of names of parameters {power :, temp: , temp_min: , temp_max: }
        :return: None
        """
        times = np.average(self.get_samples(mlmc_est, 'power_time'), axis=0)[1:]

        # Plot temperature
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Temperature [$^\circ$C]', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        self.plot_param(mlmc_est, ax1, times, 'red', 'temp', 'stimulated')
        self.plot_param(mlmc_est, ax1, times, 'orange', 'temp_ref', 'unmodified')

        #ax1.legend(["stimulated - mean", "stimulated - std", "reference - mean", "reference - std"])
        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, "temp_comparison.pdf"))
        fig.savefig(os.path.join(self.work_dir, "temp_comparison.png"))
        plt.show()

    def plot_power_ref_comparison(self, mlmc_est):
        """
        Plot temperature and power
        :param mlmc_est: mlmc.Estimate instance
        :param result_params : Dictionary of names of parameters {power :, temp: , temp_min: , temp_max: }
        :return: None
        """
        times = np.average(self.get_samples(mlmc_est, 'power_time'), axis=0)[1:]

        # Plot temperature
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Power [MW]', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        self.plot_param(mlmc_est, ax1, times, 'blue', 'power', 'stimulated')

        self.plot_param(mlmc_est, ax1, times, 'forestgreen', 'power_ref', 'unmodified')

        #ax1.legend(["stimulated - mean", "stimulated - std", "reference - mean", "reference - std"])
        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.work_dir, "power_comparison.pdf"))
        fig.savefig(os.path.join(self.work_dir, "power_comparison.png"))
        plt.show()


    def plot_density(self, mlmc):
        # for level in mlmc.levels:
        #     for f_sample, c_sample in level.collected_samples:
        #         print("f sample ", f_sample.result_data)
        mlmc.select_values({"power_time": (0, "=")}, selected_param="power")
        # for level in mlmc.levels:
        #     for f_sample, c_sample in level.collected_samples:
        #         print("f sample ", f_sample.result)

        cl = CompareLevels([mlmc],
                           output_dir=src_path,
                           quantity_name="Q [m/s]",
                           moment_class=Legendre,
                           log_scale=False,
                           n_moments=8, )
        cl.construct_densities()
        cl.plot_densities()


if __name__ == "__main__":
    process = Process()
