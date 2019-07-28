import sys
import os
import yaml
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


def load_config_dict():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


class Process(base_process.Process):
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

        # self.plot_temp_power(mlmc_est)

        mlmc_est.mlmc.clean_select()
        mlmc_est.mlmc.select_values({"n_bad_els": (-10, ">=")}, selected_param="n_bad_els")
        n_bad_els = mlmc_est.mlmc.levels[0].sample_values

        mlmc_est.mlmc.clean_select()
        # mlmc_est.mlmc.select_values(None, selected_param="temp_min")
        # mlmc_est.mlmc.select_values(None, selected_param="temp")
        mlmc_est.mlmc.select_values({"temp_min": (100000, "<=")}, selected_param="temp_min")
        temp_min = mlmc_est.mlmc.levels[0].sample_values

        mlmc_est.mlmc.clean_select()
        mlmc_est.mlmc.select_values({"temp_max": (-100000, ">=")}, selected_param="temp_max")
        temp_max = mlmc_est.mlmc.levels[0].sample_values

        n_samples = int(mlmc_est.mlmc.n_samples)
        print("N samples: ", n_samples)
        temp_v_ele = np.zeros((2 * n_samples, 2 * n_samples), dtype=int)
        MIN_T = 250
        MAX_T = 470
        print("goot temperature interval: <{},{}>".format(MIN_T, MAX_T))
        bad_temp_flag = np.zeros((len(n_bad_els),))
        bad_ele_flag = np.zeros((len(n_bad_els),))
        for i in range(0, n_samples):
            bad_temp_flag[i] = min(temp_min[i][0]) < MIN_T or max(temp_max[i][0]) > MAX_T
            bad_ele_flag[i] = n_bad_els[i][0][0] > 0
            ty = i if bad_temp_flag[i] else n_samples + i
            tx = i if bad_ele_flag[i] else n_samples + i
            temp_v_ele[ty, tx] = 1

        # print("temp_v_ele:\n", temp_v_ele)
        temp_v_ele_sum = [
            [np.sum(temp_v_ele[:n_samples, :n_samples]),
             np.sum(temp_v_ele[:n_samples, n_samples:2 * n_samples])],
            [np.sum(temp_v_ele[n_samples:2 * n_samples, :n_samples]),
             np.sum(temp_v_ele[n_samples:2 * n_samples, n_samples:2 * n_samples])]]

        print("temp[BAD, GOOD]' x mesh[BAD, GOOD]]\n{}\n{}".format(temp_v_ele_sum[0], temp_v_ele_sum[1]))

        # print(n_bad_els.shape)
        n_bad_elements = np.zeros((len(n_bad_els),))
        for i in range(len(n_bad_els)):
            n_bad_elements[i] = n_bad_els[i][0][0]

        print("min n_bad_element = ", min(n_bad_elements))
        print("max n_bad_element = ", max(n_bad_elements))

        # compute EX and varX
        bad_temp_ele_avg = 0    # EX of bad elements for bad temperature
        good_temp_ele_avg = 0   # EX of bad elements for good temperature
        for i in range(len(n_bad_els)):
            if bad_temp_flag[i]:
                bad_temp_ele_avg += n_bad_elements[i]
            else:
                good_temp_ele_avg += n_bad_elements[i]
        bad_temp_ele_avg /= temp_v_ele_sum[0][0]
        good_temp_ele_avg /= temp_v_ele_sum[1][0]
        print("bad_temp EX: ", bad_temp_ele_avg)
        print("good_temp EX: ", good_temp_ele_avg)

        bad_temp_ele_var = 0    # varX of bad elements for bad temperature
        good_temp_ele_var = 0   # varX of bad elements for good temperature
        for i in range(len(n_bad_els)):
            if bad_temp_flag[i]:
                bad_temp_ele_var += (n_bad_elements[i]-bad_temp_ele_avg)**2
            else:
                good_temp_ele_var += (n_bad_elements[i]-good_temp_ele_avg)**2
        bad_temp_ele_var /= temp_v_ele_sum[0][0]
        good_temp_ele_var /= temp_v_ele_sum[1][0]
        print("bad_temp varX: ", bad_temp_ele_var)
        print("good_temp varX: ", good_temp_ele_var)
        print("bad_temp s: ", np.sqrt(bad_temp_ele_var))
        print("good_temp s: ", np.sqrt(good_temp_ele_var))

        # print("n_bad_els:\n", n_bad_els)

        # print("TEMP_MIN: ", temp_min)
        # print("TEMP_MAX: ", temp_max)
        #
        # print("TEMP_MIN: ", min(temp_min[0][0]))
        # print("TEMP_MAX: ", max(temp_max[1][0]))

        print("PROCESS FINISHED :)")

        # self.process_analysis(cl)

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
        self.config_dict = load_config_dict()
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

        if clean:
            # Create new execution of mlmc
            mlmc_obj.create_new_execution()
        else:
            # Use existing mlmc HDF file
            mlmc_obj.load_from_file()
        return mlmc_obj

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

    def get_all_results_by_param(self, mlmc_est, param_name):
        """
        Get all result values by given param name, e.g. param_name = "temp" - return all temperatures...
        :param mlmc_est: Estimate instance
        :param param_name: Sample result param name
        :return: moments means, moments vars -> two numpy arrays
        """
        n_moments = 3
        mlmc_est.mlmc.clean_select()
        mlmc_est.mlmc.select_values({param_name: (0, ">=")})
        domain = Estimate.estimate_domain(mlmc_est.mlmc)
        moments_fn = Monomial(n_moments, domain, False, ref_domain=domain)
        means, vars = mlmc_est.estimate_moments(moments_fn)

        return means, vars

    def result_text(self, mlmc):
        for level in mlmc.levels:
            for f_sample, c_sample in level.collected_samples:
                print("Sample result data ", f_sample.result_data)
                print("Sample temp result", f_sample.result_data["temp"])

        mlmc.select_values(None, selected_param="temp")
        for l in mlmc.levels:
            print("Sample values ", l.sample_values)

    def plot_temp_power(self, mlmc_est):
        """
        Plot temperature and power
        :param mlmc_est: mlmc.Estimate instance
        :return: None
        """
        times_means, times_vars = self.get_all_results_by_param(mlmc_est, "value")
        temp_times = times_means[:, 1]

        # Temperature means and vars
        temp_means, temp_vars = self.get_all_results_by_param(mlmc_est, "temp")
        avg_temp = temp_means[:, 1]
        avg_temp_std = np.sqrt(temp_vars[:, 1])[1:]

        # Power means and vars
        power_means, power_vars = self.get_all_results_by_param(mlmc_est, "power")
        power_series = power_means[:, 1]
        power_series_std = np.sqrt(power_vars[:, 1])[1:]

        # power_time_means, power_time_vars = self.get_all_results_by_param(mlmc_est, "power_time")

        # Plot temperature
        fig, ax1 = plt.subplots()
        temp_color = 'red'
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Temperature [C deg]', color=temp_color)
        ydata = avg_temp[1:]
        ax1.plot(temp_times[1:], ydata, color=temp_color)
        ax1.fill_between(temp_times[1:], ydata - avg_temp_std, ydata + avg_temp_std,
                         color=temp_color, alpha=0.2)
        ax1.tick_params(axis='y', labelcolor=temp_color)

        # Plot power series
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        pow_color = 'blue'
        ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
        ydata = power_series[1:]
        ax2.plot(temp_times[1:], ydata, color=pow_color)

        # Shaded uncertainty region
        ax2.fill_between(temp_times[1:], ydata - power_series_std, ydata + power_series_std,
                         color=pow_color, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=pow_color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


if __name__ == "__main__":
    process = Process()
