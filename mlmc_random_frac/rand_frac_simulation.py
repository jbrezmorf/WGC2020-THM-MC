import os
import sys
import shutil
import yaml
import numpy as np
import time as t

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '../MLMC/src'))
sys.path.append(os.path.join(src_path, '../dfn/src'))

import mlmc.sample as sample
from mlmc.simulation import Simulation


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


def load_config_dict(cfg):
    with open(cfg, "r") as f:
        return yaml.safe_load(f)



class RandomFracSimulation(Simulation):

    def __init__(self, mesh_step, level_id=None, config=None, clean=False, parent_fine_sim=None):
        """

        :param config: configuration of the simulation, processed keys:
            env - Environment object.
            fields - FieldSet object
            yaml_file: Template for main input file. Placeholders:
                <mesh_file> - replaced by generated mesh
                <FIELD> - for FIELD be name of any of `fields`, replaced by the FieldElementwise field with generated
                 field input file and the field name for the component.
                 (TODO: allow relative paths, not tested but should work)
            geo_file: Path to the geometry file. (TODO: default is <yaml file base>.geo
        :param mesh_step: Mesh step, decrease with increasing MC Level.
        :param parent_fine_sim: Allow to set the fine simulation on previous level (Sim_f_l) which corresponds
        to 'self' (Sim_c_l+1) as a coarse simulation. Usually Sim_f_l and Sim_c_l+1 are same simulations, but
        these need to be different for advanced generation of samples (zero-mean control and antithetic).
        """
        if level_id is not None:
            self.sim_id = level_id
        else:
            self.sim_id = RandomFracSimulation.total_sim_id
            RandomFracSimulation.total_sim_id += 1

        self.env = config['env']
        self.step = mesh_step
        # Pbs script creater
        self.pbs_creater = self.env["pbs"]

        # @TODO: set n fine elements
        self.n_fine_elements = 0

        # Prepare base workdir for this mesh_step
        output_dir = config['output_dir']


        self.work_dir = os.path.join(output_dir, 'sim_%d_step_%f' % (self.sim_id, self.step))

        force_mkdir(self.work_dir, clean)

        self.coarse_sim = None
        self.coarse_sim_set = False
        self.pbs_script = []

        # Auxiliary param for extracting results
        self.previous_length = 0

        root_dir, _ = os.path.split(output_dir)
        self.process_dir = os.path.join(src_path, root_dir)
        self.config_dict = load_config_dict(os.path.join(self.process_dir, "config.yaml"))
        super(Simulation, self).__init__()

    def n_ops_estimate(self):
        """
        Number of operations
        :return: int
        """
        return self.n_fine_elements

    def _make_mesh(self, geo_file, mesh_file):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """
        pass

    def set_coarse_sim(self, coarse_sim=None):
        """
        Set coarse simulation ot the fine simulation so that the fine can generate the
        correlated input data sample for both.

        Here in particular set_points to the field generator
        :param coarse_sim
        """
        self.coarse_sim = coarse_sim
        self.coarse_sim_set = True

    def generate_random_sample(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        pass

    def simulation_sample(self, sample_tag, sample_id, start_time=0):
        """
        Sample generating
        :param sample_tag: Simulation sample string tag
        :param sample_id: Sample unique identifier
        :param start_time: When the sample creation was begun
        :return: Sample instance
        """
        # Get sample directory
        out_subdir = os.path.join("samples", str(sample_tag))
        sample_dir = os.path.join(self.work_dir, out_subdir)
        reuse_samples = self.config_dict.get('reuse_samples', None)
        if reuse_samples is None:
            force_mkdir(sample_dir, True)
        else:
            force_mkdir(sample_dir, False)

        # Run part which can be run via pbs
        self.create_pbs_script(sample_dir)

        # Pbs package directory
        package_dir = self.run_sim_sample(out_subdir)
        # if reuse_samples:
        #     self.pbs_creater._number_of_realizations = 0
        # else:
        self.pbs_creater.execute()

        return sample.Sample(directory=sample_dir, sample_id=sample_id,
                             job_id=package_dir, prepare_time=(t.time() - start_time))

    def create_pbs_script(self, sample_dir):
        """
        Create pbs script
        :param sample_dir: Sample dir abs path
        :return: None
        """
        finish_sleep = self.config_dict.get("finish_sleep", 30)
        self.pbs_script.append(
            """
            cd {script_dir}
            source load_modules.sh
            source env/bin/activate
            python {abs_proc_dir}/process.py {sample_dir} >{sample_dir}/STDOUT 2>&1
            sleep {finish_sleep}
            echo "done" >{sample_dir}/FINISHED
            """
            .format(script_dir=src_path, abs_proc_dir=self.process_dir, sample_dir=sample_dir, finish_sleep=finish_sleep))


    def run_sim_sample(self, out_subdir):
        """
        Add simulations realization to pbs file
        :param out_subdir: MLMC output directory
        :return: Package directory (directory with pbs job data)
        """
        # Add flow123d realization to pbs script
        package_dir = self.pbs_creater.add_realization(self.n_fine_elements, self.pbs_script,
                                                       output_subdir=out_subdir,
                                                       work_dir=self.work_dir,
                                                       flow123d=self.config_dict["flow_executable"])

        self.pbs_script = []
        return package_dir

    def _extract_time_series(self, yaml_stream, regions, extract):
        """
        :param yaml_stream:
        :param regions:
        :return: times list, list: for every region the array of value series
        """
        content = yaml.safe_load(yaml_stream)
        if content is None:
            return None, None

        data = content['data']
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
        return np.array(times, dtype=float), series

    def _extract_result(self, sample):
        """
        :param config_dict: Parsed config.yaml. see key comments there.
        : return
        """
        abs_zero_temp = 273.15
        year_sec = 60 * 60 * 24 * 365
        # Sample result structure -> you can enlarge it and other stuff will be done automatically
        self.result_struct = [["value", "power_time",
                               "power", "temp", "fluxes", "temp_min", "temp_max",
                               "power_ref", "temp_ref", "fluxes_ref", "temp_min_ref", "temp_max_ref",
                               "n_bad_els"],
                              ["f8", "f8",
                               "f8", "f8", "f8", "f8",  "f8",
                               "f8", "f8", "f8", "f8", "f8",
                               "f8"]]

        sample_dir = sample.directory
        heal_stat_file = os.path.join(sample_dir, "random_fractures_heal_stats.yaml")
        finished_file = os.path.join(sample_dir, "FINISHED")

        finished = False
        if os.path.exists(finished_file):
            with open(finished_file, "r") as f:
                content = f.read().split()                
            finished = len(content)==1 and content[0] == "done"

        if finished:
            print(sample_dir, "Finished")
            finished_map = {f:os.path.exists(f) for f in [finished_file, heal_stat_file]}
            files_exist = all(finished_map.values())
            if files_exist:
                print(sample_dir, "Files exist")
                # Sometimes content of files is not complete, sleep() seems to be workaround
                # if self.previous_length == 0:
                #     t.sleep(60)

                with open(heal_stat_file, "r") as f:
                    stat_doc = yaml.safe_load(f)
                    n_bad_els = len(stat_doc["flow_stats"]["bad_elements"]) + len(stat_doc["gamma_stats"]["bad_elements"])

                power_times, th_result_values = self._extract_result_th(os.path.join(sample_dir, "output_02_th"))
                power_times, th_ref_result_values = self._extract_result_th(os.path.join(sample_dir, "output_03_th"))

                if power_times[0] == np.inf:
                    return [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

                if self.previous_length == 0:
                    self.previous_length = len(power_times)

                result_values = []
                for i in range(len(power_times)):
                    result_values.append((i, power_times[i], *(th_result_values[i]), *(th_ref_result_values[i]), n_bad_els))
                
                return result_values
            else:
                print(sample_dir, "missing files", finished_map)
                return [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        else:
            return [None, None, None, None, None, None, None]

    def _extract_result_th(self, th_outdir):

        abs_zero_temp = 273.15
        year_sec = 60 * 60 * 24 * 365

        water_balance_file = os.path.join(th_outdir, "water_balance.yaml")
        energy_balance_file = os.path.join(th_outdir, "energy_balance.yaml")
        heat_region_stat = os.path.join(th_outdir, "Heat_AdvectionDiffusion_region_stat.yaml")

        finished_map = {f: os.path.exists(f) for f in
                        [water_balance_file, energy_balance_file, heat_region_stat]}
        files_exist = all(finished_map.values())
        if files_exist:
            print(th_outdir, "Files exist")

            bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
            regions = ['fr', 'box']
            out_regions = bc_regions[2:]

            with open(energy_balance_file, "r") as f:
                power_times, reg_powers = self._extract_time_series(f, bc_regions,
                                                                    extract=lambda frame: frame['data'][0])
            with open(heat_region_stat, "r") as f:
                temp_times, reg_temps = self._extract_time_series(f, out_regions,
                                                                  extract=lambda frame: frame['average'][0])
            with open(heat_region_stat, "r") as f:
                temp_min_times, reg_temp_min = self._extract_time_series(f, regions,
                                                                         extract=lambda frame: frame['min'][0])
            with open(heat_region_stat, "r") as f:
                temp_max_times, reg_temp_max = self._extract_time_series(f, regions,
                                                                         extract=lambda frame: frame['max'][0])
            with open(water_balance_file, "r") as f:
                flux_times, reg_fluxes = self._extract_time_series(f, out_regions,
                                                                   extract=lambda frame: frame['data'][0])

            power_series = -sum(reg_powers)

            sum_flux = sum(reg_fluxes)
            avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux

            power_series = power_series / 1e6
            power_times = power_times / year_sec
            avg_temp = avg_temp - abs_zero_temp

            # compute min and max temperature over regions
            temp_min = np.full(len(power_times), 1e10)
            temp_max = np.full(len(power_times), -1e10)
            for j in range(0, len(power_times)):
                for i in range(0, len(regions)):
                    temp_min[j] = min([temp_min[j], reg_temp_min[i][j]])
                    temp_max[j] = max([temp_max[j], reg_temp_max[i][j]])

            th_result_values = []
            for i in range(len(power_times)):
                th_result_values.append((power_series[i], avg_temp[i], sum_flux, temp_min[i], temp_max[i]))

            return power_times, th_result_values
        else:
            print(th_outdir, "missing files", finished_map)
            return [np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]
