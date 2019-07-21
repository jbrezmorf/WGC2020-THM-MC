import os
import sys
import shutil
import yaml
import numpy as np
import time as t
import copy
sys.path.append('../MLMC/src')
import gmsh_io as gmsh_io
import mlmc.sample as sample
from mlmc.simulation import Simulation


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


def load_config_dict():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


class FracSimulation(Simulation):
    HM_YAML_TEMPLATE = '01_hm_tmpl.yaml'
    TH_YAML_TEMPLATE = '02_th_tmpl.yaml'

    HM_YAML_FILE = "01_hm.yaml"
    TH_YAML_FILE = "02_th.yaml"

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
            self.sim_id = FracSimulation.total_sim_id
            FracSimulation.total_sim_id += 1

        self.env = config['env']
        self.step = mesh_step
        # Pbs script creater
        self.pbs_creater = self.env["pbs"]

        # @TODO: set n fine elements
        self.n_fine_elements = 0

        # Prepare base workdir for this mesh_step
        output_dir = config['output_dir']
        self.process_dir = os.path.split(output_dir)[0]
        self.work_dir = os.path.join(output_dir, 'sim_%d_step_%f' % (self.sim_id, self.step))
        force_mkdir(self.work_dir, clean)

        # Copy yaml templates to work directory
        self.hm_tmpl_yaml_file = os.path.join(self.work_dir, self.HM_YAML_TEMPLATE)
        self.th_tmpl_yaml_file = os.path.join(self.work_dir, self.TH_YAML_TEMPLATE)
        shutil.copyfile(self.HM_YAML_TEMPLATE, self.hm_tmpl_yaml_file)
        shutil.copyfile(self.TH_YAML_TEMPLATE, self.th_tmpl_yaml_file)

        self.coarse_sim = None
        self.coarse_sim_set = False
        self.pbs_script = []

        # Auxiliary param for extracting results
        self.previous_length = 0

        self.config_dict = load_config_dict()
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

    def compute_hm(self, sample_dir):
        """
        :param sample_dir: Sample directory path
        :return: None
        """

        # Copy yaml file to sample directory
        hm_yaml_file = os.path.join(self.process_dir, self.HM_YAML_FILE)
        sample_yaml_file = os.path.join(sample_dir, self.HM_YAML_FILE)
        shutil.copyfile(hm_yaml_file, sample_yaml_file)

        # @TODO: remove on production
        # Set random conductivity
        self.config_dict["th_params"]["bulk_conductivity"] = self.config_dict["hm_params"]["bulk_conductivity"]\
            = self.random_conductivity()

        substitute_placeholders(self.hm_tmpl_yaml_file, sample_yaml_file, self.config_dict["hm_params"])
        arguments = copy.deepcopy(self.config_dict["flow_executable"])

        arguments.extend(['--output_dir', os.path.join(sample_dir, 'output_hm'),
                          os.path.join(sample_dir, self.HM_YAML_FILE)])
        print("Running: ", " ".join(arguments))

        self.pbs_script.append(" ".join(arguments))
        self.pbs_script.append('touch {}/FINISHED'.format(sample_dir))

    def random_conductivity(self):
        from scipy.stats import lognorm
        s = 1
        return lognorm.rvs(s, loc=0, scale=0.0000001, size=1)[0]

    def compute_th(self, sample_dir):
        """
        :param sample_dir: Sample directory abs path
        :return: None
        """
        # Copy yaml file to sample directory
        th_yaml_file = os.path.join(self.process_dir, self.TH_YAML_FILE)
        sample_yaml_file = os.path.join(sample_dir, self.TH_YAML_FILE)
        shutil.copyfile(th_yaml_file, sample_yaml_file)

        substitute_placeholders(self.th_tmpl_yaml_file, sample_yaml_file, self.config_dict["th_params"])
        # Flow executable from config
        arguments = copy.deepcopy(self.config_dict["flow_executable"])

        arguments.extend(['--output_dir', os.path.join(sample_dir, 'output_th'),
                          os.path.join(sample_dir, self.TH_YAML_FILE)])

        # Append arguments to pbs script
        self.pbs_script.append('cd {}'.format(sample_dir))
        self.pbs_script.append(" ".join(arguments))
        self.pbs_script.append('touch {}/FINISHED'.format(sample_dir))

    # def get_result_description(self):
    #     """
    #     :return:
    #     """
    #     end_time = 30
    #     values = [[ValueDesctription(time=t, position="extraction_well", quantity="power", unit="MW"),
    #                ValueDesctription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
    #                ] for t in np.linspace(0, end_time, 0.1)]
    #     power_series, temp_series = zip(*values)
    #     return power_series + temp_series

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
        force_mkdir(sample_dir, True)

        # Copy mesh files to sample directory
        mesh_file_1 = os.path.join(self.process_dir, self.config_dict["hm_params"]["mesh"])
        shutil.copyfile(mesh_file_1, os.path.join(sample_dir, self.config_dict["hm_params"]["mesh"]))
        mesh_file_2 = os.path.join(self.process_dir, self.config_dict["th_params"]["mesh"])
        shutil.copyfile(mesh_file_2, os.path.join(sample_dir, self.config_dict["th_params"]["mesh"]))

        # Run part which can be run via pbs
        self.create_pbs_script(sample_dir)

        # Pbs package directory
        package_dir = self.run_sim_sample(out_subdir)

        return sample.Sample(directory=sample_dir, sample_id=sample_id,
                             job_id=package_dir, prepare_time=(t.time() - start_time))

    def create_pbs_script(self, sample_dir):
        """
        Create pbs script
        :param sample_dir: Sample dir abs path
        :return: None
        """
        self.compute_hm(sample_dir)
        # All auxiliary methods must be run in pbs script
        self.pbs_script.append("python3 {}/prepare_th_input.py {}".format(self.process_dir, sample_dir)) #causes some errors
        self.compute_th(sample_dir)

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
        series = [np.array(region_series) for region_series in reg_series.values()]
        return np.array(times), series

    def _extract_result(self, sample):
        """
        :param config_dict: Parsed config.yaml. see key comments there.
        : return
        """
        abs_zero_temp = 273.15
        year_sec = 60 * 60 * 24 * 365
        # Sample result structure -> you can enlarge it and other stuff will be done automatically
        self.result_struct = [["value", "power", "temp", "power_time"], ["f8", "f8", "f8", "f8"]]

        sample_dir = sample.directory

        water_balance_file = os.path.join(sample_dir, "output_th/water_balance.yaml")
        energy_balance_file = os.path.join(sample_dir, "output_th/energy_balance.yaml")
        heat_region_stat = os.path.join(sample_dir, "output_th/Heat_AdvectionDiffusion_region_stat.yaml")

        if os.path.exists(os.path.join(sample_dir, "FINISHED")):
            if os.path.exists(water_balance_file) and os.path.exists(energy_balance_file) and os.path.exists(heat_region_stat):
                # Sometimes content of files is not complete, sleep() seems to be workaround
                if self.previous_length == 0:
                    t.sleep(60)

                while True:
                    # extract the flux
                    bc_regions = ['.left_fr_left_well', '.left_well', '.right_fr_right_well', '.right_well']
                    out_regions = bc_regions[2:]

                    with open(energy_balance_file, "r") as f:
                        power_times, reg_powers = self._extract_time_series(f, bc_regions,
                                                                            extract=lambda frame: frame['data'][0])

                    with open(heat_region_stat, "r") as f:
                        temp_times, reg_temps = self._extract_time_series(f, out_regions,
                                                                          extract=lambda frame: frame['average'][0])
                    with open(water_balance_file, "r") as f:
                        flux_times, reg_fluxes = self._extract_time_series(f, out_regions,
                                                                           extract=lambda frame: frame['data'][0])

                    if power_times is None or temp_times is None or flux_times is None:
                        t.sleep(15)
                        continue

                    power_series = -sum(reg_powers)

                    sum_flux = sum(reg_fluxes)
                    avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux

                    power_series = power_series / 1e6
                    power_times = power_times / year_sec
                    avg_temp = avg_temp - abs_zero_temp

                    if not (len(power_series) == len(power_times) == len(avg_temp)):
                        t.sleep(15)
                    elif self.previous_length > len(power_series):
                        t.sleep(15)
                    else:
                        break

                if self.previous_length == 0:
                    self.previous_length = len(power_times)

                result_values = []
                for i in range(len(power_times)):
                    result_values.append((i, power_series[i], avg_temp[i], power_times[i]))

                return result_values
            else:
                return [np.inf, np.inf, np.inf, np.inf]

        else:
            return [None, None, None, None]
