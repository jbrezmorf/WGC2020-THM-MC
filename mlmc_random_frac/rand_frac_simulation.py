import os
import sys
import shutil
import yaml
import numpy as np
import time as t
from typing import Any, List
import attr

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

@attr.s(auto_attribs=True)
class Quantity:
    name: str     
    file: str 
    extractor: Any  # function, returning time series
    args: List[Any]
    np_type: str = "f8"
    value: Any = None
    
    


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


    def _extract_time_series(self, fname, regions, key, index):
        """
        :param yaml_stream:
        :param regions:
        :return: times list, list: for every region the array of value series
        """
        with open(fname, "r") as f:
            content = yaml.safe_load(f)
        if content is None:
            return None

        data = content['data']
        reg_series = {reg: [] for reg in regions}

        for time_data in data:
            region = time_data['region']
            if region in reg_series:
                time_data_key = time_data[key]
                if type(time_data_key) is not list:
                    time_data_key = [time_data_key]
                reg_series[region].append(time_data_key[index])        
        series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
        return series

    def extract_yaml_const(self, f, *keys):
        with open(f, "r") as f:
            content = yaml.safe_load(f)
            for k in keys:
                content = content[k]
            assert np.isscalar(content)
            return content

    def get_heal_stat(self, fname):
        with open(fname, "r") as f:
            content = yaml.safe_load(f)
            return  len(content["flow_stats"]["bad_elements"]) + len(content["gamma_stats"]["bad_elements"])


    def compute_th_quantities(self, power, temp, flux, t_min, t_max):
        """
        Every q.value is list of time series np.array for every region, we perform sort of integration over regions to
        obtain just scalar time series.
        """
        sum_power = -np.sum(power.value, axis=0) / 1e6
        sum_flux = np.sum(flux.value, axis=0)

        avg_temp = np.sum(np.array(flux.value) * np.array(temp.value), axis=0) / sum_flux
        abs_zero_temp = 273.15
        avg_temp = avg_temp - abs_zero_temp

        temp_min = np.min(np.array(t_min.value), axis=0) # min over regions
        temp_max = np.max(np.array(t_max.value), axis=0)

        power.value = sum_power
        temp.value = avg_temp
        flux.value = sum_flux
        t_min.value = temp_min
        t_max.value = temp_max

    def get_fr_param(self, f, ):
        with open(f, "r") as f:
            stat_doc = yaml.safe_load(f)

    def manipulate_quantities(self, q_dict):
        # manipulate values
        year_sec = 60 * 60 * 24 * 365
        q_dict['power_time'].value =  np.array(q_dict['power_time'].value)[0] / year_sec
        self.compute_th_quantities(q_dict['power'], q_dict['temp'], q_dict['fluxes'], q_dict['temp_min'],
                                   q_dict['temp_max'])
        self.compute_th_quantities(q_dict['power_ref'], q_dict['temp_ref'], q_dict['fluxes_ref'],
                                   q_dict['temp_min_ref'], q_dict['temp_max_ref'])

    def define_quantities(self):
        bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
        regions = ['fr', 'box']
        out_regions = bc_regions[2:]

        wb_file = "water_balance.yaml"
        eb_file = "energy_balance.yaml"
        heat_file = "Heat_AdvectionDiffusion_region_stat.yaml"
        heal_file = "random_fractures_heal_stats.yaml"
        fr_param_file = "fr_param_output.yaml"
        extract_series = self._extract_time_series

        th_dir = "output_02_th/"
        ref_dir = "output_03_th/"

        quantities = [
            Quantity("power_time", ref_dir + eb_file, extract_series, [['ALL'], 'time', 0]),

            Quantity("power", th_dir + eb_file, extract_series, [bc_regions, 'data', 0]),
            Quantity("temp", th_dir + heat_file, extract_series, [out_regions, 'average', 0]),
            Quantity("temp_min", th_dir + heat_file, extract_series, [regions, 'min', 0]),
            Quantity("temp_max", th_dir + heat_file, extract_series, [regions, 'max', 0]),
            Quantity("fluxes", th_dir + wb_file, extract_series, [out_regions, 'data', 0]),
            Quantity("bc_flux_bulk", th_dir + wb_file, extract_series, [['.right_well'], 'data', 0]),
            Quantity("bc_flux_fr", th_dir + wb_file, extract_series, [['.fr_right_well'], 'data', 0]),
            Quantity("bc_influx_bulk", ref_dir + wb_file, extract_series, [['.left_well'], 'data', 0]),
            Quantity("bc_influx_fr", ref_dir + wb_file, extract_series, [['.fr_left_well'], 'data', 0]),

            Quantity("power_ref", ref_dir + eb_file, extract_series, [bc_regions, 'data', 0]),
            Quantity("temp_ref", ref_dir + heat_file, extract_series, [out_regions, 'average', 0]),
            Quantity("temp_min_ref", ref_dir + heat_file, extract_series, [regions, 'min', 0]),
            Quantity("temp_max_ref", ref_dir + heat_file, extract_series, [regions, 'max', 0]),
            Quantity("fluxes_ref", ref_dir + wb_file, extract_series, [out_regions, 'data', 0]),
            Quantity("bc_flux_bulk_ref", ref_dir + wb_file, extract_series, [['.right_well'], 'data', 0]),
            Quantity("bc_flux_fr_ref", ref_dir + wb_file, extract_series, [['.fr_right_well'], 'data', 0]),
            Quantity("bc_influx_bulk_ref", ref_dir + wb_file, extract_series, [['.left_well'], 'data', 0]),
            Quantity("bc_influx_fr_ref", ref_dir + wb_file, extract_series, [['.fr_left_well'], 'data', 0]),

            Quantity("fr_cs_med", fr_param_file, self.extract_yaml_const, ["fr_cross_section", "median"]),
            Quantity("fr_cs_iqr", fr_param_file, self.extract_yaml_const, ["fr_cross_section", "interquantile"]),
            Quantity("fr_cs_avg", fr_param_file, self.extract_yaml_const, ["fr_cross_section", "avg"]),
            Quantity("fr_cond_med", fr_param_file, self.extract_yaml_const, ["fr_conductivity", "median"]),
            Quantity("fr_cond_iqr", fr_param_file, self.extract_yaml_const, ["fr_conductivity", "interquantile"]),
            Quantity("fr_cond_avg", fr_param_file, self.extract_yaml_const, ["fr_conductivity", "avg"]),


            Quantity("n_bad_els", heal_file, self.get_heal_stat, [])
        ]
        return quantities


    def _extract_result(self, sample):
        """
        :param config_dict: Parsed config.yaml. see key comments there.
        : return
        """
        quantities = self.define_quantities()
        q_dict = {q.name: q for q in quantities}
        self.result_struct = [ ["value"] + [q.name for q in quantities],
                               ["f8"] + [q.np_type for q in quantities] ]
        sample_dir = sample.directory
        for q in quantities:
            q.file = os.path.join(sample_dir, q.file)

        finished_file = os.path.join(sample_dir, "FINISHED")
        finished = False
        if os.path.exists(finished_file):
            with open(finished_file, "r") as f:
                content = f.read().split()                
            finished = len(content)==1 and content[0] == "done"

        if finished:
            print(sample_dir, "Finished")
            files = {q.file for q in quantities}
            finished_map = {f:os.path.exists(f) for f in files}
            files_exist = all(finished_map.values())
            if files_exist:
                print(sample_dir, "Files exist")

                # read values
                for q in quantities:
                    q.value = q.extractor(q.file, *q.args)
                self.manipulate_quantities(q_dict)

                # scalars to vectors
                n_times = len(quantities[0].value)
                for q in quantities:
                    v = np.atleast_1d(q.value)
                    if len(v) == 1:
                        v = np.full(n_times, v[0])
                    q.value = v
                    assert n_times == len(q.value)

                q_val_table = np.array([q.value for q in quantities])
                result = []
                for i_time in range(n_times):
                    result.append((i_time, *q_val_table[:, i_time]))
                return result
            else:
                print(sample_dir, "missing files", finished_map)
                return [np.inf for q in quantities]
        else:
            return [None for q in quantities]

