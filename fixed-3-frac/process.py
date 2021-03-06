import sys
sys.path.append('../MLMC/src')

import subprocess
import yaml
import attr
import numpy as np
import matplotlib.pyplot as plt
import copy

import MLMC.src.gmsh_io as gmsh_io


@attr.s(auto_attribs=True)
class ValueDesctription:
    time: float
    position: str
    quantity: str
    unit: str


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


def compute_hm(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    substitute_placeholders('01_hm_tmpl.yaml', '01_hm.yaml', config_dict['hm_params'])
    arguments = config_dict['flow_executable'].copy()
    arguments.extend(['--output_dir', 'output_hm', '01_hm.yaml'])
    print("Running: ", " ".join(arguments))
    subprocess.call(arguments)


def prepare_th_input(config_dict):
    """
    Prepare FieldFE input file for the TH simulation.
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    # pass
    # we have to read region names from the input mesh
    # input_mesh = gmsh_io.GmshIO(config_dict['hm_params']['mesh'])
    #
    # is_bc_region = {}
    # for name, (id, _) in input_mesh.physical.items():
    #     unquoted_name = name.strip("\"'")
    #     is_bc_region[id] = (unquoted_name[0] == '.')

    # read mesh and mechanichal output data
    mesh = gmsh_io.GmshIO('output_hm/mechanics.msh')

    n_bulk = len(mesh.elements)
    ele_ids = np.zeros(n_bulk, dtype=int)
    for i, id_bulk in zip(range(n_bulk), mesh.elements.items()):
        ele_ids[i] = id_bulk[0]

    init_fr_cs = float(config_dict['hm_params']['fr_cross_section'])
    init_fr_K = float(config_dict['hm_params']['fr_conductivity'])
    init_bulk_K = float(config_dict['hm_params']['bulk_conductivity'])

    field_cs = mesh.element_data['cross_section_updated'][1]

    K = np.zeros((n_bulk, 1), dtype=float)
    cs = np.zeros((n_bulk, 1), dtype=float)
    for i, valcs in zip(range(n_bulk), field_cs[1].values()):
        cs_el = valcs[0]
        cs[i, 0] = cs_el
        if cs_el != 1.0:    # if cross_section == 1, i.e. 3d bulk
            K[i, 0] = init_fr_K * (cs_el*cs_el) / (init_fr_cs*init_fr_cs)
        else:
            K[i, 0] = init_bulk_K

    # mesh.write_fields('output_hm/th_input.msh', ele_ids, {'conductivity': K})
    th_input_file = 'output_hm/th_input.msh'
    with open(th_input_file, "w") as fout:
        mesh.write_ascii(fout)
        mesh.write_element_data(fout, ele_ids, 'conductivity', K)
        mesh.write_element_data(fout, ele_ids, 'cross_section_updated', cs)

    # create field for K (copy cs)
    # posun dat K do casu 0
    # read original K = oK (define in config yaml)
    # read original cs = ocs (define in config yaml)
    # compute K = oK * (cs/ocs)^2
    # write K

    # posun dat cs do casu 0
    # write cs

    # mesh.element_data.


def compute_th(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    substitute_placeholders('02_th_tmpl.yaml', '02_th.yaml', config_dict['th_params'])
    arguments = config_dict['flow_executable'].copy()
    arguments.extend(['--output_dir', 'output_th', '02_th.yaml'])
    print("Running: ", " ".join(arguments))
    subprocess.call(arguments)


def get_result_description():
    """
    :return:
    """
    end_time = 30
    values = [ [ValueDesctription(time=t, position="extraction_well", quantity="power", unit="MW"),
                ValueDesctription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
                ] for t in np.linspace(0, end_time, 0.1)]
    power_series, temp_series = zip(*values)
    return power_series + temp_series


def extract_time_series(yaml_stream, regions, extract):
    """

    :param yaml_stream:
    :param regions:
    :return: times list, list: for every region the array of value series
    """
    data = yaml.safe_load(yaml_stream)['data']
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


def extract_results(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    : return
    """
    abs_zero_temp = 273.15
    year_sec = 60 * 60 * 24 *365
    bc_regions = ['.left_fr_left_well', '.left_well', '.right_fr_right_well', '.right_well']
    out_regions = bc_regions[2:]
    with open("output_th/energy_balance.yaml", "r") as f:
        power_times, reg_powers = extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
    power_series = -sum(reg_powers)

    with open("output_th/Heat_AdvectionDiffusion_region_stat.yaml", "r") as f:
        temp_times, reg_temps = extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
    with open("output_th/water_balance.yaml", "r") as f:
        flux_times, reg_fluxes = extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
    sum_flux = sum(reg_fluxes)
    avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux

    fig, ax1 = plt.subplots()
    temp_color = 'red'
    ax1.set_xlabel('time [y]')
    ax1.set_ylabel('Temperature [C deg]', color=temp_color)
    ax1.plot(temp_times[1:] / year_sec, avg_temp[1:] - abs_zero_temp, color=temp_color)
    ax1.tick_params(axis='y', labelcolor=temp_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    pow_color = 'blue'
    ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
    ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
    ax2.tick_params(axis='y', labelcolor=pow_color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    compute_hm(config_dict)
    prepare_th_input(config_dict)
    compute_th(config_dict)
    extract_results(config_dict)
