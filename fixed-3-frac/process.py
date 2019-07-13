import subprocess
import yaml
import attr
import numpy as np

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
    pass

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


def extract_results(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    : return
    """
    pass

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    #compute_hm(config_dict)
    #prepare_th_input(config_dict)
    compute_th(config_dict)
    extract_results(config_dict)