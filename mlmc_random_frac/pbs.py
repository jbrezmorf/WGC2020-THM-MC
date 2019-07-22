import os
import os.path
import shutil
import subprocess
import mlmc.pbs as pbs


class Pbs(pbs.Pbs):
    def add_realization(self, weight, lines, **kwargs):
        """
        Append new flow123d realization to the existing script content
        :param weight: current simulation steps
        :param kwargs: dict with params
        :return: None
        """
        if self._number_of_realizations == 0:
            self.clean_script()

        assert self.pbs_script is not None

        # lines = [
        #     'cd {work_dir}',
        #     'date +%y.%m.%d_%H:%M:%S',
        #     'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out',
        #     'date +%y.%m.%d_%H:%M:%S',
        #     'touch {output_subdir}/FINISHED',
        #     'echo \\"Finished simulation:\\" \\"{flow123d}\\" \\"{work_dir}\\" \\"{output_subdir}\\"',
        #     '']
        lines = [line.format(**kwargs) for line in lines]
        self.pbs_script.extend(lines)

        self._number_of_realizations += 1
        self._current_job_weight += weight
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
            self.execute()

        return self._pbs_config['job_name']