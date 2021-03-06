import os
import sys
from pathlib import Path

api_dir = str(Path(os.path.split(os.path.split(os.getcwd())[0])[0]))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)
import json
import logging
import os
from datetime import datetime

default_hyperparameters = {
    "classifier_name": "fcn",
    "num_classes": 2,
    "monitor": "loss",
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "epochs": 500,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 50,
    "min_lr": 0.0001
}


class HTCondorRunner:
    """
    class used to run simulations on a cluster interfaced by HTCondor
    """

    @staticmethod
    def run(hyperparameters,
            dataset_name=None,
            manual_split=None,
            manual_scale=None):
        """
        The runner of HTCondor is taking care of running jobs on HTCondor. This is done in 2 steps:
        1) creating the directory / input file / etc of each analysis one wants to perform
        2) submitting all the analysis in one job, requesting as many cores as analysis
        """
        work_dir = Path.cwd().parent.parent
        output_dir = work_dir / "src/output" / datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        main_name = "xbox2_main.py"

        # store hyperparameters in output file
        with open(output_dir / 'hyperparameters.json', 'w') as fp:
            json.dump(hyperparameters, fp)

        # creating the master bash file
        venv_exists = "venv" in os.listdir()

        master_bash_filename = output_dir / "htc_run.sh"
        with open(master_bash_filename, 'w') as file:
            try:
                file.write("#!/bin/bash\n")
                file.write(f"cd {work_dir}\n")
                if venv_exists:
                    file.write(f"echo \"Virtual environment already exists, delete folder if change necessary\"")
                    file.write(f"source {work_dir}/venv/bin/activate\n")
                else:
                    file.write(f"virtualenv venv\n")
                    file.write(f"source {work_dir}/venv/bin/activate\n")
                    file.write(f"pip3 install -r requirements.txt\n")
                file.write(f"python3 {work_dir / main_name} "
                           f"--file_path={work_dir} "
                           f"--output_path={output_dir} "
                           f"--hyperparameter_path={output_dir / 'hyperparameters.json'} "
                           f"--manual_split=\"{manual_split}\" "
                           f"--manual_scale=\"{manual_scale}\" ")
                if dataset_name:
                    file.write(f"--dataset_name={dataset_name} ")
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}")
        os.system(f"chmod +x {master_bash_filename}")

        # creating the master sub file
        master_sub_filename = output_dir / "htc_submit.sub"
        with open(master_sub_filename, 'w') as file:
            try:
                content_of_sub = (f"executable = {master_bash_filename}\n"
                                  f"arguments = $(ClusterId) $(ProcId)\n"
                                  f"output = {output_dir / 'htc_out.txt'}\n"
                                  f"error = {output_dir / 'htc_err.txt'}\n"
                                  f"log = {output_dir / 'htc_log.txt'}\n"
                                  "RequestCpus = 2\n"
                                  "request_GPUs = 1\n"
                                  "+JobFlavour = \"longlunch\"\n"
                                  "+AccountingGroup = \"group_u_TE.mpe\"\n"
                                  "requirements = regexp(\"V100\", TARGET.CUDADeviceName)\n"
                                  f"queue ")
                file.write(content_of_sub)
            except IOError as e:
                logging.info(f"I/O error({e.errno}): {e.strerror}")

        # submitting a unique request to HTCondor
        command = f"cd {output_dir} ; condor_submit  {master_sub_filename}; condor_q"
        logging.debug(f"Executing HTCondor command {command}")
        os.system(command)


if __name__ == '__main__':
    HTCondorRunner().run(
        hyperparameters=default_hyperparameters,
        manual_split="([1, 7, 2, 4, 9, 5], [6, 8], [3])",
        manual_scale="[1, 2, 3, 4, 5, 6, 7, 8, 9]")
