from datetime import datetime
import logging
import os
from pathlib import Path

class HTCondorRunner:
    """
    class used to run simulations on a cluster interfaced by HTCondor
    """

    @staticmethod
    def run(hyperparameters, dataset, manual_split=None, manual_scale=None):
        """
        The runner of HTCondor is taking care of running jobs on HTCondor. This is done in 2 steps:
        1) creating the directory / input file / etc of each analysis one wants to perform
        2) submitting all the analysis in one job, requesting as many cores as analysis
        """
        work_dir = Path.cwd().parent.parent
        htc_dir = work_dir / "src/htc"
        output_dir = work_dir / "src/output" / datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        main_name = "xbox2_main.py"

        # creating the master bash file
        venv_exists = False
        if "venv" in os.listdir():
            venv_exists = True

        master_bash_filename = htc_dir / "htc_run.sh"
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
                           f"--hyperparam={hyperparameters}"
                           f"--dataset={dataset}"
                           f"--manual_split={manual_split}"
                           f"--manual_scale={manual_scale}")
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}")
        os.system(f"chmod +x {master_bash_filename}")

        # creating the master sub file
        master_sub_filename = htc_dir / "htc_submit.sub"
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
        command = f"cd {htc_dir} ; condor_submit  {master_sub_filename}; condor_q"
        logging.debug(f"Executing HTCondor command {command}")
        os.system(command)

if __name__ == '__main__':
    HTCondorRunner().run()
