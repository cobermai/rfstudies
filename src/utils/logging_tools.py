from pathlib import Path
import glob
import pandas as pd
import argparse
import datetime


def log_to_csv(logging_path: Path, **kwargs):
    """
    log data, e.g. results, to csv file
    :param logging_path: path to log csv file
    :param kwargs: arguments to write into the csv file
    """
    if logging_path.is_file():
        df = pd.read_csv(logging_path)
        df_new = pd.DataFrame(kwargs, index=[0])
        df = pd.concat([df, df_new], axis=1)
    else:
        logging_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(kwargs, index=[0])
    df.to_csv(logging_path, index=False)


def gather_csv(output_file_path: Path, input_file_regex: str):
    """
    gathers all logged data, e.g. results, and stores them into given path
    :param output_file_path: path to store gathered data
    :param input_file_regex: regex string to load data from
    """
    files = sorted(glob.glob(input_file_regex))
    if files:
        df = pd.concat(map(pd.read_csv, files))
    else:
        df = pd.read_csv(files)
    df.reset_index(inplace=True)
    df.to_csv(output_file_path)
    print(output_file_path)
    print(df)


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='input parameter')
    parser.add_argument('--output', required=False, type=str,
                        help='name of output directory', default=str(datetime.datetime.now()))
    args = parser.parse_args()

    eos_path = Path("/eos/project-m/ml-for-alarm-system/private/CLIC_data_transfert/sensitivity_analysis/")
    files_to_load = "../output/*/results.csv"
    gather_csv(output_file_path=eos_path / (args.output + ".csv"), input_file_regex=files_to_load)
