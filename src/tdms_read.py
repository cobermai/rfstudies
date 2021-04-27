import nptdms
from time import time
import glob
import os
import multiprocessing as mp
import math
import numpy as np
import pandas as pd
import h5py
import sys
import logging
import tqdm
log = logging.getLogger("MLOG")

class ConverterToHdf:
    def __init__(self,
                 tdms_dir: str,
                 hdf_dir: str,
                 check_already_converted: bool = True,
                 num_processes: int = None):
        """
        Converts tdms files from the tdms_dir directory into hdf files in the hdf_dir directory.
        :param tdms_dir: input directory where all tdms files will be converted
        :param hdf_dir: output directory where all hdf files will be converted to
        :param check_already_converted: check if some hdf files with similar filenames to the tdms files alredy exist
        :param num_processes: number of processes for multiprocessing
        """
        self.hdf_dir = hdf_dir
        self.tdms_dir = tdms_dir
        self.check_already_converted = check_already_converted
        self.num_processes = num_processes if not num_processes is None else math.floor(os.cpu_count()/2)

    def get_tdms_paths(self) -> set:
        if self.check_already_converted:
            get_filename = lambda path: os.path.split(path)[1].split(".")[0]
            existing_hdf = {get_filename(path) for path in glob.glob(self.hdf_dir + "*.hdf")}
            return set(path for path in glob.glob(self.tdms_dir + "*.tdms") if not get_filename(path) in existing_hdf)
        else:
            return set(glob.glob(self.tdms_dir + "*.tdms"))

    def _convert(self, tdms_path: str) -> None:
        t0 = time()
        with nptdms.TdmsFile(tdms_path) as tdms:
            log.debug("reading tdms file  " + str(tdms.properties["name"]) + "     took: " + str(time() - t0) + " sec")
            t0 = time()
            tdms.as_hdf(self.hdf_dir + tdms.properties["name"] + ".hdf", mode="w", group="/")
            log.debug("tdms2hdf + writing " + str(tdms.properties["name"]) + "     took: " + str(time() - t0) + " sec")

    def run(self) -> None:
        """
        Converts all tdms files from tdms_dir to .hdf files and puts them into hdf_dir
        :param tdms_dir: The directory path that contains the input .tdms files.
        :param hdf_dir: The directory path where the converted .hdf5 files will go.
        :param check_already_converted: Check if some tdms files are already converted in the hdf_dir.
        :param num_processes: The number of processes to be converted with.
        """
        t_tot = time()
        if self.num_processes == 1:
            print(self.get_tdms_paths())
            for path in self.get_tdms_paths():
                self._convert(path)
        else:
            with mp.Pool(self.num_processes) as pool:
                pool.map(self._convert, self.get_tdms_paths()) # TODO: add get_bar and change to imap
        log.debug("In total conversion of tdms to hdf5 took: " + str(time() - t_tot) + " sec")
