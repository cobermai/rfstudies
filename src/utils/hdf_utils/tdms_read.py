from time import time
import os
import glob
from pathlib import Path
import multiprocessing as mp
import logging
import nptdms
log = logging.getLogger("MLOG")


class Convert:
    def __init__(self,
                 check_already_converted: bool = True,
                 num_processes: int = None):
        """
        Converts tdms files from the tdms_dir directory into hdf files in the hdf_dir directory.
        :param check_already_converted: check if some hdf files with similar filenames \
        to the tdms files alredy exist
        :param num_processes: number of processes for multiprocessing
        """
        self.check_already_converted = check_already_converted
        self.num_processes = num_processes if not num_processes is None else int(os.cpu_count()/2)
    def from_tdms(self, tdms_dir):
        return ConvertFromTdms(self, tdms_dir)


class ConvertFromTdms:
    def __init__(self, convert: Convert, tdms_dir: str):
        self.convert = convert
        self.tdms_dir = tdms_dir
    def to_hdf(self, hdf_dir):
        return ConvertFromTdmsToHdf(self, hdf_dir)


class ConvertFromTdmsToHdf:
    def __init__(self, fromtdms: ConvertFromTdms, hdf_dir):
        self.convert = fromtdms.convert
        self.tdms_dir = fromtdms.tdms_dir
        self.hdf_dir = hdf_dir

    def __get_file_name(self, path) -> str:
        return Path(path).name.split(".")[0]

    def __get_tdms_file_paths_to_convert(self) -> set:
        tdms_file_paths = set(glob.glob(self.tdms_dir + "*.tdms"))
        if self.convert.check_already_converted:
            hdf_file_names = {self.__get_file_name(path) for path in glob.glob(self.hdf_dir + ".hdf")}
            ret = set(path for path in tdms_file_paths if not self.__get_file_name(path) in hdf_file_names)
        else:
            ret = tdms_file_paths
        return ret

    def __convert_file(self, tdms_file_path: str) -> None:
        tdms_file_name = self.__get_file_name(tdms_file_path)
        t_0 = time()
        with nptdms.TdmsFile(tdms_file_path) as tdms_file:
            log.debug("reading tdms file  " + tdms_file_name + "     took: " + str(time() - t_0) + " sec")
            t_0 = time()
            tdms_file.as_hdf(self.hdf_dir + tdms_file_name + ".hdf", mode="w", group="/")
            log.debug("tdms2hdf + writing " + tdms_file_name + "     took: " + str(time() - t_0) + " sec")

    def run(self) -> None:
        """
        Converts all tdms files from tdms_dir to .hdf files and puts them into hdf_dir
        :param tdms_dir: The directory path that contains the input .tdms files.
        :param hdf_dir: The directory path where the converted .hdf5 files will go.
        :param check_already_converted: Check if some tdms files are already converted in the hdf_dir.
        :param num_processes: The number of processes to be converted with.
        """
        t_tot = time()
        if self.convert.num_processes == 1:
            for path in self.__get_tdms_file_paths_to_convert():
                self.__convert_file(path)
        else:
            with mp.Pool(self.convert.num_processes) as pool:
                pool.map(self.__convert_file, self.__get_tdms_file_paths_to_convert())
        log.debug("In total conversion of tdms to hdf5 took: " + str(time() - t_tot) + " sec")
