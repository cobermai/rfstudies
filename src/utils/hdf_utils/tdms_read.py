from time import time
import os
import glob
from pathlib import Path
import multiprocessing as mp
import logging
import nptdms
from functools import partial
log = logging.getLogger("MLOG")


def _get_file_name(path:str) -> str:
    return Path(path).name.split(".")[0]


def _convert_file(tdms_file_path: str, hdf_dir:str) -> None:
    tdms_file_name = _get_file_name(tdms_file_path)
    t_0 = time()
    with nptdms.TdmsFile(tdms_file_path) as tdms_file:
        log.debug("reading tdms file  %s     took: %s sec", tdms_file_name, time() - t_0)
        t_0 = time()
        tdms_file.as_hdf(hdf_dir + tdms_file_name + ".hdf", mode="w", group="/")
        log.debug("tdms2hdf + writing %s     took: %s sec", tdms_file_name, time() - t_0)


class Convert:
    """Converts tdms files from the tdms_dir directory into hdf files in the hdf_dir directory."""
    def __init__(self,
                 check_already_converted: bool = True,
                 num_processes: int = None):
        """
        Initializes the Converter
        :param check_already_converted: check if some hdf files with similar filenames \
        to the tdms files alredy exist
        :param num_processes: number of processes for multiprocessing
        """
        self.check_already_converted = check_already_converted
        self.num_processes = num_processes if not num_processes is None else int(os.cpu_count()/2)

    def from_tdms(self, tdms_dir):
        return ConvertFromTdms(self, tdms_dir)


class ConvertFromTdms:
    """Adds the from directory for conversion"""
    def __init__(self, convert: Convert, tdms_dir: str):
        """
        Initializes the ConverterFromTdms class object
        :param convert: converter class object
        :param tdms_dir: directory of the tdms files to be converted
        """
        self.convert = convert
        self.tdms_dir = tdms_dir
    def to_hdf(self, hdf_dir):
        return ConvertFromTdmsToHdf(self, hdf_dir)


class ConvertFromTdmsToHdf:
    """Adds the to destination directory for conversion"""
    def __init__(self, fromtdms: ConvertFromTdms, hdf_dir):
        """
        Initializes the ConverterFromTdmsToHdf class object
        :param fromtdms: ConverterFromTdms class object
        :param hdf_dir: destination directory of the hdf files
        """
        self.convert = fromtdms.convert
        self.tdms_dir = fromtdms.tdms_dir
        self.hdf_dir = hdf_dir

    def __get_tdms_file_paths_to_convert(self) -> set:
        """if check_already_converted -> returns the file paths that are not converted yet
        else -> return all tdms files in the tdms_dir"""
        tdms_file_paths = set(glob.glob(self.tdms_dir + "*.tdms"))
        if self.convert.check_already_converted:
            hdf_file_names = {_get_file_name(path) for path in glob.glob(self.hdf_dir + "*.hdf")}
            ret = set(path for path in tdms_file_paths if not _get_file_name(path) in hdf_file_names)
        else:
            ret = tdms_file_paths
        if len(ret)!=0: log.debug("Files to convert: %s", len(ret))
        return ret

    def run(self) -> None:
        """lets the converter run """
        t_tot = time()
        if self.convert.num_processes == 1:
            for path in self.__get_tdms_file_paths_to_convert():
                _convert_file(path, self.hdf_dir)
        else:
            convert_func = partial(_convert_file, hdf_dir = self.hdf_dir)
            with mp.Pool(self.convert.num_processes) as pool:
                pool.map(convert_func, self.__get_tdms_file_paths_to_convert())
        if time() - t_tot > 1.0: log.debug("In total conversion of tdms to hdf5 took: %s sec",time() - t_tot)
        log.debug("finished ConversionToHdf")
