"""
This module does the first step of the transformation. The reading and writing of the data.
"""
import os
from time import time
from pathlib import Path
import multiprocessing as mp
import logging
from functools import partial
import nptdms
import h5py

logger = logging.getLogger(__name__)


def _convert_file(tdms_file_path: Path, hdf_dir: Path) -> None:
    """
    converts the tdms file of tdms_file_path to an hdf file by use of the nptdms converter
    :param tdms_file_path: file path of the tdms file
    :param hdf_dir: file path of the hdf file (dir instead of file name is given because easier handling with mp)
    """
    t_0 = time()
    with nptdms.TdmsFile(tdms_file_path) as tdms_file:
        logger.debug("reading tdms file  %40s     took: %10.10s sec", tdms_file_path.stem, time() - t_0)
        hdf_file_path = hdf_dir / tdms_file_path.with_suffix(".hdf").name
        t_0 = time()
        tdms_file.as_hdf(hdf_file_path, mode="w", group="/")
        logger.debug("tdms2hdf + writing %40s     took: %10.10s sec", tdms_file_path.stem, time() - t_0)


class Convert:
    """A very general converter Object, that can be adapted to new data formats."""

    def __init__(self,
                 check_already_converted: bool = True,
                 num_processes: int = 2):
        """
        Initializes the Converter, that can run in parallel with num_processes many threads. Its possible that part of
        the files have been converted already. If check_already_converted is True, only the files that are missing will
        be converted. If check_already_converted is False, everything with be converted from scratch.
        :param check_already_converted: only convert the not yet converted files
        :param num_processes: number of processes for parallel conversion
        """
        self.check_already_converted = check_already_converted
        self.num_processes = num_processes

    def from_tdms(self, tdms_dir: Path):
        """
        Adds the source directory where the tdms files are located and returns a ConvertFromTdms object
        :param tdms_dir: file path of the directory where the tdms files are located
        :return: a ConvertFromTdms object
        """
        return ConvertFromTdms(tdms_dir, self.check_already_converted, self.num_processes)


class ConvertFromTdms(Convert):
    """Adds the from_directory (source) for Convert"""

    def __init__(self, tdms_dir: Path, check_already_converted: bool, num_processes: int):
        """
        Initializes the ConvertFromTdms class object, inherits from Convert. tdms_dir is the source directory where the
        tdms files to convert are located.
        :param tdms_dir: source directory of tdms files
        :param check_already_converted: only convert the not yet converted files
        :param num_processes: number of processes for parallel conversion
        """
        super().__init__(check_already_converted, num_processes)
        self.tdms_dir = tdms_dir

    def to_hdf(self, hdf_dir: Path):
        """
        Adds the destination directory where the hdf files will be stored and returns a ConvertFromTdmsToHdf object
        :param hdf_dir: path of the directory where the hdf files should go to
        :return: a ConvertFromTdmsToHdf object
        """
        return ConvertFromTdmsToHdf(hdf_dir, self.tdms_dir, self.check_already_converted, self.num_processes)


class ConvertFromTdmsToHdf(ConvertFromTdms):
    """Adds the destination directory (destination) for ConvertFromTdms"""

    def __init__(self, hdf_dir: Path, tdms_dir: Path, check_already_converted: bool, num_processes: int):
        """
        Initializes the ConvertFromTdmsToHdf class object
        :param hdf_dir: destination directory of the hdf files
        :param tdms_dir: ConverterFromTdms class object
        """
        super().__init__(tdms_dir, check_already_converted, num_processes)
        self.hdf_dir = hdf_dir

    def get_tdms_file_paths_to_convert(self) -> set:
        """
        returns a set of tdms files that will be converted
        if check_already_converted -> returns the file paths that are not converted yet
        else -> return all tdms files in the tdms_dir
        :return: set of file paths that will be converted
        """
        tdms_file_paths = self.tdms_dir.glob("*.tdms")
        if self.check_already_converted:
            for path in self.hdf_dir.glob("*.hdf"):
                try:
                    # if the writing process of an hdf file was aborted prematurely, the file can not be opened.
                    h5py.File(path, "r").close()
                except OSError:
                    os.remove(path)
            hdf_file_names = set(path.stem for path in self.hdf_dir.glob("*.hdf"))
            ret = set(p for p in tdms_file_paths if p.stem not in hdf_file_names)
        else:
            ret = set(tdms_file_paths)
        if len(ret) != 0:
            logger.debug("Files to convert: %s", len(ret))
        return ret

    def run(self) -> None:
        """Starts the converting process"""
        t_tot = time()
        if self.num_processes == 1:
            for path in self.get_tdms_file_paths_to_convert():
                _convert_file(path, self.hdf_dir)
        else:
            convert_func = partial(_convert_file, hdf_dir=self.hdf_dir)
            with mp.Pool(self.num_processes) as pool:
                pool.map(convert_func, self.get_tdms_file_paths_to_convert())
        logger.debug("In total tdms to hdf5 conversion took: %s sec", time() - t_tot)
