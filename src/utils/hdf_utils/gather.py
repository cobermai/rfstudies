import os
from typing import Callable
import logging
from pathlib import Path
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial
from collections.abc import Iterable
import h5py
from src.utils.system.progress import working_on
log = logging.getLogger("MLOG")


def _get_ext_link_rek(file_path: str,
                      hdf_path: str,
                      depth_to_go: int,
                      func_to_fulfill: Callable[[str, str], bool]) -> set:
    if depth_to_go==0:
        ret = {h5py.ExternalLink(file_path, hdf_path)} if func_to_fulfill(file_path, hdf_path) else set()
    elif depth_to_go>0:
        with h5py.File(file_path, "r") as file:
            ret_set = set({})
            for key in file[hdf_path].keys():
                working_on("rek: " + file_path + hdf_path + " - " + key)
                ret_set.update(set(_get_ext_link_rek(file_path=file_path,
                                                     hdf_path=hdf_path + "/" + key,
                                                     depth_to_go=depth_to_go - 1,
                                                     func_to_fulfill= func_to_fulfill)))
            ret = ret_set
    else:
        raise ValueError("depth_to_go should be a non negative integer")
    return ret

def _hdf_write_ext_links(from_file_path:str,
                        write_to_file_path: str,
                        lock: mp.Lock,
                        depth: int,
                        func_to_fulfill: Callable[[str, str], bool]) -> None:
    data_file_name = Path(from_file_path).name.split(".")[0]
    ext_link_list = _get_ext_link_rek(file_path=from_file_path,
                                      hdf_path="/",
                                      depth_to_go=depth,
                                      func_to_fulfill=func_to_fulfill)
    with lock:
        with h5py.File(write_to_file_path, "r+") as write_to_file:
            for link in ext_link_list:
                grp_name = data_file_name + " - " + link.path.split("/")[-1]
                write_to_file[grp_name] = link
            write_to_file.flush()


class Gather():
    """
    combines hdf groups of many hdf files into one by creating external links that point to the original files
    """
    def __init__(self, depth: int = 1, num_processes: int = None):
        '''
        initializes a gather object.
        :param depth: the depth of the hdf groups to combine
        :param num_processes: number of processing kernels
        '''
        self.depth = depth
        self.num_processes = os.cpu_count() if num_processes is None else num_processes

    def from_files(self, from_file_paths: Iterable ):
        return GatherFromHdfFiles(self, from_file_paths)


class GatherFromHdfFiles:
    """
    Adds from which file type to gather
    """
    def __init__(self, combiner:Gather, from_file_paths:Iterable):
        """
        :param combiner: The parent Gather object
        :param from_file_paths: defines the file paths of the hdf files to gather
        """
        self.from_file_paths = from_file_paths
        self.combiner = combiner

    def to_hdf_file(self, to_file_path: str):
        return GatherFromHdfFilesToHdfFile(self, to_file_path)


class GatherFromHdfFilesToHdfFile:
    """
    Adds to which file type output should go
    """
    def __init__(self, gather_from_hdf_files: GatherFromHdfFiles, to_file_path: str):
        """
        :param gather_from_hdf_files: parent gather from hdf fiels class
        :param to_file_path: defines the hdf file path the gathered data should go to
        """
        self.combiner = gather_from_hdf_files.combiner
        self.from_file_paths = gather_from_hdf_files.from_file_paths
        h5py.File(to_file_path, "w").close()
        self.to_file_path = to_file_path

    def if_fulfills(self, func_to_fulfill: Callable[[str, str], bool] = None, on_error: bool = None):
        return GatherFromHdfFilesToHdfFileIfFulfills(self, func_to_fulfill, on_error)

    def run(self):
        self.if_fulfills()

class GatherFromHdfFilesToHdfFileIfFulfills():
    """
    Adds a filtering mechanism to get rid of corrupt data (ex. Nan values, wrong format etc)
    """
    def __init__(self, combine_hdf2hdf: GatherFromHdfFilesToHdfFile,
                 func_to_fulfill: Callable[[str, str], bool] = None,
                 on_error: bool = None):
        """
        :param combine_hdf2hdf: parent combine from hdf to hdf file class
        :param func_to_fulfill: the function, the group objects of given depth have to fulfill
        :param on_error: defines what should be done when the func_to_fulfill has an error. default
        """
        self.combiner = combine_hdf2hdf.combiner
        self.from_file_paths = combine_hdf2hdf.from_file_paths
        self.to_file_path = combine_hdf2hdf.to_file_path
        if func_to_fulfill is None:
            func_to_fulfill = lambda str1, str2: True
        def func_to_fulfill_with_error_handling(file_path, hdf_path) -> bool:
            try:
                return func_to_fulfill(file_path, hdf_path)
            except:
                log.debug("function_to_fulfill error (%s, %s) -> %s", file_path, hdf_path, on_error)
                return on_error
        self.func_to_fulfill = func_to_fulfill_with_error_handling

    def with_external_links(self):
        multi_proc_func = partial(_hdf_write_ext_links,
                                    write_to_file_path=self.to_file_path,
                                    lock=mp.Lock(),
                                    depth=self.combiner.depth,
                                    func_to_fulfill=self.func_to_fulfill)
        with ThreadPool(self.combiner.num_processes) as pool:
            pool.map(multi_proc_func, self.from_file_paths)
        log.debug("finished Gathering for %s", self.to_file_path)
