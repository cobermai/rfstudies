"""
The gathering module combines/concatenates/groups/glues/sticks together/merges/assembles multible files into one.
Currently implemented:
* Gather hdf groups of arbitrary layer scattered on many hdf files with external links (hdf external links).
  This gathers groups without copying its contents.
"""
import os
from typing import Callable
import logging
from pathlib import Path
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial
from collections.abc import Iterable
import h5py
log = logging.getLogger("MLOG")  # the standard logger for this machine learning framework, see utils/system/logger.py


def hdf_path_combine(*argv: str) -> str:
    """
    Concatenates hdf path with "/" inbetwen. Works similar to Path(str, str, str) or the / operator for Path objects
    but for hdf paths (as strings)
    :param argv: the group names/to concatenate
    :return: the concatenated path string
    """
    path = "/".join(argv)
    path = path.replace("///", "/")
    path = path.replace("//", "/")
    if path[0]!="/":
        path = f"/{path}"
    return path

def _get_ext_link_rek(file_path: Path,
                      hdf_path: str,
                      depth_to_go: int,
                      func_to_fulfill: Callable[[Path, str], bool]) -> set:
    if depth_to_go==0:
        ret = {h5py.ExternalLink(file_path, hdf_path)} if func_to_fulfill(file_path, hdf_path) else set()
    elif depth_to_go>0:
        with h5py.File(file_path, "r") as file:
            ret_set = set({})
            for key in file[hdf_path].keys():
                ret_set.update(set(_get_ext_link_rek(file_path=file_path,
                                                     hdf_path=hdf_path_combine(hdf_path, key),
                                                     depth_to_go=depth_to_go - 1,
                                                     func_to_fulfill= func_to_fulfill)))
            ret = ret_set
    else:
        raise ValueError("depth_to_go should be a non negative integer")
    return ret

def _hdf_write_ext_links(from_file_path: Path,
                        write_to_file_path: Path,
                        lock: mp.synchronize.Lock,
                        depth: int,
                        func_to_fulfill: Callable[[Path, str], bool]) -> None:
    ext_link_list = _get_ext_link_rek(file_path=from_file_path,
                                      hdf_path="/",
                                      depth_to_go=depth,
                                      func_to_fulfill=func_to_fulfill)
    with lock:
        with h5py.File(write_to_file_path, "a") as write_to_file:
            rek_grp_name = lambda x: x + "(new)" if write_to_file.get(x, None) is not None else x
            for link in ext_link_list:

                grp_name = rek_grp_name(from_file_path.stem + " - " + link.path)
                write_to_file[grp_name] = link
            write_to_file.flush()


class Gather:
    """
    combines hdf groups of many hdf files into one by creating external links that point to the original files
    """
    def __init__(self, depth: int = 1, num_processes: int = None):
        '''
        :param depth: the depth of the hdf groups to combine
        :param num_processes: number of processing kernels, None -> os.cpu_count
        '''
        self.depth: int = depth
        self.set_num_processes(num_processes)
        self.to_file_path: Path
        self.from_file_paths: Iterable
        self.func_to_fulfill: Callable[[Path, str], bool]

    def set_num_processes(self, num_processes: int = None) -> None:
        """Sets the num_processes for the ThreadPool, if None is given its set to the number of logical cpu cores"""
        self.num_processes: int = os.cpu_count() or 1 if num_processes is None else num_processes

    def set_func_to_fulfill(self, on_error: bool, func_to_fulfill: Callable[[Path, str], bool] = None) -> None:
        """Sets the function_to_fulfill. On True the HdfObject will be added, if False it will not be added.
        Addidtionally if an error occures the HdfObject will not be added.
        :param func_to_fulfill: The filtering function/ restriction function/ function to fulfill for an HdfObject to
        be added to the output."""
        def func_to_fulfill_with_error_handling(file_path:Path, hdf_path: str) -> bool:
            try:
                if func_to_fulfill is None:
                    ret = True
                else:
                    ret = func_to_fulfill(file_path, hdf_path)
            except (KeyError,ValueError, SystemError, ArithmeticError, AttributeError, LookupError, NotImplementedError,
                    RuntimeError) :
                log.debug("function_to_fulfill error (%s, %s) -> %s", file_path, hdf_path, on_error)
                ret = on_error
            else:
                raise RuntimeWarning("An unexpected Error occurred in func_to_fulfill")
            return ret
        self.func_to_fulfill = func_to_fulfill_with_error_handling

    def from_files(self, from_file_paths: Iterable):
        """
        Sets the source file paths for the gathering
        :param from_file_paths: Iterable of file paths to be added
        :return: The Gather object
        """
        self.from_file_paths = from_file_paths
        return self

    def to_hdf_file(self, to_file_path: Path):
        """
        Sets the destination file path for the gathering
        :param from_file_paths: Path of the destination file path
        :return: The Gather object
        """
        self.to_file_path = to_file_path
        return self

    def if_fulfills(self, func_to_fulfill: Callable[[Path, str], bool] = None, on_error: bool = False):
        """
        calls set_function_to_fulfill with the on_error parameter
        :param from_file_paths: Iterable of file paths to be added
        :return: The Gather object
        """
        self.set_func_to_fulfill(on_error, func_to_fulfill)
        return self

    def run_with_external_links(self) -> None:
        """This starts the Gathering by creating hdf ExternalLink objects"""
        multi_proc_func = partial(_hdf_write_ext_links,
                                    write_to_file_path=self.to_file_path,
                                    lock=mp.Lock(),
                                    depth=self.depth,
                                    func_to_fulfill=self.func_to_fulfill)
        with ThreadPool(self.num_processes) as pool:
            pool.map(multi_proc_func, self.from_file_paths)
        log.debug("finished Gathering for %s", self.to_file_path)
