"""
The gathering module combines/concatenates/groups/glues/sticks together/merges/assembles multible files into one.
Currently implemented:
* Gather hdf groups of arbitrary layer scattered on many hdf files with external links (hdf external links).
  This gathers groups without copying its contents.
"""
from typing import Callable
import logging
from pathlib import Path
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as Lock_Data_Type
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
    def rem_double_dash(path: str) -> str:
        if path.find("//")!=-1:
            path = rem_double_dash(path.replace("//","/"))
        return path
    return rem_double_dash("/" + "/".join(argv))


def get_ext_link_rec(file_path: Path,
                     hdf_path: str,
                     depth_to_go: int,
                     func_to_fulfill: Callable[[Path, str], bool]) -> set:
    """
    recursive function to return external links of hdf files in variable depth
    :param file_path: path of the hdf file
    :param hdf_path: hdf_path of the hdf object inside the hdf file
    :param depth_to_go: goal depth to go
    :param func_to_fulfill: the function that has to be fulfilled in order to be gathered
    :return: set of external hdf links
    """
    if depth_to_go==0:
        ret = {h5py.ExternalLink(file_path, hdf_path)} if func_to_fulfill(file_path, hdf_path) else set()
    elif depth_to_go>0:
        with h5py.File(file_path, "r") as file:
            ret_set = set({})
            for key in file[hdf_path].keys():
                ret_set.update(set(get_ext_link_rec(file_path=file_path,
                                                    hdf_path=hdf_path_combine(hdf_path, key),
                                                    depth_to_go=depth_to_go - 1,
                                                    func_to_fulfill= func_to_fulfill)))
            ret = ret_set
    else:
        raise ValueError("depth_to_go should be a non negative integer")
    return ret

def hdf_write_ext_links(source_file_path: Path,
                        dest_file_path: Path,
                        lock: Lock_Data_Type,
                        depth: int,
                        func_to_fulfill: Callable[[Path, str], bool]) -> None:
    """
    writes external links from the source file of given depth into the destination file.
    :param source_file_path: file path of the hdf file
    :param dest_file_path: file path of the destination hdf file
    :param lock: lock to gain writing access to the dest hdf file
    :param depth: the depth in which external links will be created
    :param func_to_fulfill: tthe function that has to be fulfilled in order to be gathered
    """
    ext_link_list = get_ext_link_rec(file_path=source_file_path,
                                     hdf_path="/",
                                     depth_to_go=depth,
                                     func_to_fulfill=func_to_fulfill)
    with lock:
        with h5py.File(dest_file_path, "a") as dest_file:
            for link in ext_link_list:
                grp_name = source_file_path.stem + "-" + link.path.replace("/","-")
                dest_file[grp_name] = link
            dest_file.flush()


class Gather:
    """
    combines hdf groups of many hdf files into one by creating external links that point to the original files
    """
    def __init__(self, depth: int = 1, num_processes: int = 2):
        '''
        :param depth: the depth of the hdf groups to combine
        :param num_processes: number of processing kernels, None -> os.cpu_count
        '''
        self.depth: int = depth
        self.num_processes = num_processes
        self.dest_file_path: Path
        self.source_file_paths: Iterable
        self.func_to_fulfill: Callable[[Path, str], bool]

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
            except (ValueError, SystemError, ArithmeticError, AttributeError, LookupError, RuntimeError):
                log.debug("function_to_fulfill error (%s, %s) -> %s", file_path, hdf_path, on_error)
                ret = on_error
            return ret
        self.func_to_fulfill = func_to_fulfill_with_error_handling

    def from_files(self, source_file_paths: Iterable):
        """
        Sets the source file paths for the gathering
        :param source_file_paths: Iterable of file paths to be added
        :return: The Gather object
        """
        self.source_file_paths = source_file_paths
        return self

    def to_hdf_file(self, dest_file_path: Path):
        """
        Sets the destination file path for the gathering
        :param dest_file_paths: Path of the destination file path
        :return: The Gather object
        """
        h5py.File(dest_file_path, "w").close()  # overwrite old file
        self.dest_file_path = dest_file_path
        return self

    def if_fulfills(self, func_to_fulfill: Callable[[Path, str], bool] = None, on_error: bool = False):
        """
        if func_to_fulfill is fulfilled the link to the hdf_object will be added
        :param func_to_fulfill: function to be fulfilled by a feasible hdf_object
        :param on_error: if the func_to_fulfill has an expected error, the on_error parameter will be returned
        """
        self.set_func_to_fulfill(on_error, func_to_fulfill)
        return self

    def run_with_external_links(self) -> None:
        """This starts the Gathering by creating hdf ExternalLink objects"""
        multi_proc_func = partial(hdf_write_ext_links,
                                  dest_file_path=self.dest_file_path,
                                  lock=Lock(),
                                  depth=self.depth,
                                  func_to_fulfill=self.func_to_fulfill)
        with ThreadPool(self.num_processes) as pool:
            pool.map(multi_proc_func, self.source_file_paths)
        log.debug("finished Gathering for %s", self.dest_file_path)
