"""
The gather module combines/concatenates/groups/glues/sticks-together/merges/assembles hdf-groups scattered on
multiple files into one.
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
from src.utils.hdf_tools import hdf_path_combine
log = logging.getLogger(__name__)


def _get_ext_link_rec(file_path: Path,
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
    if depth_to_go == 0:
        return {h5py.ExternalLink(file_path, hdf_path)} if func_to_fulfill(file_path, hdf_path) else set()

    if depth_to_go > 0:
        with h5py.File(file_path, "r") as file:
            children_set = set()
            for key in file[hdf_path].keys():
                children_set.update(set(_get_ext_link_rec(file_path=file_path,
                                                          hdf_path=hdf_path_combine(hdf_path, key),
                                                          depth_to_go=depth_to_go - 1,
                                                          func_to_fulfill=func_to_fulfill)))
            return children_set

    raise ValueError("depth_to_go should be a non negative integer")


def _hdf_write_ext_links(source_file_path: Path,
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
    :param func_to_fulfill: the function that has to be fulfilled in order to be gathered
    """
    ext_link_list = _get_ext_link_rec(file_path=source_file_path,
                                      hdf_path="/",
                                      depth_to_go=depth,
                                      func_to_fulfill=func_to_fulfill)
    with lock:
        with h5py.File(dest_file_path, "a") as dest_file:
            for link in ext_link_list:
                grp_name = source_file_path.stem + "-" + link.path.replace("/", "-")
                dest_file[grp_name] = link


def _get_func_to_fulfill(on_error: bool,
                         func_to_fulfill: Callable[[Path, str], bool] = None) -> Callable[[Path, str], bool]:
    """Sets the function_to_fulfill. On True the HdfObject will be added, if False it will not be added.
    Additionally if an error occurs the HdfObject will not be added.
    :param func_to_fulfill: The filtering function/ restriction function/ function to fulfill for an HdfObject to
    be added to the output.
    :param on_error: boolean value that will be returned when the func_to_fulfill throws an error."""

    def func_to_fulfill_with_error_handling(file_path: Path, hdf_path: str) -> bool:
        ret = on_error
        try:
            if func_to_fulfill is None:
                ret = True
            else:
                ret = func_to_fulfill(file_path, hdf_path)
        except (ValueError, SystemError, ArithmeticError, AttributeError, LookupError, RuntimeError):
            log.info("Caught error for function_to_fulfill on input (%s, %s). Returned on_error=%s",
                     file_path, hdf_path, on_error)
        return ret

    return func_to_fulfill_with_error_handling


def gather(src_file_paths: Iterable,
           dest_file_path: Path,
           if_fulfills: Callable[[Path, str], bool] = None,
           on_error: bool = False,
           depth: int = 1,
           num_processes: int = 2) -> None:
    """gathers hdf-groups of many hdf files into one by creating external links that point to the original files.
    This way the data can be accessed though one without copying the data
    :param src_file_paths: Iterable of Path objects of the source hdf-file-paths
    :param dest_file_path: Path of the destination hdf file
    :param if_fulfills: function that needs to be fulfilled in order for the hdf-object to be added to the destination
    file via external links.
    :param on_error: what should
    :param depth: depth where the objects
    :param num_processes: number of processors for parallel gathering
    """
    h5py.File(dest_file_path, "w").close()  # overwrite destination file
    multi_proc_func = partial(_hdf_write_ext_links,
                              dest_file_path=dest_file_path,
                              lock=Lock(),
                              depth=depth,
                              func_to_fulfill=_get_func_to_fulfill(on_error, if_fulfills))
    with ThreadPool(num_processes) as pool:
        pool.map(multi_proc_func, src_file_paths)
    log.debug("finished Gathering %s", dest_file_path)
