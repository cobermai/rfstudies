import os
import h5py
import logging
from pathlib import Path
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial
from collections.abc import Iterable
from src.utils.system.progress import working_on
from typing import Callable
from typing import Union
log = logging.getLogger("MLOG")

def _get_ext_link_rek(file_path:str, hdf_path:str, depth_to_go:int, func_to_fulfill: Callable[[str, str], bool]) -> set:
    if depth_to_go==0:
        return {h5py.ExternalLink(file_path, hdf_path)} if func_to_fulfill(file_path, hdf_path) else set()
    elif depth_to_go>0:
        with h5py.File(file_path, "r") as f:
            ret_set = set()
            for key in f[hdf_path].keys():
                working_on("rek: " + file_path + hdf_path + " - " + key)
                ret_set.update(set(_get_ext_link_rek(file_path=file_path,
                                                     hdf_path=hdf_path + "/" + key,
                                                     depth_to_go=depth_to_go - 1,
                                                     func_to_fulfill= func_to_fulfill)))
            return ret_set


def _write_ext_links_to(data_file_path:str,
                        write_to_file_path: str,
                        lock: mp.Lock,
                        depth: int,
                        func_to_fulfill: Callable[[str, str], bool]) -> None:
    data_file_name = Path(data_file_path).name.split(".")[0]
    ext_link_list = _get_ext_link_rek(file_path=data_file_path,
                                      hdf_path="/",
                                      depth_to_go=depth,
                                      func_to_fulfill=func_to_fulfill)
    with lock:
        with h5py.File(write_to_file_path, "r+") as write_to_file:
            for link in ext_link_list:
                grp_name = data_file_name + " - " + link.path.split("/")[-1]
                write_to_file[grp_name] = link
            write_to_file.flush()


class ExternalLinks():
    """
    combines hdf groups of many hdf files into one by creating external links that point to the original files
    """
    def __init__(self, to_file: str, depth: int = 1, num_processes: int = None):
        """
        :param depth: the depth of the hdf groups to combine
        """
        h5py.File(to_file, "w").close()
        self.to_file = to_file
        self.depth = depth
        self.num_processes = os.cpu_count() if num_processes==None else num_processes

    def from_files(self, from_file_paths: Iterable ):
        return ExternalLinksFromFiles(self, from_file_paths)

class ExternalLinksFromFiles():
    def __init__(self, extlinks:ExternalLinks, from_file_paths:Iterable):
        self.from_file_paths = from_file_paths
        self.extlinks = extlinks

    def if_fulfills(self, func_to_fulfill: Callable[[str, str], bool] = None, on_error: bool=False):
        if func_to_fulfill is None:
            func_to_fulfill = lambda str1, str2: True
        def func_to_fulfill_with_error_handling(file_path, hdf_path) -> bool:
            try:
                return func_to_fulfill(file_path, hdf_path)
            except:
                log.debug("function_to_fulfill error on input (" + file_path + ", " + hdf_path + ")" + \
                          " returning " + str(on_error))
                return on_error

        multi_proc_func = partial(_write_ext_links_to,
                                    write_to_file_path=self.extlinks.to_file,
                                    lock=mp.Lock(),
                                    depth=self.extlinks.depth,
                                    func_to_fulfill=func_to_fulfill_with_error_handling)
        with ThreadPool(self.extlinks.num_processes) as pool:
            pool.map(multi_proc_func, self.from_file_paths)

    def run(self):
        self.if_fulfills()