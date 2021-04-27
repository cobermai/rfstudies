import h5py
import typing
import logging
from tqdm import tqdm
log = logging.getLogger("MLOG")

# TODO: MP with tasks + lock
def _get_ext_links(path) -> list:
    link_list = []
    with h5py.File(path, "r") as f:
        for grp in f.values():
            link_list.append(h5py.ExternalLink(path, grp.name))
    return link_list

class VirtualHdf():
    def __init__(self, v_file_path, layer=0) -> None:
        """
        this class filles or creates an hdf5 file with external links when applying create
        :param v_file_path: the file path of the virtual hdf file
        """
        self.v_file_path = v_file_path
        try:
            f = h5py.File(v_file_path, "w")
            f.close()
        except:
            raise AssertionError("The file is already open and can't be opened again")

    def add(self, in_file_paths: typing.Union[set, list]) -> None:
        """
        creates external links to the in_file_paths objects in layer 1
        :param in_file_paths: the hdf files to be linked
        """
        with h5py.File(self.v_file_path, "r+") as virtual:
            for path in tqdm(in_file_paths, total=len(in_file_paths), desc="creating_virtual_hdf"):
                for link in _get_ext_links(path):
                     # a group name can occure multiple times in different files, when that happens, the filename is added
                    grp_name = lambda s: s if virtual.get(s) == None else grp_name(s + " (" + path.split("/")[-1].split(".")[0] +")")
                    virtual[grp_name(link.path)] = link

    def layer_0_create(self, hdf_paths: typing.Union[set, list]):
        """
        creates external links to the in_file_paths objects in layer 0 so with hdf_path "/"
        :param in_file_paths: the hdf files to be linked
        """
        with h5py.File(self.v_file_path, "r+") as virtual:
            for path in tqdm(hdf_paths):
                filename = path.split("/")[-1].split(".")[0]
                virtual[filename] = h5py.ExternalLink(path, "/")
