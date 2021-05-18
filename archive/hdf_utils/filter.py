import h5py   # type: ignore
import typing
import logging
log = logging.getLogger("MLOG")

def _apply_rek(file_path: str,
               hdf_path: str,
               filter_rule: typing.Callable[[str, str], bool],
               on_filter_do: typing.Callable[[str, str], None],
               layers_to_go: int,
               lock=None) -> None:
    """
    use with Filter.apply(file_path)
    :param file_path: the path of the .hdf file
    :param hdf_path: the path inside the .hdf
    :param filter_rule: on True -> action, on False -> pass
    :param on_filter_do: when filter_rule True apply to .hdf object
    :param layers_to_go: how deep this recursive function should go into the hdf file
    :param lock: when used in paralell, a lock is needed for writing into files
    :return: void
    """
    if layers_to_go == 0:
        if filter_rule(file_path, hdf_path):
            on_filter_do(file_path, hdf_path)
    elif layers_to_go > 0:
        with h5py.File(file_path, "r+") as f:
            for key in f[hdf_path].keys():
                _apply_rek(file_path, hdf_path + key + "/", filter_rule, on_filter_do, layers_to_go - 1)
    else:
        raise ValueError(
            "Recursion variable out of range. It has to be a non negative int, but got " + str(layers_to_go))


class Filter():
    def __init__(self,
                 filter_rule: typing.Callable[[str, str], bool],
                 apply_on_layer: int = 1,
                 on_filter_do: typing.Callable[[str, str], None] = None):
        """
        applying the filter object on a hdf_file applies the on_filter_do function on the hdf5-objects
        apply_on_layer-layers deep if the filter_rule is true.
        :param filter_rule: On True apply on_filter_do to objects of layer opply_on_layer. On False pass.
        :param apply_on_layer: which layer of the hdf file tree the filter should be applied to
        :param on_filter_do: The action to be done when filter_rule is true. On default its DELETE.
        """
        self.with_multiprocessing = False  # TODO: MP
        self.filter_rule = filter_rule
        self.layer = apply_on_layer
        if on_filter_do==None:
            def do(file_path:str, hdf_path:str):
                with h5py.File(file_path, "r+") as f:
                    print("Filter rule True for " + str(f[hdf_path]))
            self.on_filter_do = do
        else:
            self.on_filter_do = on_filter_do

    def apply(self, file_path: str):
        _apply_rek(file_path = file_path,
                   hdf_path = "/",
                   filter_rule = self.filter_rule,
                   on_filter_do = self.on_filter_do,
                   layers_to_go = self.layer)
