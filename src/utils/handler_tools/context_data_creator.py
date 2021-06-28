"""This module contains a class structure for creating a context data file. The ContextDataCreator class organizes the
creation of the context data file."""
import typing
import logging
import itertools
from pathlib import Path
from functools import partial
import multiprocessing as mp
import tsfresh
import pandas as pd
import numpy as np
import tqdm
import h5py
from src.utils.handler_tools.feature import EventDataFeature
from src.utils.handler_tools.features_for_xb2 import get_event_data_features

logger = logging.getLogger(__name__)


#def task(feature: EventDataFeature,
#         ed_file_path: Path,
#         dest_file_path: Path):
#    """
#    One task is calculating one feature for the whole dataset.
#    :param feature: the Feature object to calculate
#    :param src_file_path: the file path of the source file
#    :param dest_file_path: the file path of the destination file, where the features will be stored
#    :param write_lock: the lock for writing into the destination file path
#    """
#    logger.debug("calculate feature %s for %s", feature.name, feature.dest_hdf_path)
#    vec = feature.apply(ed_file_path)
#    logger.debug("finished calculating feature %s for %s", feature.name, feature.dest_hdf_path)
#    return vec

def task_calculate_tsfresh(key: typing.Iterable, ed_file_path: Path):
    ret = pd.DataFrame()
    for number_of_samples in [3200, 500]:
        with h5py.File(ed_file_path, "r") as file:
            grp = file[key]
            data = {key: grp[key][:] for key in grp.keys()
                        if len(grp[key][:]) == number_of_samples and "Amplitude" in key}
        df = pd.DataFrame(data=data)
        df['column_sort'] = df.index
        df_molten = df.melt(id_vars='column_sort')
        settings = tsfresh.feature_extraction.EfficientFCParameters()  # here we can add custom features
        ret[[key for key in data.keys()]] = tsfresh.extract_features(timeseries_container=df_molten,
                                        column_id="variable",
                                        column_sort="column_sort",
                                        column_value="value",
                                        default_fc_parameters=settings,
                                        n_jobs=0, disable_progressbar=True).T # run serial
    return ret


class ContextDataCreator:
    """operates the creation of the context data file (a file filled with calculated features for each group in the
     input file."""

    def __init__(self, ed_file_path: Path,
                 td_file_path: Path,
                 dest_file_path: Path,
                 get_features):
        self.ed_file_path: Path = ed_file_path
        self.td_file_path: Path = td_file_path
        self.dest_file_path: Path = dest_file_path
        self.get_event_data_features: typing.Callable[[], typing.Iterable[EventDataFeature]] = get_features
        with h5py.File(self.ed_file_path, "r") as file:
            self.len: int = len(file)  # = number of keys
        self.chunk_size = 20
        self.data_chunk: dict = {}

    def write_data_chunk(self, i: int, df: pd.DataFrame):
        if i == 0:
            self.data_chunk.update({chn: {feature: np.empty(self.chunk_size, dtype=float)
                                         for feature in df.index}
                               for chn in df.columns})
        for chn in df.columns:
            for feature in df.index:
                self.data_chunk[chn][feature][i] = df[chn][feature]

    def calc_tsfresh_features(self, num_processors: int = 4):
        """assigns the tasks, calculating a chunk of tsfresh features, to multiple processors.
        :param num_processors: the number of threads for multiprocessing"""
        h5py.File(self.dest_file_path, "w").close()

        with h5py.File(self.ed_file_path, "r") as src_file:
            length = len(src_file)
        init_dest = True

        for start in range(0, length and 2*self.chunk_size, self.chunk_size):
            chunk_slice = slice(start, start + self.chunk_size and length)
            # TODO: propper chunks
            partial_task = partial(task_calculate_tsfresh, ed_file_path=self.ed_file_path)

            with mp.Pool(num_processors) as pool:
                with h5py.File(self.ed_file_path, "r") as src_file:
                    keys = (key for key, _ in zip(src_file.keys(), range(self.chunk_size)))
                    self.data_chunk = {}
                    for df, i in zip(pool.imap(partial_task, keys), itertools.count(0)):
                        self.write_data_chunk(i, df)

                    with h5py.File(self.dest_file_path, "a") as dest_file:
                        # init dest file datasets and groups
                        if init_dest:
                            init_dest = False
                            for chn, ch in self.data_chunk.items():
                                for feature_name, values in ch.items():
                                    dest_chn = dest_file.require_group(chn)
                                    dest_chn.create_dataset(name=feature_name, shape=self.chunk_size, dtype=float, chunks=True)
                        # write into destination
                        for chn, ch in self.data_chunk.items():
                            for feature_name, values in ch.items():
                                dest_file[chn][feature_name][chunk_slice] = self.data_chunk[chn][feature_name][:]

    #def calc_custom_features(self, num_processors: int = 2):
    #    partial_task = partial(task, ed_file_path=self.ed_file_path)
    #    with h5py.File(self.ed_file_path, "r") as file:
    #        keys = [key for key, _ in zip(file.keys(), range(12))]
    #    with mp.Pool(num_processors) as pool:
    #        for key in tqdm.tqdm(pool.imap(keys)):
    #            print(partial_task(key))



if __name__=="__main__":
    creator = ContextDataCreator(ed_file_path=Path("~/output_files/EventDataExtLinks.hdf").expanduser(),
                                 td_file_path=Path("~/output_files/TrendDataExtLinks.hdf").expanduser(),
                                 dest_file_path=Path("~/output_files/contextd.hdf").expanduser(),
                                 get_features=get_event_data_features())
    creator.calc_tsfresh_features(num_processors=4)
