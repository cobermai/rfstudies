"""This module contains a class structure for creating a context data file. The ContextDataCreator class organizes the
creation of the context data file."""
import typing
from dataclasses import dataclass, field
import logging
import itertools
from pathlib import Path
from functools import partial
import multiprocessing as mp
import tsfresh
import pandas as pd
import numpy as np
import time
import tqdm
import h5py
from src.utils.hdf_tools import hdf_path_combine
from src.utils.handler_tools.customfeature import RowWiseFeature, EventDataFeature, EventAttributeFeature, TrendDataFeature
from src.utils.handler_tools.event_attribute_features import get_event_attribute_features
from src.utils.handler_tools.event_data_features import get_event_data_features
from src.utils.handler_tools.trend_data_features import get_trend_data_features
from src.utils.handler_tools.contextdatahandler import ColumnWiseContextDataHandler, RowWiseContextDataHandler

logger = logging.getLogger(__name__)
CUT_OFF = range(1000)  # itertools.count(0)

def tsfresh_from_df(df: pd.DataFrame, settings):
    df['column_sort'] = df.index
    df_molten = df.melt(id_vars='column_sort', value_name="tsfresh", var_name="channel")
    return tsfresh.extract_features(timeseries_container=df_molten,
                                    column_id="channel",
                                    column_sort="column_sort",
                                    column_value="tsfresh",
                                    default_fc_parameters=settings,
                                    n_jobs=0 if settings==tsfresh.feature_extraction.MinimalFCParameters() else 4,
                                    disable_progressbar=True).T


def tsfresh_on_event_data(data):
    def gen_df(data):
        num_values = 3200
        df = pd.DataFrame({key: val for key, val in data.items() if len(val) == num_values if "Amplitude" in key})
        yield tsfresh_from_df(df=df, settings=tsfresh.feature_extraction.MinimalFCParameters())  # EfficientFCParameters()

        df = pd.DataFrame({key: val for key, val in data.items() if len(val) == num_values if "Phase" in key})
        yield tsfresh_from_df(df=df, settings=tsfresh.feature_extraction.MinimalFCParameters())

        num_values = 500
        df = pd.DataFrame({key: val for key, val in data.items() if len(val) == num_values})
        yield tsfresh_from_df(df=df, settings=tsfresh.feature_extraction.MinimalFCParameters())

    return list(gen_df(data))


@dataclass
class ContextDataDirector:
    """operates the creation of the context data file (a file filled with calculated features for each group in the
     input file."""
    ed_file_path: Path
    td_file_path: Path
    dest_file_path: Path
    chunk_size: int = 10
    num_events: int = field(init=False)

    def __post_init__(self):
        with h5py.File(self.ed_file_path, "r") as file:
            self.num_events: int = len(file)  # number of event keys
            self.num_events = len([x for x in CUT_OFF])

    def manage_features(self):
        t0 = time.time()
        feature_list = list(get_event_attribute_features(self.num_events))
        self.manage_event_attribute_features(feature_list)
        print("took ", time.time() - t0)

        t0 = time.time()
        feature_list = list(get_event_data_features())
        self.manage_event_data_features(feature_list)
        print("took ", time.time() - t0)

        t0 = time.time()
        feature_list = list(get_trend_data_features(self.num_events, self.td_file_path))
        self.manage_trend_data_features(feature_list)
        print("took ", time.time() - t0)

    def manage_event_attribute_features(self, features: typing.List):
        # calculate features
        with h5py.File(self.ed_file_path, "r") as file:
            attrs_gen = (grp.attrs for grp in file.values())
            for attrs, index in zip(attrs_gen, CUT_OFF):
                for feature in features:
                    feature.vec[index] = feature.func(attrs)
        # write them to the context data
        cw_handler = ColumnWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        for feature in features:
            cw_handler.write(feature)

    def manage_trend_data_features(self, features: typing.List):
        with h5py.File(self.td_file_path, "r") as trend_data_file, \
            h5py.File(self.dest_file_path, "r") as context_data_file:
            trend_ts = np.array(trend_data_file["Timestamp"][:])
            event_ts = np.array(context_data_file["Timestamp"][:])
            loc = np.searchsorted(trend_ts, event_ts) - 1
        cw_handler = ColumnWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        for feature in features:
            feature.vec = feature.calc(loc)
            cw_handler.write(feature)


    def manage_event_data_features(self, features: typing.List):
        rw_handler = RowWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        rw_handler.init_features(features)

        with h5py.File(self.ed_file_path, "r") as file:
            def data_gen() -> typing.Generator:
                for grp in file.values():
                    yield {key: channel[:] for key, channel in grp.items()}

            for data, index in zip(data_gen(), CUT_OFF):
                custom_features_vals = [feature.apply(data) for feature in features]
                def custom_features_vals_store(feature_vals: typing.List[RowWiseFeature]) -> typing.Generator:
                    for feature, val in zip(features, feature_vals):
                        yield (feature.full_hdf_path, val)

                rw_handler.write_row(index, custom_features_vals_store(custom_features_vals))

                def tsfresh_vals_store(tsfresh_df_list: typing.List) -> typing.Generator:
                    for df in tsfresh_df_list:
                        for row_index, row in df.iterrows():
                            for column_index, value in row.items():
                                yield (hdf_path_combine(str(column_index), str(row_index)), value)
                if index==0:
                    rw_handler.init_tsfresh(tsfresh_vals_store(tsfresh_on_event_data(data)))
                rw_handler.write_row(index, tsfresh_vals_store(tsfresh_on_event_data(data)))


if __name__=="__main__":

    creator = ContextDataDirector(ed_file_path=Path("~/output_files/EventDataExtLinks.hdf").expanduser(),
                                  td_file_path=Path("~/output_files/combined.hdf").expanduser(),
                                  dest_file_path=Path("~/output_files/contextd.hdf").expanduser(),
                                  )
    h5py.File(creator.dest_file_path, "w").close()
    creator.manage_features()
