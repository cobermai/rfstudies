"""This module contains a class structure for creating a context data file. The ContextDataCreator class organizes the
creation of the context data file."""
import typing
from dataclasses import dataclass, field
import logging
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import tsfresh
from src.utils.hdf_tools import hdf_path_combine
from src.utils.handler_tools.features.attribute import get_event_attribute_features
from src.utils.handler_tools.features.event import get_event_data_features
from src.utils.handler_tools.features.trend import get_trend_data_features
from src.utils.handler_tools.features.tsfresh import get_tsfresh
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
    num_events: int = field(init=False)

    def __post_init__(self):
        with h5py.File(self.ed_file_path, "r") as file:
            self.num_events: int = len(file)  # number of event keys

    def manage_features(self):
        t0 = time.time()
        feature_list = list(get_event_attribute_features(self.num_events))
        self.manage_event_attribute_features(feature_list)
        print("event attribute features took ", time.time() - t0)

        t0 = time.time()
        feature_list = list(get_event_data_features())
        self.manage_event_data_features(feature_list)
        print("event data features calculation took ", time.time() - start)

        feature_list = list(get_trend_data_features(self.num_events, self.td_file_path))
        self.manage_trend_data_features(feature_list)
        print("trend data      features took ", time.time() - t0)

    def manage_event_attribute_features(self, features: typing.List):
        # calculate features
        with h5py.File(self.ed_file_path, "r") as file:
            attrs_gen = (grp.attrs for grp in file.values())
            for attrs, index in zip(attrs_gen, itertools.count(0)):
                for feature in features:
                    feature.vec[index] = feature.func(attrs)
        # write them to the context data
        cw_handler = ColumnWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        for feature in features:
            cw_handler.write_clm(feature)

    def manage_event_data_features(self, features: typing.List) -> None:
        """manages the calculation and writing of event data features, so features that are calculated from the event
        data time series. It calculates some custom features written in the event data features and tsfresh features.
        :param features: a list of EventDataFeatures"""
        rw_handler = RowWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        with h5py.File(self.ed_file_path, "r") as file:
            data_gen = ({key: channel[:] for key, channel in grp.items()} for grp in file.values())
            for data, index in tqdm(zip(data_gen, itertools.count(0))):
                features_vals = [feature.apply(data) for feature in features]
                val_gen = zip(features, features_vals)
                rw_handler.write_row_custom_features(index, val_gen)

                tsfresh_df = get_tsfresh(data, tsfresh.feature_extraction.MinimalFCParameters())
                val_gen = ((hdf_path_combine(str(clm_id), str(row_id)), val)
                           for row_id, row in tsfresh_df.iterrows()
                           for clm_id, val in row.items())
                rw_handler.write_row_from_external(index, val_gen)

    def manage_trend_data_features(self, features: typing.List):
        with h5py.File(self.td_file_path, "r") as trend_data_file, \
                h5py.File(self.dest_file_path, "r") as context_data_file:
            trend_ts = np.array(trend_data_file["Timestamp"][:])
            event_ts = np.array(context_data_file["Timestamp"][:])
            loc = np.searchsorted(trend_ts, event_ts) - 1
        cw_handler = ColumnWiseContextDataHandler(self.dest_file_path, length=self.num_events)
        for feature in features:
            feature.vec = feature.calc_all(loc)
            cw_handler.write_clm(feature)


if __name__ == "__main__":
    creator = ContextDataDirector(ed_file_path=Path("~/output_files/EventDataExtLinks.hdf").expanduser(),
                                  td_file_path=Path("~/output_files/combined.hdf").expanduser(),
                                  dest_file_path=Path("~/output_files/contextd.hdf").expanduser(),
                                  )
    h5py.File(creator.dest_file_path, "w").close()
    creator.manage_features()
