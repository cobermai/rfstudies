"""In this module all feature calculation for xbox2 dataset is done."""
import logging
from pathlib import Path
from dataclasses import dataclass, field
import itertools
import typing
import argparse
import coloredlogs
import numpy as np
import h5py
from tqdm import tqdm
import tsfresh
from src.union import sort_by
from src.utils.hdf_tools import hdf_path_combine
from src.utils.handler_tools.context_data_creator import ContextDataCreator
from src.utils.handler_tools.xbox2_features.attribute import get_event_attribute_features
from src.utils.handler_tools.xbox2_features.event import get_event_data_features
from src.utils.handler_tools.xbox2_features.trend import get_trend_data_features
from src.utils.handler_tools.xbox2_features.tsfresh import get_tsfresh
from src.utils.handler_tools.context_data_writer import ColumnWiseContextDataWriter, RowWiseContextDataWriter

logger = logging.getLogger(__name__)


@dataclass
class XBox2ContextDataCreator(ContextDataCreator):
    """operates the creation of the context data file (a file filled with calculated features for each group in the
     input file."""
    ed_file_path: Path
    td_file_path: Path
    num_events: int = field(init=False)

    def __post_init__(self):
        with h5py.File(self.ed_file_path, "r") as file:
            self.num_events: int = len(file)  # number of event keys

    def manage_features(self):
        """calls the feature calculation process. The features are generated by the python files in the
        src/utils/handler_tools/features folder."""
        feature_list = list(get_event_attribute_features(self.num_events))
        self.manage_event_attribute_features(feature_list)

        feature_list = list(get_event_data_features())
        self.manage_event_data_features(feature_list)

        feature_list = list(get_trend_data_features(self.num_events, self.td_file_path))
        self.manage_trend_data_features(feature_list)

        self.feature_post_processing()

    def manage_event_attribute_features(self, features: typing.List) -> None:
        """manages the reading and writing of attributes in the event data.
        :param features: a list of EventAttributeFeatures"""
        # calculate features
        with h5py.File(self.ed_file_path, "r") as file:
            attrs_gen = (grp.attrs for grp in file.values())
            for attrs, index in zip(attrs_gen, itertools.count(0)):
                for feature in features:
                    feature.calc(index, attrs)
        # write them to the context data
        cw_handler = ColumnWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        for feature in features:
            cw_handler.write_clm(feature)

    def manage_event_data_features(self, features: typing.List) -> None:
        """manages the calculation and writing of event data features, so features that are calculated from the event
        data time series. It calculates some custom features written in the event data features and tsfresh features.
        :param features: a list of EventDataFeatures"""
        rw_handler = RowWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        with h5py.File(self.ed_file_path, "r") as file:
            data_gen = ({key: channel[:] for key, channel in grp.items()} for grp in file.values())
            for data, index in tqdm(zip(data_gen, itertools.count(0))):
                feature_values = [feature.apply(data) for feature in features]
                rw_handler.write_row_custom_features(index=index, data_iter=zip(features, feature_values))

                tsfresh_df = get_tsfresh(data, tsfresh.feature_extraction.MinimalFCParameters())
                val_gen = ((hdf_path_combine(str(clm_id), str(row_id)), val)
                           for row_id, row in tsfresh_df.iterrows()
                           for clm_id, val in row.items())
                rw_handler.write_row_from_external(index=index, data_iter=val_gen)

    def manage_trend_data_features(self, features: typing.List) -> None:
        """manage the reading and writing of the trend data features, it reads and writes the closest preceding trend
        data for every event.
        :param features: a list of TrendDataFeatures"""
        with h5py.File(self.td_file_path, "r") as trend_data_file, \
                h5py.File(self.dest_file_path, "r") as context_data_file:
            trend_ts = np.array(trend_data_file["Timestamp"][:])
            event_ts = np.array(context_data_file["Timestamp"][:])
            loc = np.searchsorted(trend_ts, event_ts) - 1
        cw_handler = ColumnWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        for feature in features:
            feature.vec = feature.calc_all(loc)
            cw_handler.write_clm(feature)

    def feature_post_processing(self):
        """After the other features have been calculated, some new features will be added resulting from the ones
        already calculated."""
        sort_by(self.dest_file_path, "Timestamp")
        with h5py.File(self.dest_file_path, "r+") as file:
            clic_label = file["/clic_label"]

            dc_up_threshold_label = file["dc_up_threshold_reached"][:]
            is_bd = clic_label["is_bd_in_40ms"][:-2] & \
                    clic_label["is_bd_in_20ms"][1:-1] & \
                    clic_label["is_bd"][2:] & \
                    dc_up_threshold_label[2:]
            is_bd = np.append([False, False], is_bd)

            file.create_dataset(name="is_bd", data=is_bd)
            file.create_dataset(name="is_bd_in_20ms", data=np.append(is_bd[1:], [False]))
            file.create_dataset(name="is_bd_in_40ms", data=np.append(is_bd[2:], [False, False]))
            file.create_dataset(name="is_healthy", data=file["clic_label/is_healthy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for xbox2 dataset: extract features for ml")
    parser.add_argument("td", type=Path, help="file path of an hdf file where all TrendData is merged into one dataset"
                                              "for each signal.")
    parser.add_argument("ed", type=Path, help="file path of an hdf file where all EventData groups are gathered (e.g."
                                              "with ExternalLinks).")
    parser.add_argument("dest", type=Path, help="file path of the hdf file where the features will be written.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug log messages")
    args = parser.parse_args()
    if args.verbose:
        coloredlogs.install(level="DEBUG")
    else:
        coloredlogs.install(level="INFO")
    logger = logging.getLogger(__name__)

    cd_creator = XBox2ContextDataCreator(ed_file_path=args.ed.resolve(),
                                    td_file_path=args.td.resolve(),
                                    dest_file_path=args.dest.resolve())
    cd_creator.manage_features()
