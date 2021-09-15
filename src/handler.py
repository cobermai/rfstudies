"""In this module all feature calculation for xbox2 dataset is done."""

import argparse
from dataclasses import dataclass, field
import logging
import itertools
from pathlib import Path
import typing
import coloredlogs
import h5py
import numpy as np
import tsfresh
from tqdm import tqdm
from src.utils.handler_tools.context_data_creator import ContextDataCreator
from src.utils.handler_tools.context_data_writer import ColumnWiseContextDataWriter, RowWiseContextDataWriter
from src.utils.hdf_tools import hdf_path_combine, sort_by
from src.xbox2_specific.feature_definition.attribute import get_event_attribute_features
from src.xbox2_specific.feature_definition.event import get_event_data_features
from src.xbox2_specific.feature_definition.trend import get_trend_data_features
from src.xbox2_specific.feature_definition.tsfresh import get_tsfresh


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
        """
        calls the feature calculation process. The features are generated by the python files in the
        src/utils/handler_tools/features folder."""
        feature_list = list(get_event_attribute_features(length=self.num_events))
        self.manage_event_attribute_features(feature_list)

        feature_list = list(get_trend_data_features(length=self.num_events, trend_data_file_path=self.td_file_path))
        self.manage_trend_data_features(feature_list)

        feature_list = list(get_event_data_features(length=self.num_events))
        self.manage_event_data_and_tsfresh_features(event_data_features=feature_list)

        self.feature_post_processing()

    def manage_event_attribute_features(self, features: typing.List) -> None:
        """manages the reading and writing of attributes in the event data.
        :param features: a list of EventAttributeFeatures"""
        # calculate features
        with h5py.File(self.ed_file_path, "r") as file:
            attrs_gen = (grp.attrs for grp in file.values())
            for attrs, index in zip(attrs_gen, itertools.count(0)):
                for feature in features:
                    feature.calculate_single(index, attrs)
        # write them to the context data
        column_wise_handler = ColumnWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        for feature in features:
            column_wise_handler.write_column(feature)

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
            feature.vec = feature.calculate_all(loc)
            cw_handler.write_column(feature)

    def manage_event_data_and_tsfresh_features(self, event_data_features: typing.List) -> None:
        """manages the calculation and writing of event data features, so features that are calculated from the event
        data time series. It calculates some custom features written in the event data features and tsfresh features.
        :param event_data_features: a list of EventDataFeatures"""
        row_wise_handler = RowWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        with h5py.File(self.ed_file_path, "r") as file:
            data_gen = ({key: channel[:] for key, channel in grp.items()} for grp in file.values())
            for data, index in tqdm(zip(data_gen, itertools.count(0))):
                for feature in event_data_features:
                    feature.calculate_single(index, data)

                tsfresh_df = get_tsfresh(data=data, settings=tsfresh.feature_extraction.MinimalFCParameters())
                val_gen = ((hdf_path_combine(str(clm_id), str(row_id)), val)
                           for row_id, row in tsfresh_df.iterrows()
                           for clm_id, val in row.items())
                row_wise_handler.write_row_from_external(index=index, data_iter=val_gen)

        column_wise_handler = ColumnWiseContextDataWriter(self.dest_file_path, length=self.num_events)
        for feature in event_data_features:
            column_wise_handler.write_column(feature)

    @staticmethod
    def assign_run_no(timestamps: np.datetime64):
        # Hardcoded timestamps for classifying runs. Format is [start, stop, run_no]. Negative run_no is commissioning.
        timestamp_list_run = np.array([
            [np.datetime64('2018-05-15 21:16:59.626459'), np.datetime64('2018-05-23 14:41:58.024856'), -1],
            [np.datetime64('2018-05-23 14:42:58.036909'), np.datetime64('2018-06-01 11:41:23.304752'), 1],
            [np.datetime64('2018-06-01 11:42:23.326588'), np.datetime64('2018-06-10 09:57:46.167949'), -2],
            [np.datetime64('2018-06-10 09:58:46.176918'), np.datetime64('2018-06-20 10:49:26.504749'), 2],
            [np.datetime64('2018-06-20 10:50:26.517873'), np.datetime64('2018-06-27 01:56:14.100424'), -3],
            [np.datetime64('2018-06-27 01:57:14.092206'), np.datetime64('2018-07-17 08:49:18.581234'), 3],
            [np.datetime64('2018-07-17 08:50:18.644042'), np.datetime64('2018-07-17 14:24:22.808270'), -4],
            [np.datetime64('2018-07-17 14:25:22.808983'), np.datetime64('2018-07-26 15:56:55.238768'), 4],
            [np.datetime64('2018-07-26 15:57:55.254684'), np.datetime64('2018-07-30 12:58:28.800693'), -5],
            [np.datetime64('2018-07-30 12:59:28.850502'), np.datetime64('2018-08-09 07:18:19.717621'), 5],
            [np.datetime64('2018-08-09 07:19:19.717776'), np.datetime64('2018-08-16 07:48:45.260491'), -6],
            [np.datetime64('2018-08-16 07:49:45.217265'), np.datetime64('2018-08-22 19:07:06.581874'), 6],
            [np.datetime64('2018-08-24 22:53:03.560161'), np.datetime64('2018-08-27 20:21:22.319445'), -7],
            [np.datetime64('2018-08-27 20:22:22.331644'), np.datetime64('2018-09-03 09:53:18.547360'), 7],
            [np.datetime64('2018-09-03 09:54:18.540067'), np.datetime64('2018-09-05 16:48:36.589576'), -8],
            [np.datetime64('2018-09-05 16:49:36.595947'), np.datetime64('2018-09-17 06:27:33.398326'), 8],
            [np.datetime64('2018-09-17 06:28:33.412608'), np.datetime64('2018-09-19 00:05:14.894480'), -9],
            [np.datetime64('2018-09-19 00:06:14.912150'), np.datetime64('2018-09-25 09:51:59.222968'), 9]
        ])

        runs_assigned = np.zeros(shape=timestamps.shape, dtype=int)
        for run in timestamp_list_run:
            run_start = run[0]
            run_end = run[1]
            runs_assigned[(timestamps >= run_start) & (timestamps <= run_end)] = run[2]
        return runs_assigned

    def _get_run_no(self, file):
        """
        returns the Timestamp from group properties/attribute in numpy datetime format
        :param attrs: the h5py.AttributeManager of an hdf.Group object
        :return: numpy datetime format of the timestamp
        """
        timestamps = file["Timestamp"]
        run_no = self.assign_run_no(timestamps)
        return run_no

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
            file.create_dataset(name="run_no", data=self._get_run_no(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for xbox2 dataset: extract features for ml")
    parser.add_argument("td", type=Path, help="file path of an hdf file where all TrendData is merged into one dataset"
                                              "for each signal.")
    parser.add_argument("ed", type=Path, help="file path of an hdf file where all EventData groups are gathered (e.g."
                                              "with ExternalLinks).")
    parser.add_argument("dest", type=Path, help="file path of the hdf file where the features will be written.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug log messages")
    args = parser.parse_args()
    coloredlogs.install(level="DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)

    cd_creator = XBox2ContextDataCreator(ed_file_path=args.ed.resolve(),
                                         td_file_path=args.td.resolve(),
                                         dest_file_path=args.dest.resolve())
    cd_creator.manage_features()
