"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.dataset_creator import DatasetCreator
from src.utils.hdf_tools import hdf_to_df_selection
from src.xbox2_specific.utils import dataset_utils


class SimpleSelect(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def select_events(context_data_file_path: Path) -> pd.DataFrame:
        """
        selection of events in data
        :param context_data_file_path: path to context data file
        :return selection: boolean filter for selecting breakdown events
        """
        selection_list = ["is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]
        selection = dataset_utils.select_events_from_list(context_data_file_path, selection_list)
        df = hdf_to_df_selection(context_data_file_path, selection=selection)
        return df

    @staticmethod
    def select_features(df: pd.DataFrame) -> np.ndarray:
        """
        returns features of selected events for modeling
        :param df: dataframe with selected events
        :return X: label of selected events
        """
        selection_list = ["Timestamp",
                          "PrevTrendData__Timestamp",
                          "is_bd", "is_healthy",
                          "is_bd_in_20ms",
                          "is_bd_in_40ms"]
        feature_names = df.columns.difference(pd.Index(selection_list))

        X = df[feature_names].to_numpy(dtype=float)
        X = X[..., np.newaxis]
        X = np.nan_to_num(X)
        return X

    @staticmethod
    def select_labels(df: pd.DataFrame) -> np.ndarray:
        """
        returns labels of selected events for supervised machine learning
        :param df: dataframe with selected events
        :return y: label of selected events
        """
        y = df["is_healthy"].to_numpy(dtype=bool)
        return y
