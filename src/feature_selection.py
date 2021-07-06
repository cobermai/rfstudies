"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from src.utils.hdf_tools import hdf_to_df_selection

context_data_file_path = Path("~/output_files/contextd.hdf").expanduser()


with h5py.File(context_data_file_path, "r") as file:
    selection = np.logical_or(file["is_bd_in_40ms"][:], file["is_bd_in_20ms"][:], file["is_log"][:])
    # querying datetime is not possible. See pandas numexpr.necompiler.getType
    event_ts = file["Timestamp"][:]
    trend_ts = file["PrevTrendData/Timestamp"][:]
    diff = event_ts - trend_ts
    threshold = diff.sort_values(ascending=True)[int(len(diff) / 40)]  # alternatively pd.to_timedelta(2,"s")
    filter_timestamp_diff = diff < threshold

    selection = np.logical_and(selection, filter_timestamp_diff)

    is_log = file["is_log"]
    selection[is_log] = np.random.choice(a=[True, False],
                                         size=(sum(is_log),),
                                         p=[0.025, 0.975])  # select 2.5% of log signals randomly

df = hdf_to_df_selection(context_data_file_path, selection=selection)

clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_log",
                                                   "is_bd_in_20ms", "is_bd_in_40ms"]))
X = df[clm_for_training].to_numpy(dtype=float)
Y = df["is_log"].to_numpy(dtype=int)
