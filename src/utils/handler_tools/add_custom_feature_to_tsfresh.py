from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import EfficientFCParameters
import tsfresh
from pathlib import Path
import h5py
import pandas as pd

def ts_fresh_features(file_path: Path, hdf_path: str):
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""
    with h5py.File(file_path, "r") as file:
        grp = file[hdf_path]
        df = pd.DataFrame(data={key: grp[key][:] for key in grp.keys() if len(grp[key][:]) == 3200 and "Amplitude" in key})
        df['column_sort'] = df.index
        df_molten = df.melt(id_vars='column_sort')
        settings = EfficientFCParameters()
        settings["my_pulse_length"] = None
        return tsfresh.extract_features(timeseries_container=df_molten,
                                        column_id="variable",
                                        column_sort="column_sort",
                                        column_value="value",
                                        default_fc_parameters=settings,n_jobs=0)


df = ts_fresh_features(Path("~/output_files/data/EventData_20180401.hdf").expanduser(), "/Log_2018.04.01-23:39:54.227")
print(df)