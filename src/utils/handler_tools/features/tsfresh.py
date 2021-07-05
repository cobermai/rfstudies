"""This module contains a function that calls the tsfresh feature calculation with predefined settings."""
import pandas as pd
import tsfresh


def get_tsfresh(data, settings, njobs=0) -> pd.DataFrame:
    """calls the tsfresh feature calculation on all input data. It combines all calculated features into a single
    dataframe and returns it.
    :param data: a dictionary of {vec_name:data_vector}
    :param settings: the tsfresh feature settings.
    :param njobs: the number of cores used to calculate the features. Attension, for features that can be calculated
    quickly, using multiple cores can have a big overhead.
    :return: dataframe of calculated features. Columns are the input data vector names and index are  the features."""
    df_molten = pd.DataFrame()
    for length in {val.shape[0] for val in data.values()}:
        df_tmp = pd.DataFrame({key: val for key, val in data.items() if len(val) == length})
        df_tmp['column_sort'] = df_tmp.index
        df_molten = df_molten.append(df_tmp.melt(id_vars='column_sort', value_name="tsfresh", var_name="channel"))

    df_ret = tsfresh.extract_features(timeseries_container=df_molten,
                                          column_id="channel",
                                          column_sort="column_sort",
                                          column_value="tsfresh",
                                          default_fc_parameters=settings,
                                          n_jobs=njobs,
                                          disable_progressbar=True).T
    return df_ret
