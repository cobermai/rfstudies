"""This module contains the definition and gathering of EvenAttributeFeatures for the XBox2 data set."""
import typing
import h5py
import numpy as np
from src.utils.handler_tools.feature_class import EventAttributeFeature


def get_event_attribute_features(length: int) -> typing.Generator:
    """This function generates all EventAttributeFeatures for the xbox2 data set.
    :param length: number of values that will be calculated by each feature.
    :return: generator of features"""
    yield EventAttributeFeature(name="Timestamp",
                                func=_get_timestamp,
                                length=length,
                                hdf_path="/",
                                output_dtype=h5py.opaque_dtype('M8[us]'),
                                info="The timestamp of the EventData is a property of the event group. It is given in "
                                     "a datetime format with micro seconds precision.")

    for is_type in ["is_healthy", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]:
        func = _log_type_creator(is_type)
        yield EventAttributeFeature(name=is_type,
                                    func=func,
                                    length=length,
                                    hdf_path="/clic_label/",
                                    output_dtype=bool,
                                    info="These values originated from the Log_Type assigned by the CLIC-Team."
                                         "Originally the Log_Type property had values in {0,1,2,3} where 0 stood for a"
                                         "healthy or normal log signal, and 3 for a breakdown. The label 1 and 2 stood"
                                         "for breakdown in 20ms and 40ms, so the signals prior to a breakdown.")

    yield EventAttributeFeature(name="run_no",
                                func=_get_run_no,
                                length=length,
                                hdf_path="/",
                                output_dtype=int,
                                info="The run number of the based on manually defined stable run periods. "
                                     "Run numbers 1-9 define stable runs and negative numbers denote commisioning"
                                     "A 0 denotes data outside stable runs.")


def _log_type_creator(type_of_interest: str) -> typing.Callable:
    """creates functions that return True if the input value matches the translation of the is_type label and
    False in the other cases.
    :param type_of_interest: string of the type of interest (in {"is_log", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"})
    """
    log_type_dict = {"is_healthy": 0, "is_bd_in_40ms": 1, "is_bd_in_20ms": 2, "is_bd": 3}

    def is_type(attrs: h5py.AttributeManager) -> bool:
        """
        This function translates the 'Log Type' group properties of the event data into a boolean value.
        :param attrs: the h5py.AttributeManager of an hdf.Group object
        :return: True if (is_log -> 0, is_bd_in40ms -> 1, is_bd_in20ms -> 2, is_bd -> 3) in other cases return False
        """
        label = attrs["Log Type"]
        if label in log_type_dict.values():
            is_defined_type = label == log_type_dict[type_of_interest]
        else:
            raise ValueError(f"'Log Type' label not valid no translation for {label} in {log_type_dict}!")
        return is_defined_type

    return is_type


def _get_timestamp(attrs: h5py.AttributeManager):
    """
    returns the Timestamp from group properties/attribute in numpy datetime format
    :param attrs: the h5py.AttributeManager of an hdf.Group object
    :return: numpy datetime format of the timestamp
    """
    datetime_str = attrs["Timestamp"][:-1]
    return np.datetime64(datetime_str).astype(h5py.opaque_dtype('M8[us]'))


def assign_run_no(timestamp: np.datetime64):
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

    run_no_assigned = 0
    for run in timestamp_list_run:
        run_start = run[0]
        run_end = run[1]
        if (timestamp >= run_start) & (timestamp <= run_end):
            run_no_assigned = run[2]
            break

    return run_no_assigned


def _get_run_no(attrs: h5py.AttributeManager):
    """
    returns the Timestamp from group properties/attribute in numpy datetime format
    :param attrs: the h5py.AttributeManager of an hdf.Group object
    :return: numpy datetime format of the timestamp
    """
    datetime_str = attrs["Timestamp"][:-1]
    timestamp = np.datetime64(datetime_str)
    run_no = assign_run_no(timestamp)
    return run_no
