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
