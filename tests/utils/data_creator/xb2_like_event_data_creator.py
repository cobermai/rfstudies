"""creating XBox2 like event data for testing"""
from pathlib import Path

import numpy as np

from tests.utils.data_creator.file_creator_for_testing import CreatorTestFiles


def _event_data_creator(tdms_dir_path: Path, hdf_dir_path: Path, file_stem: Path) -> CreatorTestFiles:
    tdms_creator = CreatorTestFiles(hdf_file_path=hdf_dir_path / file_stem.with_suffix(".hdf"),
                                    tdms_file_path=tdms_dir_path / file_stem.with_suffix(".tdms"),
                                    root_prop_dict={"name": f"{file_stem}", "Version": 2})
    dummy_date = "2021-01-01T00:00:00.000000"
    tdms_creator.grp_prop_dict = {
        "BD_BLM TIA": True,
        "BD_DC Down": True,
        "BD_PSR log": True,
        "BD_Col.": True,
        "BD_PKR log": True,
        "BD_BLM": True,
        "BD_PER log": True,
        "BD_DC Up": True,
        "Pulse Count": 1,
        "Timestamp": dummy_date,
        "Breakdown Flags": 0,
        "Log Type": 0,
    }

    tdms_creator.ch_prop_dict = {
        "PKI Amplitude": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PKI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 168610.0,
                          "Scale_Coeff_c1": -1240300.0, "Scale_Coeff_c2": 52222000.0,
                          "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PKI Phase": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PKI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PSI Amplitude": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 0.0,
                          "Scale_Coeff_c1": -1317300.0, "Scale_Coeff_c2": 78582000.0,
                          "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PSI Phase": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PSR Amplitude": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSR Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                          "wf_xunit_string": "s", },
        "PSR Phase": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSR Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PEI Amplitude": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PEI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 0.0,
                          "Scale_Coeff_c1": -378810.0, "Scale_Coeff_c2": 44043000.0, "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PEI Phase": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PEI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "BLM TIA": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "BLM TIA",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "DC Down": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "DC Down",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "PSR log": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PSR log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "Col.": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                 "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "Col.", "NI_UnitDescription": "V",
                 "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PKR log": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PKR log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "BLM": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "BLM", "NI_UnitDescription": "V",
                "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PER log": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PER log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "DC Up": {"wf_start_time": np.datetime64(dummy_date), "wf_start_offset": 0.0,
                  "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "DC Up",
                  "NI_UnitDescription": "V",
                  "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
    }
    tdms_creator.ch_data_dict = {}
    for chn in tdms_creator.ch_prop_dict.keys():
        data_size = int(tdms_creator.ch_prop_dict[chn]["wf_samples"])
        tdms_creator.ch_data_dict.update({chn: [float(val) for val in range(2 ** 32, 2 ** 32 + data_size)]})
    return tdms_creator


def _create_empty(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    tdms_creator = _event_data_creator(tdms_dir_path=created_tdms_files_dir,
                                       hdf_dir_path=created_hdf_files_dir,
                                       file_stem=Path("EventData_20210101_empty"))
    tdms_creator.write()


def _create_ok_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    """creates a tmds file with the specified path that is similar to event data. And tests the functionality of
    the transformation part of the continuous integration."""
    tdms_creator = _event_data_creator(tdms_dir_path=created_tdms_files_dir,
                                       hdf_dir_path=created_hdf_files_dir,
                                       file_stem=Path("EventData_20210101_ok"))
    for flag, group_name in zip([0, 1, 2, 3], ["LogGroupTest_ok_normallog", "LogGroupTest_ok_bdin40ms",
                                               "LogGroupTest_ok_bdin20ms", "BreakdownGroupTest_ok_db"]):
        tdms_creator.grp_prop_dict.update({"Log Type": flag})
        tdms_creator.add_artificial_group(group_name)

    tdms_creator.write()


def _create_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    tdms_creator = _event_data_creator(tdms_dir_path=created_tdms_files_dir,
                                       hdf_dir_path=created_hdf_files_dir,
                                       file_stem=Path("EventData_20210101_corrupt"))

    chn_to_alter = list(tdms_creator.ch_data_dict.keys())[6]
    tdms_creator.ch_data_dict[chn_to_alter][2] = np.NaN
    tdms_creator.add_artificial_group("LogTest_corrupt_NaNvalue")

    tdms_creator.ch_data_dict.update({chn_to_alter: np.array(range(11), dtype=np.float64)})
    tdms_creator.add_artificial_group("LogTest_corrupt_len")

    tdms_creator.write()


def _create_semi_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    tdms_creator = _event_data_creator(tdms_dir_path=created_tdms_files_dir,
                                       hdf_dir_path=created_hdf_files_dir,
                                       file_stem=Path("EventData_20210101_semicorrupt"))

    tdms_creator.add_artificial_group("2021.01.01-00:00:00.000_ok_normal")

    chn_list = ["a", "b", "c"]
    tdms_creator.ch_prop_dict = {chn: {"wf_samples": 5} for chn in chn_list}
    tdms_creator.ch_data_dict = {}
    for chn in chn_list:
        data_size = tdms_creator.ch_prop_dict[chn]["wf_samples"]
        tdms_creator.ch_data_dict.update({chn: [2 ** 32 + i for i in range(data_size)]})
    tdms_creator.add_artificial_group("LogTest_corrupt_chn")

    tdms_creator.write()


def create_event_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    """
    runs all the event data creators and creates tdms and hdf files for testing in the specified directories
    :param created_tdms_files_dir: the destination directory for the tdms files
    :param created_hdf_files_dir: the destination directory for the hdf files
    """
    _create_empty(created_tdms_files_dir, created_hdf_files_dir)
    _create_semi_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_ok_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)
