from tests.utils.data_creator.test_files_creator import MakeTestFiles
from pathlib import Path
import numpy as np
import nptdms  # type: ignore

def _event_data_creator(tdms_file_path: Path, hdf_file_path: Path) -> MakeTestFiles:
    tdms_maker = MakeTestFiles(hdf_file_path=hdf_file_path,
                               tdms_file_path=tdms_file_path,
                               root_prop_dict={"name": tdms_file_path.stem, "Version": 2})
    tdms_maker.grp_prop_dict = {
        "BD_BLM TIA": True,
        "BD_DC Down": True,
        "BD_PSR log": True,
        "BD_Col.": True,
        "BD_PKR log": True,
        "BD_BLM": True,
        "BD_PER log": True,
        "BD_DC Up": True,
        "Pulse Count": 1,
        "Timestamp": "2021-01-01T00:00:00.000000",
        "Breakdown Flags": 0,
        "Log Type": 0,
    }
    """
        import numpy as np
        f = nptdms.TdmsFile("/home/lfischl/project_data/CLIC_DATA_Xbox2_T24PSI_2/EventData_20180410.tdms")
        dict_str="{\n"
        for ch in f['Breakdown_2018.04.10-02:33:54.997'].channels():
            dict_str += "\"" + ch.name  + "\" : {"
            for item in ch.properties.items():
                if isinstance(item[1], np.datetime64):
                    pre = "np.datetime64(\""
                    post = "\")"
                elif isinstance(item[1], str):
                    pre  = "\""
                    post = "\""
                else:
                    pre  = ""
                    post = ""
                dict_str += "\"" + item[0] + "\" : " + pre + str(item[1]) + post + ", "
            dict_str += "}, \n"
        dict_str += "}"
        print(dict_str)
        f.close()
    """
    tdms_maker.ch_prop_dict = {
        "PKI Amplitude": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PKI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 168610.0,
                          "Scale_Coeff_c1": -1240300.0, "Scale_Coeff_c2": 52222000.0,
                          "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PKI Phase": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PKI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PSI Amplitude": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 0.0,
                          "Scale_Coeff_c1": -1317300.0, "Scale_Coeff_c2": 78582000.0,
                          "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PSI Phase": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PSR Amplitude": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSR Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                          "wf_xunit_string": "s", },
        "PSR Phase": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PSR Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "PEI Amplitude": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                          "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PEI Amplitude",
                          "NI_UnitDescription": "V", "unit_string": "V", "Scale_Coeff_c0": 0.0,
                          "Scale_Coeff_c1": -378810.0, "Scale_Coeff_c2": 44043000.0, "Scale_Type": "Polynomial",
                          "Scale_Unit": "W", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PEI Phase": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                      "wf_increment": 6.25e-10, "wf_samples": 3200, "NI_ChannelName": "PEI Phase",
                      "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                      "wf_xunit_string": "s", },
        "BLM TIA": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "BLM TIA",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "DC Down": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "DC Down",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "PSR log": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PSR log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "Col.": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                 "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "Col.", "NI_UnitDescription": "V",
                 "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PKR log": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PKR log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "BLM": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "BLM", "NI_UnitDescription": "V",
                "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
        "PER log": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                    "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "PER log",
                    "NI_UnitDescription": "V", "unit_string": "V", "wf_xname": "Time",
                    "wf_xunit_string": "s", },
        "DC Up": {"wf_start_time": np.datetime64("2021-01-01T00:00:00.000000"), "wf_start_offset": 0.0,
                  "wf_increment": 4e-09, "wf_samples": 500, "NI_ChannelName": "DC Up",
                  "NI_UnitDescription": "V",
                  "unit_string": "V", "wf_xname": "Time", "wf_xunit_string": "s", },
    }
    tdms_maker.ch_data_dict = {}
    for chn in tdms_maker.ch_prop_dict.keys():
        data_size = int(tdms_maker.ch_prop_dict[chn]["wf_samples"])
        tdms_maker.ch_data_dict.update({chn: [float(val) for val in range(data_size)]})
    return tdms_maker

def _create_empty(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    file_name = "EventData_20210101_empty"
    tdms_maker = _event_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

def _create_ok_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    """creates a tmds file with the specified path that is similar to event data. And tests the functionality of
    the transformation part of the continuous integration."""
    file_name = "EventData_20210101_ok"
    tdms_maker = _event_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    tdms_maker.grp_prop_dict.update({"Log Type": 0})
    tdms_maker.add_artificial_group("LogGroupTest_ok_normallog")

    tdms_maker.grp_prop_dict.update({"Log Type": 1})
    tdms_maker.add_artificial_group("LogGroupTest_ok_bdin40ms")

    tdms_maker.grp_prop_dict.update({"Log Type": 2})
    tdms_maker.add_artificial_group("LogGroupTest_ok_bdin20ms")

    tdms_maker.grp_prop_dict.update({"Log Type": 3})
    tdms_maker.add_artificial_group("BreakdownGroupTest_ok_db")


def _create_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    file_name = "EventData_20210101_corrupt"
    tdms_maker = _event_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    chn_to_alter = list(tdms_maker.ch_data_dict.keys())[6]
    tdms_maker.ch_data_dict[chn_to_alter][2] = np.NaN
    tdms_maker.add_artificial_group("LogTest_corrupt_NaNvalue")

    tdms_maker.ch_data_dict.update({chn_to_alter: np.array(range(11), dtype=np.float64)})
    tdms_maker.add_artificial_group("LogTest_corrupt_len")


def _create_semi_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    file_name = "EventData_20210101_semicorrupt"
    tdms_maker = _event_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    tdms_maker.add_artificial_group("2021.01.01-00:00:00.000_ok_normal")

    chn_list = ["a", "b", "c"]
    tdms_maker.ch_prop_dict = {chn:{"wf_samples": 5} for chn in chn_list}
    tdms_maker.ch_data_dict = {}
    for chn in chn_list:
        data_size = tdms_maker.ch_prop_dict[chn]["wf_samples"]
        tdms_maker.ch_data_dict.update({chn: [i for i in range(data_size)]})
    tdms_maker.add_artificial_group("LogTest_corrupt_chn")

def create_all(created_tdms_files_dir: Path, created_hdf_files_dir:Path) -> None:
    _create_empty(created_tdms_files_dir, created_hdf_files_dir)
    _create_semi_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_ok_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)