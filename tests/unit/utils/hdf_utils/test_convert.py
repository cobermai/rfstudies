"""module to test the convert.py"""
import os
from pathlib import Path
from shutil import rmtree
from datetime import date
import pytest
import h5py
from src.utils.transf_tools import convert
from tests.utils.data_creator.file_creator_for_testing import CreatorTestFiles
from tests.utils.data_creator.tdms_file_creator import CreatorTdmsFile


def test_convert_file():
    """tests the convert_file function"""
    # ARRANGE
    data_dir_path = Path(__file__).parent / f"data_{date.today()}"
    data_dir_path.mkdir()
    tdms_file_path = data_dir_path / "test.tdms"
    path_of_expected = data_dir_path / "expected.hdf"
    test_creator = CreatorTestFiles(hdf_file_path=path_of_expected,
                                    tdms_file_path=tdms_file_path,
                                    root_prop_dict={"root_prop_key": 0})
    test_creator.grp_prop_dict = {"grp_prop_key": 1}
    test_creator.ch_prop_dict = {"ch1": {"ch_prop_key": 2}, "ch2": {"ch_prop_key": 2}}
    test_creator.ch_data_dict = {"ch1": [1, 2, 3], "ch2": [4, 5, 6]}
    test_creator.add_artificial_group("grp1")
    test_creator.write()

    # ACT
    convert.convert_file(tdms_file_path=tdms_file_path, hdf_dir=data_dir_path)

    # ASSERT
    path_of_output = data_dir_path / tdms_file_path.with_suffix(".hdf").name
    print(f"h5diff {path_of_output} {path_of_expected}")
    is_equal = os.system(f"h5diff {path_of_output} {path_of_expected}") == 0
    assert is_equal

    # CLEAN
    rmtree(data_dir_path)


class TestConvert:
    """tests teh Convert class"""
    @staticmethod
    def test_init():
        """Testing the inizialisation of the Convert class"""
        # Arrange Act
        conv = convert.Convert()
        # Assert
        assert conv.check_already_converted is True
        assert conv.num_processes == 2

        # Arrange Act
        conv = convert.Convert(check_already_converted=False, num_processes=100)
        # Assert
        assert conv.check_already_converted is False
        assert conv.num_processes == 100
    @staticmethod
    def test_from_tdms():
        """tests from_tdms"""
        # ARRANGE
        conv = convert.Convert(check_already_converted=False, num_processes=100)
        # ACT
        conv_from_tdms = conv.from_tdms(tdms_dir=Path("/"))
        # ASSERT
        assert isinstance(conv_from_tdms, convert.ConvertFromTdms)
        assert conv_from_tdms.tdms_dir == Path("/")
        assert conv_from_tdms.num_processes == 100
        assert conv_from_tdms.check_already_converted is False


class TestConvertFromTdms:
    """tests the ConvertFromTdms class"""
    @staticmethod
    def test_to_hdf():
        """tests to_hdf"""
        # ARRANGE
        conv = convert.Convert(num_processes=20, check_already_converted=False)
        tdms_dir_path = Path("/path/to/tdms/dir")
        hdf_dir_path = Path("/path/to/hdf/dir")
        conv_from_tdms = conv.from_tdms(tdms_dir=tdms_dir_path)

        # ACT
        cft2h = conv_from_tdms.to_hdf(hdf_dir_path)

        # ASSERT
        assert isinstance(cft2h, convert.ConvertFromTdmsToHdf)
        assert cft2h.hdf_dir == hdf_dir_path
        assert cft2h.tdms_dir == tdms_dir_path
        assert cft2h.num_processes == 20
        assert cft2h.check_already_converted is False


class TestConvertFromTdmsToHdf:
    """tests the ConvertFromTdmsToHdf class"""
    @staticmethod
    def test_get_tdms_file_paths_to_convert():
        """tests get_tdms_file_paths_to_convert"""
        # ARRANGE
        data_dir_path = Path(__file__).parent / f"data_{date.today()}"
        data_dir_path.mkdir()
        hdf_dir_path = data_dir_path / "hdf"
        hdf_dir_path.mkdir()
        tdms_dir_path = data_dir_path / "tdms"
        tdms_dir_path.mkdir()
        os.system(command=f"""
            hdf_path={hdf_dir_path}
            tdms_path={tdms_dir_path}
            for file_stem in test1 test2 test3
            do
                echo > $tdms_path/$file_stem.tdms
            done
            echo > $hdf_path/test1.hdf
        """)
        expected = set(tdms_dir_path.glob("*.tdms"))

        # ACT AND ASSERT
        conv_tdms2hdf = convert.Convert()\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with a faulty hdf file with check"

        h5py.File(hdf_dir_path / "test1.hdf", "w").close()
        conv_tdms2hdf = convert.Convert(check_already_converted=False)\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with healthy hdf file no check"

        expected = set(tdms_dir_path.glob("*2.tdms")).union(set(tdms_dir_path.glob("*3.tdms")))
        conv_tdms2hdf = convert.Convert(check_already_converted=True) \
            .from_tdms(tdms_dir_path) \
            .to_hdf(hdf_dir_path)
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with a healthy hdf file with check"

        # CLEAN
        rmtree(data_dir_path, ignore_errors=True)

    @staticmethod
    def test_run():
        """tests run"""
        # ARRANGE
        data_dir_path = Path(__file__).parent / f"data_{date.today()}"
        data_dir_path.mkdir()
        hdf_dir_path = data_dir_path / "hdf"
        hdf_dir_path.mkdir()
        tdms_dir_path = data_dir_path / "tdms"
        tdms_dir_path.mkdir()
        for index in range(5):
            CreatorTdmsFile(tdms_dir_path/f"test{index}.tdms", {"root_prop": index}).write()
        for num_processes in [1, 3]:
            conv_tdms2hdf = convert.Convert(num_processes=num_processes, check_already_converted=False)\
                .from_tdms(tdms_dir_path)\
                .to_hdf(hdf_dir_path)
            # ACT
            conv_tdms2hdf.run()
            assert {p.stem for p in hdf_dir_path.glob("*.hdf")} == {p.stem for p in tdms_dir_path.glob("*.tdms")}
        rmtree(data_dir_path, ignore_errors=True)
