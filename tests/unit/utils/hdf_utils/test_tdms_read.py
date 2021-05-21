"""module to test the tdms_reader"""
import os
from pathlib import Path
from shutil import rmtree
import pytest
from src.utils.hdf_utils import tdms_read
from tests.utils.dir_handler import remkdir
from tests.utils.data_creator.file_creator_for_testing import CreatorTestFiles
from tests.utils.data_creator.tdms_file_creator import CreatorTdmsFile


def test_convert_file():
    """tests the convert_file function"""
    # ARRANGE
    data_dir_path = remkdir(Path(__file__).parent / "data")
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
    tdms_read.convert_file(tdms_file_path=tdms_file_path, hdf_dir=data_dir_path)

    # ASSERT
    path_of_output = data_dir_path / tdms_file_path.with_suffix(".hdf").name
    is_equal = os.system(f"h5diff  {path_of_output} {path_of_expected}") == 0
    assert is_equal


class TestConvert:
    """tests teh Convert class"""
    @staticmethod
    def test_from_tdms():
        """tests from_tdms"""
        conv = tdms_read.Convert()
        assert conv.check_already_converted is True
        assert conv.num_processes == 2

        conv = tdms_read.Convert(check_already_converted=False, num_processes=100)
        assert conv.check_already_converted is False
        assert conv.num_processes == 100

        conv_from_tdms = conv.from_tdms(tdms_dir=Path("/"))
        assert isinstance(conv_from_tdms, tdms_read.ConvertFromTdms)
        assert conv_from_tdms.converter == conv
        assert conv_from_tdms.tdms_dir == Path("/")

    @staticmethod
    def test_run():
        """tests run"""
        with pytest.raises(NotImplementedError):
            tdms_read.Convert().run()


class TestConvertFromTdms:
    """tests the ConvertFromTdms class"""
    @staticmethod
    def test_to_hdf():
        """tests to_hdf"""
        conv = tdms_read.Convert()
        conv_from_tdms = conv.from_tdms(tdms_dir=Path("/path/to/tdms/dir"))
        cft2h = conv_from_tdms.to_hdf(Path("/path/to/hdf/dir"))
        assert isinstance(cft2h, tdms_read.ConvertFromTdmsToHdf)
        assert cft2h.hdf_dir == Path("/path/to/hdf/dir")
        assert cft2h.tdms_dir == Path("/path/to/tdms/dir")
        assert cft2h.converter == conv

    @staticmethod
    def test_run():
        """tests run"""
        with pytest.raises(NotImplementedError):
            tdms_read.Convert().from_tdms(Path("/path/to/tdms/dir")).run()


class TestConvertFromTdmsToHdf:
    """tests the ConvertFromTdmsToHdf class"""

    @staticmethod
    def test_get_tdms_file_paths_to_convert():
        """tests get_tdms_file_paths_to_convert"""
        data_dir_path = remkdir(Path(__file__).parent / "data")
        hdf_dir_path = remkdir(data_dir_path / "hdf")
        tdms_dir_path = remkdir(data_dir_path / "tdms")
        os.system(command=f"""
            hdf_path={hdf_dir_path}
            tdms_path={tdms_dir_path}
            for file_stem in test1_ok test2_ok test_discard
            do
                echo > $tdms_path/$file_stem.tdms
            done
            echo > $hdf_path/test_discard.hdf
        """)
        conv_tdms2hdf = tdms_read.Convert()\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == set(tdms_dir_path.glob("*ok.tdms"))
        conv_tdms2hdf = tdms_read.Convert(check_already_converted=False)\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == set(tdms_dir_path.glob("*.tdms"))
        rmtree(data_dir_path, ignore_errors=True)

    @staticmethod
    def test_run():
        """tests run"""
        # ARRANGE
        data_dir_path = remkdir(Path(__file__).parent / "data")
        tdms_dir_path = remkdir(data_dir_path / "tdms")
        hdf_dir_path = remkdir(data_dir_path / "hdf")
        for index in range(5):
            CreatorTdmsFile(tdms_dir_path/f"test{index}.tdms", {"root_prop": index}).write()
        for num_processes in [1, 3]:
            conv_tdms2hdf = tdms_read.Convert(num_processes=num_processes, check_already_converted=False)\
                .from_tdms(tdms_dir_path)\
                .to_hdf(hdf_dir_path)
            # ACT
            conv_tdms2hdf.run()
            assert {p.stem for p in hdf_dir_path.glob("*.hdf")} == {p.stem for p in tdms_dir_path.glob("*.tdms")}
            remkdir(hdf_dir_path)
        rmtree(data_dir_path, ignore_errors=True)