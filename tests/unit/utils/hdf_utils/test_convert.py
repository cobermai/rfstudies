"""module to test the convert.py"""
import os
from pathlib import Path
import h5py
from src.utils.transf_tools import convert
from tests.utils.data_creator.file_creator_for_testing import CreatorTestFiles
from tests.utils.data_creator.tdms_file_creator import CreatorTdmsFile


def test__convert_file(tmp_path):
    """tests the _convert_file function"""
    # ARRANGE
    data_dir_path = tmp_path
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
    convert._convert_file(tdms_file_path=tdms_file_path, hdf_dir=data_dir_path)  # pylint: disable=protected-access

    # ASSERT
    path_of_output = data_dir_path / tdms_file_path.with_suffix(".hdf").name
    print(f"h5diff {path_of_output} {path_of_expected}")
    is_equal = os.system(f"h5diff {path_of_output} {path_of_expected}") == 0
    assert is_equal


class TestConvert:
    """tests the Convert class"""
    @staticmethod
    def test_init():
        """Tests the initialisation of the Convert class"""
        # ARRANGE ACT
        conv = convert.Convert()
        # ASSERT
        assert conv.check_already_converted is True
        assert conv.num_processes == 2

        # ARRANGE ACT
        conv = convert.Convert(check_already_converted=False, num_processes=100)
        # ASSERT
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
    def test_get_tdms_file_paths_to_convert(tmp_path_factory):
        """tests get_tdms_file_paths_to_convert"""
        # ARRANGE
        tdms_dir_path = tmp_path_factory.mktemp("tdms_files")
        hdf_dir_path = tmp_path_factory.mktemp("hdf_files")
        for index in [1, 2, 3]:
            open(tdms_dir_path / f"test{index}.tdms", "w")
        open(hdf_dir_path / "test1.hdf", "w")

        expected = set(tdms_dir_path.glob("*.tdms"))

        # ACT
        conv_tdms2hdf = convert.Convert(check_already_converted=True)\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        # ASSERT
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with no healthy hdf file, with check"

        # ACT
        h5py.File(hdf_dir_path / "test1.hdf", "w").close()
        conv_tdms2hdf = convert.Convert(check_already_converted=False)\
            .from_tdms(tdms_dir_path)\
            .to_hdf(hdf_dir_path)
        # ASSERT
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with one healthy hdf file, no check"

        # ACT
        expected = set(tdms_dir_path.glob("*2.tdms")).union(set(tdms_dir_path.glob("*3.tdms")))
        conv_tdms2hdf = convert.Convert(check_already_converted=True) \
            .from_tdms(tdms_dir_path) \
            .to_hdf(hdf_dir_path)
        # ASSERT
        assert conv_tdms2hdf.get_tdms_file_paths_to_convert() == expected, "with one healthy hdf file, with check"

    @staticmethod
    def test_run(tmp_path_factory):
        """tests run"""
        # ARRANGE
        hdf_dir_path = tmp_path_factory.mktemp("hdf_files")
        tdms_dir_path = tmp_path_factory.mktemp("tmp_files")
        for index in range(5):
            CreatorTdmsFile(tdms_dir_path / f"test{index}.tdms", {"root_prop": index}).write()
        for num_processes in [1, 3]:
            # ACT
            convert.Convert(num_processes=num_processes, check_already_converted=False)\
                .from_tdms(tdms_dir_path)\
                .to_hdf(hdf_dir_path)\
                .run()
            # ASSERT
            assert {p.stem for p in hdf_dir_path.glob("*.hdf")} == {p.stem for p in tdms_dir_path.glob("*.tdms")}
