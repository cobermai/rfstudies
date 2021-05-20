"""
tests for gather module
"""
import pytest
import h5py
import logging
from pathlib import Path
from src.utils.hdf_utils import gather
from tests.utils.dir_handler import get_clean_data_dir, mkdir_ret
from multiprocessing import Lock
log = logging.getLogger("TESTLOG")
#import unittest
#class TestGather(unittest.TestCase):
#    def test__hdf_path_combine(self):
#        desired_outcome = "/a/b"
#        self.assertEqual(gather.hdf_path_combine("a", "b"), desired_outcome, f"should be {desired_outcome}")
#
#        desired_outcome = "/a/b"
#        self.assertEqual(gather.hdf_path_combine("/a", "b"), desired_outcome, f"should be {desired_outcome}")
#
#        desired_outcome = "/a/b"
#        self.assertEqual(gather.hdf_path_combine("/a/", "/b"), desired_outcome, f"should be {desired_outcome}")
#
#        desired_outcome = "faulty outcome"
#        self.assertEqual(gather.hdf_path_combine("a", "b"), desired_outcome, f"should be {desired_outcome}")

def sanity_func(file_path: Path, hdf_path: str) -> bool:
    if "discard" in hdf_path:
        ret = False
    else:
        ret = True
    return ret

def test__hdf_path_combine() -> None:
    """tests hdf_path_combine function"""
    ### ARRANGE
    input_list = (("a", "b", "c"),
                 ("/a/b", "/c"),
                 ("/a///b///", "/c"))
    expected_output = "/a/b/c"
    ### ACT + ASSERT
    for input in input_list:
        output = gather.hdf_path_combine(*input)
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"

def test__get_ext_link_rek() -> None:
    """tests _get_ext_link_rek function"""
    ### ARRANGE
    data_dir_path = get_clean_data_dir(__file__)
    hdf_file_path = data_dir_path.with_name("test_file.hdf")
    with h5py.File(hdf_file_path, "w") as fl:
        fl.create_group("aaa")
        fl["aaa"].create_dataset("ds_at_layer_1", (1,), int)
        fl.create_group("asdf")
        fl["asdf"].create_group("discard_this_group")
        fl["asdf"].create_group("qwer")
        fl["asdf/qwer"].create_dataset("ds_at_layer_2", (1,), int)
    expected_output = {h5py.ExternalLink(hdf_file_path, "/asdf/qwer"),
                       h5py.ExternalLink(hdf_file_path, "/aaa/ds_at_layer_1")}

    ### ACT
    output = gather._get_ext_link_rek(hdf_file_path, "/", 2, sanity_func)
    with pytest.raises(ValueError):
        gather._get_ext_link_rek(hdf_file_path, "/", -10, lambda x, y: True)

    ### ASSERT
    link_tuple_set = lambda link_set: {(link.filename, link.path) for link in link_set}
    assert link_tuple_set(output) == link_tuple_set(expected_output), \
        f"expected {expected_output}\nbut got {output}"


def test__hdf_write_ext_links() -> None:
    """tests _hdf_write_ext_links function"""
    ### ARRANGE
    data_dir_path = get_clean_data_dir(__file__)
    dest_file_path = data_dir_path / "dest.hdf"
    h5py.File(dest_file_path, "w").close()
    source_file_path = mkdir_ret(data_dir_path / "source_dir") / "test.hdf"
    with h5py.File(source_file_path, "w") as fl:
        fl.create_group("aaa")
        fl["aaa"].create_dataset("ds_at_layer_1", (1,), int)

    ### ACT
    gather._hdf_write_ext_links(source_file_path, dest_file_path, Lock(), 1, lambda x, y: True)

    ### ASSERT
    with h5py.File(dest_file_path, "r") as dest_fl, h5py.File(source_file_path, "r") as source_fl:
        assert len(dest_fl.keys())==1, "too many keys in output list, expected 1"
        expected_output = source_fl["aaa"]
        output = dest_fl["test:aaa"]
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"


#class test_Gather():
#    """ testing the Gather class"""
#    ### ARRANGE
#    data_dir_path = get_clean_data_dir(__file__)
#    dest_file_path = data_dir_path / "dest.hdf"
#    h5py.File(dest_file_path, "w").close()
#    source_dir_path = mkdir_ret(data_dir_path / "source_dir")
#    for index in range(3):
#        source_file_path = source_dir_path / f"test{index}.hdf"
#        with h5py.File(source_file_path, "w") as fl:
#            fl.create_group("grp")
#            fl["aaa"].create_dataset("ds_at_layer_1", (1,), int)
#            fl.create_group("discard_this")
#    ### ACT
#    gather.Gather(num_processes=2) \
#        .from_files(source_dir_path.glob("data/*.hdf")) \
#        .to_hdf_file(data_dir_path/ "TestDataExtLinks.hdf") \
#        .if_fulfills(sanity_func, on_error=False) \
#        .run_with_external_links()
#    ###ASSERT
#    for index in range(3):
#        source_file_path = source_dir_path / f"test_{index}.hdf"
#        with h5py.File(source_file_path, "w") as fl:
#            for key in fl.keys():
#                source_file_path.stem + ":" + key[1:]
#
def test_() -> None:
    """ """
    ### ARRANGE
    ### ACT
    ###ASSERT

if __name__ == "__main__":
    test__hdf_path_combine()
    test__get_ext_link_rek()
    test__hdf_write_ext_links()