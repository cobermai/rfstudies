"""
tests for gather module
"""
from pathlib import Path
import logging
from multiprocessing import Lock
import h5py
import pytest
from src.utils.hdf_utils import gather
from tests.utils.dir_handler import remkdir
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

def _sanity_func(file_path: Path, hdf_path: str) -> bool:
    """a function to fulfill for testing purposes"""
    h5py.File(file_path, "r").close()
    if "discard" in hdf_path:
        ret = False
    else:
        ret = True
    return ret

def test__hdf_path_combine() -> None:
    """tests hdf_path_combine function"""
    ### ARRANGE
    arg_list = (("a", "b", "c"),
                 ("/a/b", "/c"),
                 ("/a///b///", "/c"))
    expected_output = "/a/b/c"
    ### ACT + ASSERT
    for args in arg_list:
        output = gather.hdf_path_combine(*args)
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"

def test__get_ext_link_rek() -> None:
    """tests _get_ext_link_rek function"""
    ### ARRANGE
    data_dir_path = remkdir(Path(__file__).parent / "data")
    hdf_file_path = data_dir_path / "test_file.hdf"
    with h5py.File(hdf_file_path, "w") as file:
        file.create_group("aaa")
        file["aaa"].create_dataset("ds_at_layer_1", (1,), int)
        file.create_group("asdf")
        file["asdf"].create_group("discard_this_group")
        file["asdf"].create_group("qwer")
        file["asdf/qwer"].create_dataset("ds_at_layer_2", (1,), int)
    expected_output = {h5py.ExternalLink(hdf_file_path, "/asdf/qwer"),
                       h5py.ExternalLink(hdf_file_path, "/aaa/ds_at_layer_1")}

    ### ACT
    output = gather.get_ext_link_rec(hdf_file_path, "/", 2, _sanity_func)
    with pytest.raises(ValueError):
        gather.get_ext_link_rec(hdf_file_path, "/", -10, lambda x, y: True)

    ### ASSERT
    link_tuple_set = lambda link_set: {(link.filename, link.path) for link in link_set}
    assert link_tuple_set(output) == link_tuple_set(expected_output), \
        f"expected {expected_output}\nbut got {output}"

def test__hdf_write_ext_links() -> None:
    """tests _hdf_write_ext_links function"""
    ### ARRANGE
    data_dir_path = remkdir(Path(__file__).parent / "data")
    dest_file_path = data_dir_path / "dest.hdf"
    h5py.File(dest_file_path, "w").close()
    source_file_path = remkdir(data_dir_path / "source_dir") / "test.hdf"
    with h5py.File(source_file_path, "w") as file:
        file.create_group("aaa")
        file["aaa"].create_dataset("ds_at_layer_1", (1,), int)

    ### ACT
    gather.hdf_write_ext_links(source_file_path, dest_file_path, Lock(), 1, lambda x, y: True)

    ### ASSERT
    with h5py.File(dest_file_path, "r") as dest_file, h5py.File(source_file_path, "r") as source_file:
        assert len(dest_file.keys())==1, "too many keys in output list, expected 1"
        expected_output = source_file["aaa"]
        output = dest_file["test--aaa"]
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"

class TestGather():
    """ testing the Gather class"""
    gather = gather.Gather(depth=1, num_processes=2)

    def test_set_func_to_fulfill(self):
        """checks what happens when an error occures"""
        # an unexpected error
        def func_unexpected_error(_file_path:Path, _hdf_path:str):
            raise InterruptedError
        self.gather.set_func_to_fulfill(False, func_unexpected_error)
        with pytest.raises(InterruptedError):
            self.gather.func_to_fulfill(Path("/"), "/")

        # list of all expected errors
        error_list = [KeyError,ValueError, SystemError, ArithmeticError, AttributeError, LookupError,
                       NotImplementedError, RuntimeError]
        on_error = True
        for err in error_list:
            on_error = not on_error
            def func_expected_errors(_file_path:Path, _hdf_path:str):
                raise err  # pylint: disable=cell-var-from-loop
            self.gather.set_func_to_fulfill(on_error, func_expected_errors)
            assert self.gather.func_to_fulfill(Path("/"), "/") == on_error

    def test_from_files(self):
        """testing from files"""
        assert self.gather.from_files({Path("/"), Path("another/path")})\
                   .source_file_paths  == {Path("/"), Path("another/path")}

    def test_to_hdf_file(self):
        """testing to hdf file"""
        assert self.gather.to_hdf_file(Path("a/random/path"))\
                   .dest_file_path == Path("a/random/path")

    def test_if_fulfills(self):
        """testing if fulfills"""

        def my_func(file_path: Path, hdf_path: str):
            raise RuntimeError
        for on_error in [True, False]:
            expected_output = on_error
            assert self.gather.if_fulfills(my_func, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output
            expected_output = True
            assert self.gather.if_fulfills(lambda x, y: True, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output
            expected_output = False
            assert self.gather.if_fulfills(lambda x, y: False, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output

    def test_run_with_external_links(self):
        """tetsing run with external links"""
        ### ARRANGE
        data_dir_path = remkdir(Path(__file__).parent / "data")
        dest_file_path = data_dir_path / "TestDataExtLinks.hdf"
        h5py.File(dest_file_path, "w").close()
        source_dir_path = remkdir(data_dir_path / "source_dir")
        for index in range(3):
            source_file_path = source_dir_path / f"test{index}.hdf"
            with h5py.File(source_file_path, "w") as file:
                file.create_group("grp")
                file["grp"].create_dataset("ds_at_layer_1", (1,), int)
                file.create_group("discard_this")
        expected_keys = {'test0--grp', 'test1--grp', 'test2--grp'}
        ### ACT
        self.gather = gather.Gather()
        self.gather \
            .from_files(source_dir_path.glob("*.hdf")) \
            .to_hdf_file(dest_file_path) \
            .if_fulfills(_sanity_func, on_error=False) \
            .run_with_external_links()
        ### ASSERT
        with h5py.File(dest_file_path, "r") as file:
            output = set(file.keys())
        assert output == expected_keys, f"expected{expected_keys}\nbut got {output}"


if __name__ == "__main__":
    test__hdf_path_combine()
    test__get_ext_link_rek()
    test__hdf_write_ext_links()
    TestGather()
