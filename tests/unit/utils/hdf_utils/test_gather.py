"""
tests for gather module
"""
from shutil import rmtree
from datetime import date
from typing import Callable
from pathlib import Path
from functools import partial
import logging
from multiprocessing import Lock
import h5py
import pytest
from src.utils.transf_tools import gather

LOG = logging.getLogger("TESTLOG")


def _sanity_func(file_path: Path, hdf_path: str) -> bool:
    """a function to fulfill for testing purposes"""
    h5py.File(file_path, "r").close()
    if "discard" in hdf_path:
        ret = False
    else:
        ret = True
    return ret


def test__get_ext_link_rek() -> None:
    """tests _get_ext_link_rek function"""
    # ARRANGE
    data_dir_path = Path(__file__).parent / f"data_{date.today()}"
    data_dir_path.mkdir()
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

    # ACT
    output = gather.get_ext_link_rec(hdf_file_path, "/", 2, _sanity_func)
    with pytest.raises(ValueError):
        gather.get_ext_link_rec(hdf_file_path, "/", -10, lambda _, __: True)

    # ASSERT
    def link_tuple_set(link_set) -> set:
        return {(link.filename, link.path) for link in link_set}
    assert link_tuple_set(output) == link_tuple_set(expected_output), \
        f"expected {expected_output}\nbut got {output}"

    # CLEAN
    rmtree(data_dir_path)


def test__hdf_write_ext_links() -> None:
    """tests _hdf_write_ext_links function"""
    # ARRANGE
    data_dir_path = Path(__file__).parent / f"data_{date.today()}"
    data_dir_path.mkdir()
    dest_file_path = data_dir_path / "dest.hdf"
    h5py.File(dest_file_path, "w").close()
    source_dir_path = data_dir_path / "source"
    source_dir_path.mkdir()
    source_file_path = source_dir_path / "test.hdf"
    with h5py.File(source_file_path, "w") as file:
        file.create_group("aaa")
        file["aaa"].create_dataset("ds_at_layer_1", (1,), int)

    # ACT
    gather.hdf_write_ext_links(source_file_path, dest_file_path, Lock(), 1, lambda x, y: True)

    # ASSERT
    with h5py.File(dest_file_path, "r") as dest_file, h5py.File(source_file_path, "r") as source_file:
        assert len(dest_file.keys()) == 1, "too many keys in output list, expected 1"
        expected_output = source_file["aaa"]
        output = dest_file["test--aaa"]
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"

    # CLEAN
    rmtree(data_dir_path)


class TestGather:
    """ testing the Gather class"""
    @staticmethod
    def test_set_func_to_fulfill():
        """checks what happens when an error occurs"""
        # unexpected Errors
        # ARRANGE
        gather_obj = gather.Gather(depth=1, num_processes=2)
        def func_unexpected_error(_file_path: Path, _hdf_path: str):
            raise InterruptedError  # an unexpected error
        # ACT
        gather_obj.set_func_to_fulfill(False, func_unexpected_error)
        # ASSERT
        with pytest.raises(InterruptedError):
            gather_obj.func_to_fulfill(Path("/"), "/")

        # expected Errors
        # ARRANGE
        error_list = [KeyError, ValueError, SystemError, ArithmeticError, AttributeError, LookupError,
                      NotImplementedError, RuntimeError]
        on_error = True

        def func_expected_errors(_file_path: Path, _hdf_path: str, expected_error: BaseException):
            raise expected_error
        for err in error_list:
            on_error = not on_error  # Trying out different error handlers
            fun: Callable = partial(func_expected_errors, expected_error=err)
            # ACT
            gather_obj.set_func_to_fulfill(on_error=on_error, func_to_fulfill=fun)
            # ASSERT
            assert gather_obj.func_to_fulfill(Path("/"), "/") == on_error

    @staticmethod
    def test_from_files():
        """testing from files"""
        gather_obj = gather.Gather(depth=1, num_processes=2)
        assert gather_obj.from_files({Path("/"), Path("another/path")})\
            .source_file_paths == {Path("/"), Path("another/path")}

    @staticmethod
    def test_to_hdf_file():
        """testing to hdf file"""
        # ARRANGE
        gather_obj = gather.Gather(depth=1, num_processes=2)
        data_dir_path = Path(__file__).parent / f"data_{date.today()}"
        data_dir_path.mkdir()
        # ACT
        hdf_file_path = data_dir_path / "test.hdf"
        # ASSERT
        assert gather_obj.to_hdf_file(hdf_file_path)\
                   .dest_file_path == hdf_file_path
        # CLEAN
        rmtree(data_dir_path)

    @staticmethod
    def test_if_fulfills():
        """testing if fulfills"""
        # ARRANGE
        gather_obj = gather.Gather(depth=1, num_processes=2)
        def my_func(file_path: Path, hdf_path: str):
            raise RuntimeError
        # ACT AND ASSERT
        for on_error in [True, False]:
            expected_output = on_error
            assert gather_obj.if_fulfills(my_func, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output
            expected_output = True
            assert gather_obj.if_fulfills(lambda x, y: True, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output
            expected_output = False
            assert gather_obj.if_fulfills(lambda x, y: False, on_error)\
                       .func_to_fulfill(Path("/"), "/") == expected_output

    @staticmethod
    def test_run_with_external_links():
        """testing run with external links"""
        # ARRANGE
        gather_obj = gather.Gather()
        data_dir_path = Path(__file__).parent / f"data_{date.today()}"
        data_dir_path.mkdir()
        dest_file_path = data_dir_path / "TestDataExtLinks.hdf"
        h5py.File(dest_file_path, "w").close()
        source_dir_path = data_dir_path / "source_dir"
        source_dir_path.mkdir()
        for index in range(3):
            source_file_path = source_dir_path / f"test{index}.hdf"
            with h5py.File(source_file_path, "w") as file:
                file.create_group("grp")
                file["grp"].create_dataset("ds_at_layer_1", (1,), int)
                file.create_group("discard_this")
        expected_keys = {'test0--grp', 'test1--grp', 'test2--grp'}
        # ACT
        gather_obj.from_files(source_dir_path.glob("*.hdf")) \
            .to_hdf_file(dest_file_path) \
            .if_fulfills(_sanity_func, on_error=False) \
            .run_with_external_links()
        # ASSERT
        with h5py.File(dest_file_path, "r") as file:
            output = set(file.keys())
        assert output == expected_keys, f"expected{expected_keys}\nbut got {output}"
        # CLEAN
        rmtree(data_dir_path)


if __name__ == "__main__":
    test__get_ext_link_rek()
    test__hdf_write_ext_links()
    TestGather()
