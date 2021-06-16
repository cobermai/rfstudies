"""
tests for gather module
"""
from typing import Callable
from pathlib import Path
from functools import partial
from multiprocessing import Lock
import h5py
import pytest
from src.utils.transf_tools import gather


def _sanity_func(file_path: Path, hdf_path: str) -> bool:
    """
    a function to fulfill for testing purposes
    :param file_path: file path of the hdf file
    :param hdf_path: hdf-path of the hdf object
    """
    h5py.File(file_path, "r").close()  # check if reading is possible (if hdf file is valid)
    if "discard" in hdf_path:
        ret = False
    else:
        ret = True
    return ret


def test__get_ext_link_rec(tmp_path) -> None:
    """tests _get_ext_link_rek function"""
    # ARRANGE
    hdf_file_path = tmp_path / "test_file.hdf"
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
    output = gather._get_ext_link_rec(hdf_file_path, "/", 2, _sanity_func)  # pylint: disable=protected-access
    with pytest.raises(ValueError):
        gather._get_ext_link_rec(hdf_file_path, "/", -10, lambda _, __: True)  # pylint: disable=protected-access

    # ASSERT
    def link_tuple_set(link_set) -> set:
        return {(link.filename, link.path) for link in link_set}
    assert link_tuple_set(output) == link_tuple_set(expected_output), \
        f"expected {expected_output}\nbut got {output}"


def test__hdf_write_ext_links(tmp_path_factory) -> None:
    """tests _hdf_write_ext_links function"""
    # ARRANGE
    data_dir_path = tmp_path_factory.mktemp("data")

    dest_file_path = data_dir_path / "dest.hdf"
    h5py.File(dest_file_path, "w").close()

    source_dir_path = tmp_path_factory.mktemp("src_files")
    source_file_path = source_dir_path / "test.hdf"

    with h5py.File(source_file_path, "w") as file:
        file.create_group("aaa")
        file["aaa"].create_dataset("ds_at_layer_1", (1,), int)

    # ACT
    gather._hdf_write_ext_links(source_file_path, dest_file_path,  # pylint: disable=protected-access
                                Lock(), 1, lambda x, y: True)

    # ASSERT
    with h5py.File(dest_file_path, "r") as dest_file, h5py.File(source_file_path, "r") as source_file:
        assert len(dest_file.keys()) == 1, "too many keys in output list, expected 1"
        expected_output = source_file["aaa"]
        output = dest_file["test--aaa"]
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"


def test__get_func_to_fulfill():
    """tests _get_func_to_fulfill (i.e. checks if the error handling with on_error works)"""
    # ### unexpected Errors
    # ARRANGE
    def func_unexpected_error(_file_path: Path, _hdf_path: str):
        raise InterruptedError  # an unexpected error
    # ACT
    func_to_fulfill_with_error_handling = gather\
        ._get_func_to_fulfill(False, func_unexpected_error)  # pylint: disable=protected-access
    # ASSERT
    with pytest.raises(InterruptedError):
        func_to_fulfill_with_error_handling(Path("/"), "/")

    # ### expected Errors
    # ARRANGE
    error_list = [KeyError, ValueError, SystemError, ArithmeticError, AttributeError, LookupError,
                  NotImplementedError, RuntimeError]
    on_error = True

    def func_expected_errors(_file_path: Path, _hdf_path: str, expected_error: BaseException):
        raise expected_error
    for err in error_list:
        on_error = not on_error  # Trys out different error handlers
        fun: Callable = partial(func_expected_errors, expected_error=err)
        # ACT
        func_to_fulfill_with_error_handling = gather\
            ._get_func_to_fulfill(on_error=on_error, func_to_fulfill=fun)  # pylint: disable=protected-access
        # ASSERT
        assert func_to_fulfill_with_error_handling(Path("/"), "/") == on_error


def test_gather(tmp_path_factory):
    """tests run with external links"""
    # ARRANGE
    dest_file_path = tmp_path_factory.mktemp("dest") / "TestDataExtLinks.hdf"
    source_dir_path = tmp_path_factory.mktemp("src_files")
    for index in range(3):
        source_file_path = source_dir_path / f"test{index}.hdf"
        with h5py.File(source_file_path, "w") as file:
            file.create_group("grp")
            file["grp"].create_dataset("ds_at_layer_1", (1,), int)
            file.create_group("discard_this")
    expected_keys = {'test0--grp', 'test1--grp', 'test2--grp'}

    # ACT
    gather.gather(src_file_paths=source_dir_path.glob("*.hdf"),
                  dest_file_path=dest_file_path,
                  if_fulfills=_sanity_func,
                  on_error=False)
    # ASSERT
    with h5py.File(dest_file_path, "r") as file:
        output = set(file.keys())
    assert output == expected_keys, f"expected{expected_keys}\nbut got {output}"
