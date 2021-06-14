"""this module tests the functions from hdf_tools"""
from src.utils.hdf_tools import hdf_path_combine


def test__hdf_path_combine() -> None:
    """tests hdf_path_combine function"""
    # ARRANGE
    arg_list = (("a", "b", "c"),
                ("/a/b", "/c"),
                ("/a///b///", "/c"))
    expected_output = "/a/b/c"
    # ACT + ASSERT
    for args in arg_list:
        output = hdf_path_combine(*args)
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"
