"""
tests for gather module
"""
from src.utils.hdf_utils import gather  # type: ignore

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


def test_hdf_path_combine() -> None:
    """tests hdf_path_combine functionality"""
    desired_outcome = "/a/b"
    assert gather.hdf_path_combine("a", "b") == desired_outcome, f"should be {desired_outcome}"
    desired_outcome = "/a/b"
    assert gather.hdf_path_combine("/a", "b") == desired_outcome, f"should be {desired_outcome}"
    desired_outcome = "/a/b"
    assert gather.hdf_path_combine("/a/", "/b") == desired_outcome, f"should be {desired_outcome}"
    desired_outcome = "faulty outcome"
    assert gather.hdf_path_combine("a", "b") == desired_outcome, f"should be {desired_outcome}"

if __name__ == "__main__":
    test_hdf_path_combine()
