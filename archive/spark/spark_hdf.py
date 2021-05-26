"""
This code is meant to try selecting data from hdf files with a spark rdd.
"""
import sys
from pathlib import Path
import h5py
from pyspark import SparkContext

sc = SparkContext()
partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
hdf_file_name = Path("spark_please_read_me.h5").absolute()
with h5py.File(hdf_file_name, "w") as file:
    for k in range(16):
        file.create_dataset(f"ds{k}", data=[i + k for i in range(11)])

def fun(file_name: Path, hdf_path: str):
    """function to read hdf data from file_name at hdf_path"""
    with h5py.File(file_name, "r") as _file:
        return list(_file[hdf_path][:])  # list(f[hdf_path][:])

rdd = sc.parallelize((f"/ds{k}" for k in range(16))).map(lambda x: fun(file_name=hdf_file_name, hdf_path=x))

print("\n")
print(f"count {rdd.count()}")
print(f"min {rdd.min()}")
# doesnt work # print(f"mean {rdd.mean()}")
# doenst work # print(f"stdev {rdd.stdev()}")
print(f"max {rdd.max()}")

sc.stop()
