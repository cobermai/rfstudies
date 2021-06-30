from pathlib import Path
import pandas as pd
import typing
from dataclasses import dataclass
import h5py
from src.utils.handler_tools.customfeature import CustomFeature, ColumnWiseFeature, RowWiseFeature
from src.utils.hdf_tools import hdf_path_combine


@dataclass
class ContextDataHandler:
    context_data_file_path: Path
    length: int

    def init_feature(self, feature: CustomFeature):
        with h5py.File(self.context_data_file_path, "a") as file:
            file.create_dataset(feature.full_hdf_path,
                                shape=(self.length,),
                                dtype=feature.output_dtype,
                                chunks=True)
            if feature.info != "":
                file[feature.full_hdf_path].attrs.create(name="info", data=feature.info)


class ColumnWiseContextDataHandler(ContextDataHandler):
    def write(self, feature: ColumnWiseFeature):
        self.init_feature(feature)
        with h5py.File(self.context_data_file_path, "a") as file:
            file[feature.full_hdf_path][:] = feature.vec


class RowWiseContextDataHandler(ContextDataHandler):

    def init_features(self, features):
        for feature in features:
            self.init_feature(feature)

    def init_tsfresh(self, data_gen):
        with h5py.File(self.context_data_file_path, "a") as file:
            for hdf_path, value in data_gen:
                try:
                    file.create_dataset(hdf_path,
                                        shape=(self.length,),
                                        dtype=type(value),
                                        chunks=True)
                except ValueError:
                    print(hdf_path.ljust(40), list(file[hdf_path.split("/")[1]].keys()))
                file[hdf_path].attrs.create(name="info", data="from module tsfresh")

    def write_row(self, index: int, data_gen: typing.Generator):
        with h5py.File(self.context_data_file_path, "a") as file:
            for hdf_path, value in data_gen:
                file[hdf_path][index] = value

