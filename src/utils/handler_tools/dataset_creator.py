import typing
from abc import ABC, abstractmethod
from pathlib import Path
from collections import namedtuple
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.xbox2_specific.datasets import XBOX2_event_bd20ms
from src.xbox2_specific.datasets import XBOX2_trend_bd20ms

class DatasetCreator(ABC):
    """
    Template to create dataset
    """

    @abstractmethod
    def select_data(self, context_data_file_path: Path):
        """
        dummy
        """
        pass

    def one_hot_encode(self, y):
        """
        Transforms the labels from integers to one hot vectors
        :param y: array with labels to encode
        :return: array of one hot encoded labels
        """
        enc = OneHotEncoder(categories='auto')
        return enc.fit_transform(y.reshape(-1, 1)).toarray()

    def train_valid_test_split(self, X, y, splits: Optional[tuple] = None) -> typing.Tuple:
        """
        Splits data into training, testing and validation set using random sampling
        :param X: input data array of shape (event, sample, feature)
        :param y: output data array of shape (event)
        :param splits: tuple specifying splitting fractions (training, validation, test)
        :return: train, valid, test: Tuple with data of type named tuple
        """
        if splits is None:
            splits = (0.7, 0.2, 0.1)

        if splits[0] == 1:
            raise ValueError('Training set fraction cannot be 1')

        idx = np.arange(len(X))
        X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = \
            train_test_split(X, y, idx, train_size=splits[0])
        X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
            train_test_split(X_tmp, y_tmp, idx_tmp, train_size=splits[1] / (1 - (splits[0])))

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X_test, y_test, idx_test)

        return train, valid, test

def load_dataset(creator: DatasetCreator, work_dir: Path) -> typing.Tuple:
    """
    some doc
    """
    X, y = creator.select_data(context_data_file_path=work_dir / "context.hdf")
    train, valid, test = creator.train_valid_test_split(X=X, y=y)

    return train, valid, test

#if __name__ == '__main__':
#   train, valid, test = load_dataset(SimpleSelect(),
#                                      Path('/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/context.hdf'))