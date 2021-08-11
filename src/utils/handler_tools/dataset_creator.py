import typing
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from src.xbox2_speciffic.datasets import simple_select


def one_hot_encode(y):
    """
    Transforms the labels from integers to one hot vectors
    :param y: array with labels to encode
    :return: array of one hot encoded labels
    """
    enc = OneHotEncoder(categories='auto')
    return enc.fit_transform(y.reshape(-1, 1)).toarray()


def load_dataset(data_path: Path, dataset_name: str) -> typing.Tuple:
    """
    Loads the specified data set, does one hot encoding on labels and splits data into train, valid and test set
    :param data_path: Path to input data
    :param dataset_name: Name of the data set
    :return: Tuple of named tuples containing training, validation and test set
    """
    if dataset_name == "simple_select":
        X, y = simple_select.select_data(context_data_file_path=data_path / "context.hdf")
        X_scaled = simple_select.scale_data(X)
        y_hot = one_hot_encode(y)
        train, valid, test = simple_select.train_valid_test_split(X=X_scaled, y=y_hot, splits=(0.7, 0.2, 0.1))
    else:
        raise AssertionError

    return train, valid, test
