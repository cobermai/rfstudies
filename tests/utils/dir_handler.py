"""easy handling of directories of data generated for testing"""
from pathlib import Path
from shutil import rmtree
from typing import Union

def remkdir(file_path: Union[str, Path]) -> Path:
    """deletes the old file_path and makes a new one
    :param path: the path of the dir to remake
    :return: remade path"""
    path = Path(file_path).absolute()
    if path.exists():
        rmtree(path)
    path.mkdir(parents=False, exist_ok=False)
    return path
