"""easy handling of directories of data generated for testing"""
from pathlib import Path
from shutil import rmtree

def get_clean_data_dir(file_path: str) -> Path:
    # delete old test data
    path = Path(file_path)
    data_dir_path = path.with_name("data_" + path.stem)
    try:
        rmtree(Path(data_dir_path).absolute())
    except FileNotFoundError:
        pass  # we want to delete the folder anyways.
    data_dir_path.mkdir(parents=False, exist_ok=False)
    return data_dir_path

def mkdir_ret(path:Path) -> Path:
    path.mkdir()
    return path