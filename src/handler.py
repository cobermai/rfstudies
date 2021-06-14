"""future data handler. UNFINISHED"""
import os.path
import logging
from datetime import datetime
import typing
import math
import itertools
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from src.utils.transf_tools.gather import hdf_path_combine
from setup_logging import setup_logging
from utils.handler_tools.union import union, clean_by_row, sort_by
from src.utils.handler_tools.context_data_creator import ContextDataCreator
from src.utils.handler_tools.features import get_features

setup_logging()
LOG = logging.getLogger("test_handler")



if __name__ == "__main__":
    source_file_path = Path("~/output_files/TrendDataExtLinks.hdf").expanduser()
    dest_file_path = Path("~/output_files/combined_td.hdf").expanduser()

    h5py.File(dest_file_path, mode="w").close()

    union(source_file_path=src_file_path, dest_file_path=dest_file_path)
    clean_by_row(file_path=dest_file_path)
    sort_by(file_path=dest_file_path, sort_by_name="Timestamp")



