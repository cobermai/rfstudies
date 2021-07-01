"""In this module the merge clean_by_row and sort_by function is applied on the trend data and the context data
creator is called."""
import logging
from pathlib import Path
import argparse
import coloredlogs
from src.utils.handler_tools.context_data_creator import ContextDataDirector

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for xbox2 dataset: extract features for ml")
    parser.add_argument("td", type=Path, help="file path of an hdf file where all TrendData is merged into one dataset"
                                              "for each signal.")
    parser.add_argument("ed", type=Path, help="file path of an hdf file where all EventData groups are gathered (e.g."
                                              "with ExternalLinks).")
    parser.add_argument("dest", type=Path, help="file path of the hdf fiel where the features will be written.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug log messages")
    args = parser.parse_args()
    if args.verbose:
        coloredlogs.install(level="DEBUG")
    else:
        coloredlogs.install(level="INFO")
    logger = logging.getLogger(__name__)

    cd_creator = ContextDataDirector(ed_file_path=args.ed.resolve(),
                                     td_file_path=args.td.resolve(),
                                     dest_file_path=args.dest)
    cd_creator.manage_features()
