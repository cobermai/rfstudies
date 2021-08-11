"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import argparse
from datetime import datetime
import json
import pandas as pd
from src.transformation import transform
from src.handler import XBox2ContextDataCreator
from src.model.classifier import Classifier
from src.utils.handler_tools import dataset_creator
from src.utils import hdf_tools


def parse_input_arguments():
    """
    Parses input arguments
    :return: ArgumentParser object which stores input arguments, e.g. path to input data
    """
    parser = argparse.ArgumentParser(description='input parameter')
    parser.add_argument('--file_path', required=False, type=Path,
                        help='path of main.py file', default=Path().absolute())
    parser.add_argument('--data_path', required=False, type=Path,
                        help='path of to data',
                        default=Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/").expanduser())
    parser.add_argument('--dataset_name', required=False, type=str,
                        help='path of to data', default="simple_select")
    parser.add_argument('--transform_to_hdf5', required=False, type=bool,
                        help="retrainform original dataset to hdf5", default=False)
    parser.add_argument('--calculate_features', required=False, type=bool,
                        help="recalculate features", default=False)
    return parser.parse_args()


def transformation(work_dir: Path):
    """TRANSFORMATION"""
    src_dir = Path("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/").expanduser()
    transform(tdms_dir=src_dir,
              hdf_dir=work_dir)

    gathered_trend_data = work_dir / "TrendDataExtLinks.hdf"
    combined_trend_data_path = work_dir / "combined.hdf"

    hdf_tools.merge(source_file_path=gathered_trend_data,
                    dest_file_path=combined_trend_data_path)
    hdf_tools.convert_iso8601_to_datetime(file_path=combined_trend_data_path)
    hdf_tools.sort_by(file_path=combined_trend_data_path, sort_by_name="Timestamp")

def feature_handling(work_dir: Path):
    """DATA HANDLING"""
    gathered_event_data_path = work_dir / "EventDataExtLinks.hdf"
    context_data_file_path = work_dir / "context.hdf"
    combined_trend_data_path = work_dir / "combined.hdf"

    creator = XBox2ContextDataCreator(ed_file_path=gathered_event_data_path,
                                      td_file_path=combined_trend_data_path,
                                      dest_file_path=context_data_file_path)
    creator.manage_features()

def modeling(train_set, valid_set, test_set, work_dir: Path):
    """MODELING"""
    hp_file = open(work_dir / "model/default_hyperparameters.json", 'r')
    hp_dict = json.load(hp_file)
    output_path = work_dir / "output"
    clf = Classifier(output_path, **hp_dict)

    fit_classifier = True
    if fit_classifier:
        clf.fit_classifier(train_set, valid_set)
    clf.model.load_weights(output_path / 'best_model.hdf5')
    results = clf.model.evaluate(x=test_set.X, y=test_set.y, return_dict=True)
    pd.DataFrame.from_dict(results, orient='index').T.to_csv(output_path / "results.csv")

if __name__ == '__main__':
    args = parse_input_arguments()

    if args.transform_to_hdf5:
        transformation(work_dir=args.data_path)

    if args.calculate_features:
        feature_handling(work_dir=args.data_path)

    train, valid, test = dataset_creator.load_dataset(args.data_path, args.dataset_name)
    modeling(train_set=train, valid_set=valid, test_set=test, work_dir=args.file_path)



