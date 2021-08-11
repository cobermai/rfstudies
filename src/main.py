"""example code how to select from context data and prepare data for machine learning. """
import typing
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import json
from src.model.classifier import Classifier
from src.utils import dataset_creator


def parse_input_arguments():
    """
    Parses input arguments
    :return: ArgumentParser object which stores input arguments, e.g. path to input data
    """
    parser = argparse.ArgumentParser(description='input parameter')
    parser.add_argument('--file_path', required=False, type=str,
                        help='path of main.py file', default=Path().absolute())
    parser.add_argument('--data_path', required=False, type=str,
                        help='path of to data',
                        default=Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/").expanduser())
    parser.add_argument('--dataset_name', required=False, type=str,
                        help='path of to data', default="trend_bd_next_pulse")
    parser.add_argument('--output_folder', required=False, type=str,
                        help="name of output directory", default=Path("output/"
                                                                      + datetime.now().strftime("%Y-%m-%dT%H_%M_%S")))
    parser.add_argument('--fit_classifier', required=False, type=bool,
                        help="retrain classifier", default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_input_arguments()

    train, valid, test = dataset_creator.load_dataset(args.data_path, args.dataset_name)

    hp_file = open(args.file_path / "model/default_hyperparameters.json", 'r')
    hp_dict = json.load(hp_file)
    output_path = args.file_path / args.output_folder
    clf = Classifier(output_path, **hp_dict)

    if args.fit_classifier:
        clf.fit_classifier(train, valid)

    clf.model.load_weights(output_path / 'best_model.hdf5')
    results = clf.model.evaluate(x=test.X, y=test.y, return_dict=True)
    pd.DataFrame.from_dict(results, orient='index').T.to_csv(output_path / "results.csv")
