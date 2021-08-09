"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import argparse
import datetime
import json
import pandas as pd
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
                        default=Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfer/Xbox2_hdf/")
                        .expanduser())
    parser.add_argument('--dataset_name', required=False, type=str,
                        help='path of to data', default="trend_bd_next_pulse")
    parser.add_argument('--output_folder', required=False, type=str,
                        help="name of output directory", default="output/" + str(datetime.datetime.now()))
    parser.add_argument('--fit_classifier', required=False, type=bool,
                        help="retrain classifier", default=True)
    return parser.parse_args()


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
