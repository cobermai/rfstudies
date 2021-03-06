"""example code how to select from context data and prepare data for machine learning. """
import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tensorflow import keras

from src.handler import XBox2ContextDataCreator
from src.model.classifier import Classifier
from src.model.explainer import explain_samples
from src.model.sample_explainers.gradient_shap import ShapGradientExplainer
from src.transformation import transform
from src.utils import hdf_tools
from src.dataset_creator import data_array_to_numpy, load_dataset
from src.utils.result_logging_tools import log_to_csv
from src.xbox2_specific import datasets


def parse_input_arguments(args):
    """
    Parses input arguments
    :param args: List of strings to parse. The default is taken from sys.argv.
    :return: ArgumentParser object which stores input arguments, e.g. path to input data
    """
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('--file_path',
                        required=False,
                        type=Path,
                        help='path of xbox2_main.py file',
                        default=Path().absolute())
    parser.add_argument('--raw_data_path',
                        required=False,
                        type=Path,
                        help='path of data',
                        default=Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2"))
    parser.add_argument('--data_path',
                        required=False,
                        type=Path,
                        help='path of data',
                        default=Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf_new2/"))
    parser.add_argument('--output_path',
                        required=False,
                        type=Path,
                        help='path of data',
                        default=Path().absolute() / "src/output" / datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
    parser.add_argument('--dataset_name',
                        required=False,
                        type=str,
                        help='name of data set',
                        default="XBOX2TrendAllBD20msSelect")
    parser.add_argument('--transform_to_hdf5',
                        required=False,
                        type=bool,
                        help="retransform from original files to hdf5 (True/False)",
                        default=False)
    parser.add_argument('--calculate_features',
                        required=False,
                        type=bool,
                        help="recalculate features (True/False)",
                        default=False)
    parser.add_argument('--explain_predictions',
                        required=False,
                        type=bool,
                        help="explain predictions(True/False)",
                        default=False)
    parser.add_argument('--hyperparameter_path',
                        required=False,
                        type=str,
                        help='path to hyperparameters',
                        default="src/model/default_hyperparameters.json")
    parser.add_argument('--manual_split',
                        required=False,
                        type=ast.literal_eval,
                        help='tuple of manual split index',
                        default=([1, 7, 2, 4, 9, 5], [6, 8], [3]))
    parser.add_argument('--manual_scale',
                        required=False,
                        type=ast.literal_eval,
                        help='list of manual scale index',
                        default=None)
    return parser.parse_args(args)


def transformation(raw_data_dir: Path, work_dir: Path):
    """
    Function for converting tdms files into hdf files, creating external link files and combined trend data.
    :param work_dir: path where EventDataExtLinks.hdf, combined.hdf is found. Context data file is put here.
    """
    transform(tdms_dir=raw_data_dir, hdf_dir=work_dir)

    # Path to gathered trend data
    gathered_trend_data = work_dir / "TrendDataExtLinks.hdf"
    # Path to put combined trend data
    combined_trend_data_path = work_dir / "combined.hdf"
    # Merge all trend data into one hdf file
    hdf_tools.merge(source_file_path=gathered_trend_data,
                    dest_file_path=combined_trend_data_path)
    hdf_tools.convert_iso8601_to_datetime(file_path=combined_trend_data_path)
    hdf_tools.sort_by(file_path=combined_trend_data_path,
                      sort_by_name="Timestamp")


def feature_handling(work_dir: Path):
    """
    Function for creating context data file from event data and trend data
    :param work_dir: path where EventDataExtLinks.hdf, combined.hdf is found. Context data file is put here.
    """
    gathered_event_data_path = work_dir / "EventDataExtLinks.hdf"
    context_data_file_path = work_dir / "context.hdf"
    combined_trend_data_path = work_dir / "combined.hdf"

    creator = XBox2ContextDataCreator(ed_file_path=gathered_event_data_path,
                                      td_file_path=combined_trend_data_path,
                                      dest_file_path=context_data_file_path)
    creator.manage_features()


def modeling(train_set,
             valid_set,
             test_set,
             hp_path: str,
             output_dir: Path,
             fit_classifier: bool = True):
    """
    Function for creating and potentially training a classifier model
    :param train_set: training set as namedtuple(X, y, idx)
    :param valid_set: validation set as namedtuple(X, y, idx)
    :param test_set: validation set as namedtuple(X, y, idx)
    :param hp_path: path to hyperparameter .json file
    :param output_dir: path to output directory
    :param fit_classifier: bool stating whether the model should be trained
    :return: instance of classifier model
    """
    hp_file = open(Path().absolute() / hp_path, 'r')
    hp_dict = json.load(hp_file)

    clf = Classifier(input_shape=train_set.X.shape,
                     output_directory=output_dir,
                     **hp_dict)
    if fit_classifier:
        clf.fit_classifier(train_set, valid_set)
    clf.model.load_weights(output_dir / 'best_model.hdf5')

    results = clf.model.evaluate(x=test_set.X, y=test_set.y, return_dict=True, batch_size=len(test_set.X))

    log_to_csv(logging_path=output_dir / "results.csv", **hp_dict)
    log_to_csv(logging_path=output_dir / "results.csv", **results)
    return clf


if __name__ == '__main__':
    args_in = parse_input_arguments(args=sys.argv[1:])

    def print_header(message: str):
        """
        Function printing what the main function is doing
        :param message: message corresponding to the section of the code executed
        :return: nothing, just print on the stdout
        """
        print("\n",'*'*len(message),'\n',message,'\n','*'*len(message),"\n")

    if args_in.transform_to_hdf5:
        print_header('* TRANSFORMING RAW DATA *')
        transformation(raw_data_dir=args_in.raw_data_path, work_dir=args_in.data_path)

    if args_in.calculate_features:
        print_header('* CALCULATING FEATURES *')
        feature_handling(work_dir=args_in.data_path)

    print_header(f"* TRAINING ON {args_in.dataset_name} *")

    if args_in.dataset_name == "XBOX2EventAllBD20msSelect":
        dataset_creator = datasets.XBOX2EventAllBD20msSelect()
    elif args_in.dataset_name == "XBOX2EventPrimoBD20msSelect":
        dataset_creator = datasets.XBOX2EventPrimoBD20msSelect()
    elif args_in.dataset_name == "XBOX2EventFollowupBD20msSelect":
        dataset_creator = datasets.XBOX2EventFollowupBD20msSelect()
    elif args_in.dataset_name == "XBOX2TrendAllBD20msSelect":
        dataset_creator = datasets.XBOX2TrendAllBD20msSelect()
    elif args_in.dataset_name == "XBOX2TrendPrimoBD20msSelect":
        dataset_creator = datasets.XBOX2TrendPrimoBD20msSelect()
    elif args_in.dataset_name == "XBOX2TrendFollowupBD20msSelect":
        dataset_creator = datasets.XBOX2TrendFollowupBD20msSelect()
    else:
        raise AssertionError("Dataset name does not exist")

    train, valid, test = load_dataset(creator=dataset_creator,
                                      data_path=args_in.data_path,
                                      manual_split=args_in.manual_split,
                                      manual_scale=args_in.manual_scale)
    train_numpy, valid_numpy, test_numpy = data_array_to_numpy(train, valid, test)

    clf = modeling(train_set=train_numpy,
                   valid_set=valid_numpy,
                   test_set=test_numpy,
                   hp_path=args_in.hyperparameter_path,
                   output_dir=args_in.output_path)

    log_to_csv(logging_path=args_in.output_path / "results.csv",
               dataset_name=args_in.dataset_name,
               manual_split=str(args_in.manual_split),
               manual_scale=str(args_in.manual_scale),
               output_path=str(args_in.output_path))

    if args_in.explain_predictions:
        explanation = explain_samples(explainer=ShapGradientExplainer(),
                                      model=clf.model,
                                      X_reference=train_numpy.X,
                                      X_to_explain=test_numpy.X[:1, :, :])
        pd.DataFrame(explanation[0][0]).to_csv(args_in.output_path /
                                               "explanations.csv")

    keras.backend.clear_session()
