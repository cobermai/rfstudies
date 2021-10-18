"""example code how to select from context data and prepare data for machine learning. """
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import pandas as pd
from src.handler import XBox2ContextDataCreator
from src.model.classifier import Classifier
from src.transformation import transform
from src.utils.dataset_creator import load_dataset
from src.utils import hdf_tools
from src.xbox2_specific.datasets.XBOX2_event_all_bd_20ms import XBOX2EventAllBD20msSelect
from src.xbox2_specific.datasets.XBOX2_event_primo_bd_20ms import XBOX2EventPrimoBD20msSelect
from src.xbox2_specific.datasets.XBOX2_event_followup_bd_20ms import XBOX2EventFollowupBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_all_bd_20ms import XBOX2TrendAllBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_primo_bd_20ms import XBOX2TrendPrimoBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_followup_bd_20ms import XBOX2TrendFollowupBD20msSelect
from src.model.explainer import explain_samples
from src.model.sample_explainers.gradient_shap import ShapGradientExplainer


def parse_input_arguments(args):
    """
    Parses input arguments
    :return: ArgumentParser object which stores input arguments, e.g. path to input data
    """
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('--file_path', required=False, type=Path,
                        help='path of xbox2_main.py file', default=Path().absolute())
    parser.add_argument('--data_path', required=False, type=Path,
                        help='path of data',
                        default=Path(
                            "/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/")
                        )
    parser.add_argument('--output_path', required=False, type=Path, help='path of data',
                        default=Path().absolute() / "src/output" / datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
    parser.add_argument('--dataset_name', required=False, type=str,
                        help='name of data set', default="ECG200")
    parser.add_argument('--transform_to_hdf5', required=False, type=bool,
                        help="retransform from original files to hdf5 (True/False)p", default=False)
    parser.add_argument('--calculate_features', required=False, type=bool,
                        help="recalculate features (True/False)", default=False)
    parser.add_argument('--explain_predictions', required=False, type=bool,
                        help="explain predictions(True/False)", default=False)
    hp_file = open(Path().absolute() / "src/model/default_hyperparameters.json", 'r')
    hp_dict = json.load(hp_file)
    parser.add_argument('--hyperparam', required=False, type=dict, help='dict of hyperparameters', default=hp_dict)
    parser.add_argument('--dataset', required=False, type=object, help='class object to create dataset',
                        default=XBOX2TrendFollowupBD20msSelect())
    parser.add_argument('--manual_split', required=False, type=tuple, help='tuple of manual split index', default=None)
    parser.add_argument('--manual_scale', required=False, type=object, help='list of manual scale index', default=None)

    return parser.parse_args(args)


def transformation(work_dir: Path):
    """TRANSFORMATION"""
    src_dir = Path("/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2")
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


def modeling(train_set, valid_set, test_set, hp_dict: dict, output_dir: Path, fit_classifier: bool = True):
    """MODELING"""
    clf = Classifier(input_shape=train_set.X.shape, output_directory=output_dir, **hp_dict)
    if fit_classifier:
        clf.fit_classifier(train_set, valid_set)
    clf.model.load_weights(output_dir / 'best_model.hdf5')
    results = clf.model.evaluate(x=test_set.X, y=test_set.y, return_dict=True)
    pd.DataFrame.from_dict(results, orient='index').T.to_csv(output_dir / "results.csv")
    return clf


if __name__ == '__main__':
    args_in = parse_input_arguments(args=sys.argv[1:])

    if args_in.transform_to_hdf5:
        transformation(work_dir=args_in.data_path)

    if args_in.calculate_features:
        feature_handling(work_dir=args_in.data_path)

    train_runs = [1, 2, 4, 5, 6, 8, 9]
    valid_runs = [1, 7]
    test_runs = [3]
    train, valid, test = load_dataset(creator=args_in.dataset,
                                      data_path=args_in.data_path,
                                      manual_split=(train_runs, valid_runs, test_runs),
                                      manual_scale=[1, 2, 3, 4, 5, 6, 7, 8, 9]
                                      )
    clf = modeling(train_set=train, valid_set=valid, test_set=test,
                   hp_dict=args_in.hyperparam, output_dir=args_in.output_path)

    if args_in.explain_predictions:
        explanation = explain_samples(explainer=ShapGradientExplainer(), model=clf.model,
                                      X_reference=train.X, X_to_explain=test.X[:1, :, :])
        pd.DataFrame(explanation[0][0]).to_csv(args_in.output_path / "explanations.csv")
