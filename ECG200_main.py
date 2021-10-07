"""example code how to select from context data and prepare data for machine learning. """
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import pandas as pd
import matplotlib as mpl
from src.handler import XBox2ContextDataCreator
from src.model.classifier import Classifier
from src.transformation import transform
from src.utils.dataset_creator import load_dataset
from src.utils import hdf_tools
from src.datasets.ECG200 import ECG200
from src.model.explainer import explain_samples
from src.model.sample_explainers.gradient_shap import ShapGradientExplainer
from src.model.concept_explainers.concept_shap import ShapConceptExplainer

def parse_input_arguments(args):
    """
    Parses input arguments
    :return: ArgumentParser object which stores input arguments, e.g. path to input data
    """
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('--file_path', required=False, type=Path,
                        help='path of xbox2_main.py file', default=Path().absolute())
    parser.add_argument('--data_path', required=False, type=Path,
                        help='path of data', default=Path().absolute() / "src/datasets/ECG200")
    parser.add_argument('--output_path', required=False, type=Path, help='path of data',
                        default=Path().absolute() / "src/output" / datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
    parser.add_argument('--dataset_name', required=False, type=str,
                        help='name of data set', default="ECG200")
    parser.add_argument('--param_name', required=False, type=str,
                        help='name of hyperparameter file', default="default_hyperparameters.json")
    parser.add_argument('--transform_to_hdf5', required=False, type=bool,
                        help="retransform from original files to hdf5 (True/False)p", default=False)
    parser.add_argument('--calculate_features', required=False, type=bool,
                        help="recalculate features (True/False)", default=False)
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


def modeling(train_set, valid_set, test_set, param_dir: Path, output_dir: Path, fit_classifier: bool = True):
    """MODELING"""
    hp_file = open(param_dir, 'r')
    hp_dict = json.load(hp_file)

    clf = Classifier(input_shape=train_set.X.shape, output_directory=output_dir, **hp_dict)
    if fit_classifier:
        clf.fit_classifier(train_set, valid_set)
    clf.model.load_weights(output_dir / 'best_model.hdf5')
    results = clf.model.evaluate(x=test_set.X, y=test_set.y)
    #pd.DataFrame.from_dict(results, orient='index').T.to_csv(output_dir / "results.csv")
    return clf

def explanation(classifier, train_set, valid_set, test_set, output_dir: Path):

    def plot_importance(X_to_explain, y_pred, explanation):
        cmap = mpl.colors.LinearSegmentedColormap.from_list('shap', [mpl.cm.cool(0), (1, 1, 1, 1), mpl.cm.cool(256)],
                                                            N=256)

        fig, ax = mpl.pyplot.subplots(figsize=(7, 5))
        ax.plot(X_to_explain[(y_pred.argmax(axis=1) == 0), :, :].mean(axis=0), linewidth=3, c="b")
        ax.plot(X_to_explain[(y_pred.argmax(axis=1) == 1), :, :].mean(axis=0), linewidth=3, c="r")

        extent = [0, len(X_to_explain[0]), ax.get_ylim()[0], ax.get_ylim()[1]]
        im1 = ax.imshow(explanation[0].mean(axis=0).T, cmap=cmap, aspect="auto", alpha=0.8, extent=extent)
        cbar1 = fig.colorbar(im1, ax=ax)
        cbar1.set_label("SHAP values (relative)")

        ax.set_xlabel("samples")
        ax.set_title("Explanations of Correct Predictions")
        ax.legend(["mean normal", "mean ischemia"])
        fig.savefig(output_dir / 'explanation.png')

    y_pred = classifier.model.predict(x=test_set.X)
    is_correct_pred = (y_pred.argmax(axis=1) == test_set.y.argmax(axis=1))
    ex_pred = explain_samples(explainer=ShapGradientExplainer(),
                              model=classifier.model,
                              X_reference=train_set.X,
                              X_to_explain=test_set.X[is_correct_pred, :, :])

    plot_importance(X_to_explain=test_set.X, y_pred=y_pred, explanation=ex_pred)

if __name__ == '__main__':
    args_in = parse_input_arguments(args=sys.argv[1:])

    if args_in.transform_to_hdf5:
        transformation(work_dir=args_in.data_path)

    if args_in.calculate_features:
        feature_handling(work_dir=args_in.data_path)

    train, valid, test = load_dataset(creator=ECG200(),
                                      data_path=args_in.data_path)
    clf = modeling(train_set=train, valid_set=valid, test_set=test,
                   param_dir=args_in.file_path / "src/model" / args_in.param_name, output_dir=args_in.output_path)

    y_pred = clf.model.predict(x=test.X)

    #explanation(classifier=clf, train_set=train, valid_set=valid, test_set=test, output_dir=args_in.output_path)

    ex_pred = ShapConceptExplainer()
    ex_pred.explain(model=clf.model, train=train, valid=valid, test=test,output_dir=args_in.output_path)

