import os
import sys
from pathlib import Path

api_dir = str(Path(os.path.split(os.path.split(os.getcwd())[0])[0]))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)
import numpy as np
import pandas as pd
from htc_runner import HTCondorRunner

hyperparameters = {
    "classifier_name": "fcn",
    "num_classes": 2,
    "monitor": "loss",
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "epochs": 500,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 50,
    "min_lr": 0.0001
}

def sensitivity():
    # For leave one out crossval with runs
    run_groups = [[1, 7], [2], [4, 9], [5], [6, 8]]
    train_runs = []
    val_runs = []
    test_runs = []
    for i, group in enumerate(run_groups):
        train_runs.append(sum([r for r in run_groups if r != group], []))
        val_runs.append(group)
        test_runs.append([3])

    # Set parameter grid
    param_grid = {
        "data_split": [str((t, v, te)) for t, v, te in zip(train_runs, val_runs, test_runs)],
        'model': ["fcn", "fcn_2dropout", "resnet", "time_cnn", "inception"],
        'datasets': ["XBOX2EventAllBD20msSelect",
                     "XBOX2EventPrimoBD20msSelect",
                     "XBOX2EventFollowupBD20msSelect",
                     "XBOX2TrendAllBD20msSelect",
                     "XBOX2TrendPrimoBD20msSelect",
                     "XBOX2TrendFollowupBD20msSelect"]
    }

    # Set parameter grid
    param_grid = {
        "data_split": [str((t, v, te)) for t, v, te in zip(train_runs, val_runs, test_runs)],
        'model': ["fcn"],
        'datasets': ["XBOX2EventAllBD20msSelect"]
    }

    vary_values = list(map(param_grid.get, param_grid.keys()))
    meshgrid = np.array(np.meshgrid(*vary_values)).T.reshape(-1, len(param_grid.keys()))
    df_meshgrid = pd.DataFrame(meshgrid, columns=param_grid.keys())
    for index, row in df_meshgrid.iterrows():
        hyperparameters["classifier_name"] = row["model"]

        if "Trend" in str(row["datasets"]):
            scale_by_run = list(np.arange(1, 10))
        else:
            scale_by_run = None

        HTCondorRunner.run(hyperparameters=hyperparameters,
                           dataset_name=row["datasets"],
                           manual_split=str(row["data_split"]),
                           manual_scale=str(scale_by_run))


if __name__ == '__main__':
    sensitivity()
