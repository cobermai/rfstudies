import sys
from pathlib import Path
import os
api_dir = str(Path(os.path.split(os.path.split(os.getcwd())[0])[0]))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)
import pandas as pd
import numpy as np
from htc_runner import HTCondorRunner
from src.xbox2_specific.datasets.XBOX2_event_all_bd_20ms import XBOX2EventAllBD20msSelect
from src.xbox2_specific.datasets.XBOX2_event_primo_bd_20ms import XBOX2EventPrimoBD20msSelect
from src.xbox2_specific.datasets.XBOX2_event_followup_bd_20ms import XBOX2EventFollowupBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_all_bd_20ms import XBOX2TrendAllBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_primo_bd_20ms import XBOX2TrendPrimoBD20msSelect
from src.xbox2_specific.datasets.XBOX2_trend_followup_bd_20ms import XBOX2TrendFollowupBD20msSelect


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
    run_pairs = [[1, 7], [2], [4, 9], [5], [6, 8]]
    train_runs = []
    val_runs = []
    test_runs = []
    for i, pair in enumerate(run_pairs):
        train_runs.append(sum([r for r in run_pairs if r != pair], []))
        val_runs.append(pair)
        test_runs.append([3])

    # Set parameter grid
    param_grid = {
        'train_runs': train_runs,
        'val_runs': val_runs,
        'test_runs': test_runs,
        'model': ["fcn", "fcn_2dropout", "resnet", "cnn", "inception"],
        'datasets': [XBOX2EventAllBD20msSelect(),
                     XBOX2EventPrimoBD20msSelect(),
                     XBOX2EventFollowupBD20msSelect(),
                     XBOX2TrendAllBD20msSelect(),
                     XBOX2TrendPrimoBD20msSelect(),
                     XBOX2TrendFollowupBD20msSelect()]
    }
    vary_values = list(map(param_grid.get, param_grid.keys()))
    meshgrid = np.array(np.meshgrid(*vary_values)).T.reshape(-1, len(param_grid.keys()))

    df_meshgrid = pd.DataFrame(meshgrid, columns=param_grid.keys())
    for index, row in df_meshgrid.iterrows():
        hyperparameters["model"] = row["model"]
        if "trend" in str(row["datasets"]):
            scale_by_run = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            scale_by_run = None
        HTCondorRunner.run(hyperparameters=str(hyperparameters),
                           dataset=row["datasets"],
                           manual_split=str((row["train_runs"],row["val_runs"],row["test_runs"])),
                           manual_scale=str(scale_by_run))

if __name__ == '__main__':
    sensitivity()
