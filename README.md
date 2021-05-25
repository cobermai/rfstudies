# Machine Learning Framework ( = mlframework)
## INTRODUCTION

## Getting Started
This is a framework for machine learning. It consists of three steps:
1) **Transformation**: transforming data into the fileformat hdf
2) **Handling**: 
3) **Exploration**:
4) **Training**:
### Requirements
In order to follow the installation guide it is required to install
```pip (?)``` and ```python >=3.8 <4 ``` and ```git```

### Installation
```bash
git clone https://gitlab.cern.ch/machine-protection-ml/mlframework.git # Clone the gitlab project
git checkout mlframework #  and checkout the branch mlframework
cd mlframework
pip3 install virtualenv
virtualenv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

### Usage
```python
# Transformation from .tdms files source/files/path/ to destination/path/
from src.transformation import transform
transform("source/files/path", "destination/path")
```

```angular2html (tree -A -I "archive|venv|__pycache__|log_files)
.
├── docu
├── mypy.ini
├── requirements.txt
├── setup.py
├── src
│   ├── handler.py
│   ├── __init__.py
│   ├── README.md
│   ├── transformation.py
│   └── utils
│       ├── hdf_utils
│       │   ├── gather.py
│       │   ├── __init__.py
│       │   └── tdms_read.py
│       ├── __init__.py
│       └── system
│           ├── __init__.py
│           ├── logger.py
│           ├── progress.py
│           └── work_load.py
└── tests
    ├── __init__.py
    ├── integration
    │   └── __init__.py
    ├── unit
    │   ├── data
    │   │   ├── dest.hdf
    │   │   └── source_dir
    │   │       └── test.hdf
    │   ├── test_transformation.py
    │   └── utils
    │       └── hdf_utils
    │           ├── test_gather.py
    │           └── test_tdms_read.py
    └── utils
        ├── data_creator
        │   ├── file_creator_for_testing.py
        │   ├── __init__.py
        │   ├── tdms_file_creator.py
        │   ├── xb2_like_event_data_creator.py
        │   └── xb2_like_trend_data_creator.py
        ├── dir_handler.py
        └── __init__.py


```