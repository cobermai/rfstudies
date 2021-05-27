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

```angular2html (tree -A -I "__init__.py|venv|__pycache__|log_files)
.
├── src                        <<< source directory
│   ├── handler.py               < the data handler UNFINISHED
│   ├── transformation.py        < tranforms unknown data files to commonly known ones (ex.: .tdms -> .hdf)
│   └── utils                    < utilities
│       ├── transf_tools         < utilities used in the transformation
│       │   ├── gather.py        < combines data without copying it
│       │   └── tdms_read.py     < reads tmds files and transformes them in to hdf files
│       └── system               < development utils for logging and displaying progress
│── tests                      <<< tests for main code in the source directory
│   ├── integration              < tests from start to beginning (also called end to end test), NOT IMPLEMENTED YET
│   ├── unit                     < tests every function/class (also called atomic test)
│   └── utils                    < utilyties for the testing suite (ex. creating test files)
├── archive                    <<< OLD CODE that is "stored" here but not part of the main project. 
│   ├── API                      < old implementation of the API and refactored code
│   ├── Data_Plot                < code for plotting xb2 data
│   ├── file_format_presentation < presentation to explain the advantages of the old file format (df+pickle) vs hdf
│   ├── hdf_utils                < an old hdf tool that is now part of gather.py
│   ├── spark                    < DISCARDED tried to use spark for feature calculating
│   ├── tdms_reader              < tdms to pandas dataframe converter
│   └── XBox3_notebooks          < notebooks regarding the xbox3 data set
├── docu                       <<< information regarding some documentation. Not propperly filled yet.
├── mlf.env                      <
├── mypy.ini                     < MyPy config file
├── README.md                    < explanation file
├── requirements.txt             < list of required packages
├── setup.py                     < Setup file for using the project with "pip install mlframework"
```