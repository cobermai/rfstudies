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
```bash
# for debian based systems
apt update
sudo xargs apt install <requirements.system

```
### Installation in CERN environment
for an installation in the CERN environment follow the [Acc-Py](https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py) guide.

### Installation local
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
transform("source/files/dir", "destination/dir")
```

```angular2html ( cleanpy .; tree -A -I "__init__.py|venv|__pycache__|log_files")
.
├── log_config.yml              < config file for logging
├── mypy.ini                    < MyPy config file
├── README.md                   < explanation file
├── requirements.txt            < list of required packages
├── setup_logging.py            < setup the logging mechanism
├── setup.py                    < Setup file for using the project with "pip install mlframework"
├── src                         <<< source directory
│   ├── transformation.py       < tranforms unknown data files to commonly known ones (ex.: .tdms -> .hdf)
│   └── utils                   < utilities
│       ├── transf_tools        < utilities used in the transformation
│       │   ├── gather.py       < combines data without copying it
│       │   └── tdms_read.py    < reads tmds files and transformes them in to hdf files
│       └── system              
│── tests                       <<< tests for main code in the source directory
│   ├── integration             < tests from start to beginning (also called end to end test), NOT IMPLEMENTED YET
│   ├── unit                    < tests every function/class (also called atomic test)
│   └── utils                   < utilyties for the testing suite (ex. creating test files)
```
