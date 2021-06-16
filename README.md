# Machine Learning Framework ( = mlframework)
## INTRODUCTION

## Getting Started
This is a framework for machine learning. It consists of three steps:
1) **Transformation**: 
   * transform source data into a more handy data format (eg. HDF5)
   * make accessible: gather hdf files together if they have the wanted data structure
2) **Exploration**: 
   * handling: reformat, clean and sort
   * calculate features: extract features for machine learning
   * data exploration: unsupervised machine learning for data analysis
3) **Modelling**: 
   * (un)supervised machine learning
   * explain results and validate by application
    

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
├── log_config.yml              < logging configurations
├── mypy.ini                    < configuration file for mypy
├── README.md                   < explanation file
├── requirements.system         < list of required system packages
├── requirements.txt            < list of required python packages and version numbers
├── setup.py                    < Setup file for using the project with "pip install mlframework"
├── sonar-project.properties    < properties for sonar code analyzation tool
├── src                         <<< source directory
│   ├── transformation.py       < tranforms data in special formats into a handy format (ex.: .tdms -> .hdf)
│   └── utils                   < utilities
│       ├── hdf_tools.py        < tools to handle hdf files
│       ├── handler_tools       < combines data without copying it
│       ├── system              < system utils: setup_logging and dev_tools
│       └── transf_tools        < utilities used in the transformation
│           ├── convert.py      < converting tool
│           └── gather.py       < gather data scattered on multiple files
└── tests                       <<< tests for main code in the source directory
    ├── integration             < tests from start to beginning (also called end to end test), NOT IMPLEMENTED YET
    ├── unit                    < tests every function/class (also called atomic test)
    └── utils                   < utilyties for the testing suite (ex. creating test files)

```
