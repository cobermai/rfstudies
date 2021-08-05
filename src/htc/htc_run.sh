#!/bin/bash
# condor_submit htc_submit.sub settings=test_86
# requirements          = regexp("V100", TARGET.CUDADeviceName)
echo "ClusterId values is:"
echo $1

echo "ProcId values is:"
echo $2

echo "Output values is:"
echo $3

echo "Sensibility analysis:"
echo $4

echo "activate environment"
# TODO: from clicml_old, how should this be done now
python -m virtualenv myvenv
source /afs/cern.ch/user/c/cobermai/Desktop/afs_work/miniconda3/bin/activate base

echo "start script"
filepath=/afs/cern.ch/user/c/cobermai/Desktop/afs_work/PycharmProjects/clicmlframework/src

python "${filepath}/main.py" --output=$3 --ProcId=$2 --filepath=$filepath


