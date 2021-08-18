#!/bin/bash
echo "ClusterId values is:"
echo $1

echo "ProcId values is:"
echo $2

echo "Output values is:"
echo $3

echo "Sensibility analysis:"
echo $4

echo "activate environment"

echo "start script"
filepath=/afs/cern.ch/user/c/cobermai/Desktop/afs_work/PycharmProjects/clicmlframework

python -m virtualenv myvenv
pip install -r ${filepath}/requirements.txt

python "${filepath}/xbox2_main.py" --file_path=$filepath


