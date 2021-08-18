#!/bin/bash

folder=/afs/cern.ch/user/c/cobermai/Desktop/afs_work/PycharmProjects/clicmlframework/src/

for entry in "${folder}settings"/*.json
do
  settings=`expr "${entry}" | cut -f13 -d"/" | cut -f1 -d"."`
  echo $settings
  condor_submit htc_submit_temp.sub settings=$settings sanalysis='True'
done
condor_q
