#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py37
python /home/stewells/AfricaNowcasting/rt_code/sat_transfer.py historical --fStruct YMD --startDate 202606011200 --endDate 202606011200
date
