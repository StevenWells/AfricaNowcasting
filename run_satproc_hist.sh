#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py37
python /home/stewells/AfricaNowcasting/rt_code/sat_transfer.py historical --startDate 202504201445 --endDate 202504201445
date
