#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py37

python /home/stewells/AfricaNowcasting/rt_code/portal_tidy.py
