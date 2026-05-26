#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py39_ecmwf
python /home/stewells/AfricaNowcasting/rt_code/afnow_ecmwfprods.py 
