#!/bin/bash
# Run the python script to produce ECMWF fields
date
source /etc/profile.d/conda.sh
conda activate py39_ecmwf
python /home/stewells/AfricaNowcasting/rt_code/afnow_ecmwfprods.py 
