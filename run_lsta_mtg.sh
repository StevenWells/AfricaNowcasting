#!/bin/bash

date
source /etc/profile.d/conda.sh
conda activate py37

cd /home/stewells/AfricaNowcasting/rt_code

python lsta_mtg_daymean.py


date
