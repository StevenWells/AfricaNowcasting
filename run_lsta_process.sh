#!/bin/bash

date
source /etc/profile.d/conda.sh
conda activate py37

cd /home/stewells/AfricaNowcasting/rt_code


python calc_lsta_full_ssa_rt.py

date
