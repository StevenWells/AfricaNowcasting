#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py311_netncc

python /home/stewells/AfricaNowcasting/rt_code/PanAfrica_NetNCC_leadtimes_1to6_0p05deg.py --mode realtime


