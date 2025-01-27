#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate nflics_tensor
python /home/stewells/AfricaNowcasting/rt_code/ZA_jan_feb_allhr_using_1hr_real_time.py --mode realtime 
date