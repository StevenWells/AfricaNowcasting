#!/bin/sh
#set -ex
date
source /etc/profile.d/conda.sh
conda activate py37

#currently run with crontab line
# */15 * * * * /users/global/cmt/ASCAT/HSAF/H122/hsaf_h122_data_copy>/dev/null 2>&1

YEAR=`date -u +"%Y"`
MONTH=`date -u +"%m"`
DAY=`date -u +"%d"`
#DAY=`date -d yesterday -u +"%d"`
#DAY=29
#DAY=21
OUTDIR=/mnt/prj/swift/ASCAT_H122/H122_vn2J_daily
#
#mkdir -p $OUTDIR
#f9nc /users/global/cmt/ASCAT/HSAF/H122/grid_NRT_vn2J

#
# copy h122 files across from hsaf
# currently being run as a cron job on wlsc-lin11
#

version='h122'

# download the files
echo "Downloading files..."
wget -N -nv "ftp://ftphsaf.meteoam.it/$version/h122_cur_mon_nc/*$YEAR$MONTH$DAY*" -P "/mnt/scratch/stewells/$version/"


# make output folder
mkdir -p $OUTDIR/$YEAR/$MONTH

# process the files
echo "processing files..."
/home/stewells/AfricaNowcasting/rt_code/grid_NRT_vn2J_afnow $YEAR$MONTH$DAY

chmod a+r $OUTDIR/*
#ls -lt $OUTDIR|head

# create the geotiff
echo "Creating geotiff..."
#python /home/stewells/AfricaNowcasting/rt_code/portal_soilmoisture_convert.py --mode realtime --domain SSA_6k_archive --outDir /home/stewells/AfricaNowcasting/testout/
#python /home/stewells/AfricaNowcasting/rt_code/portal_soilmoisture_convert.py --mode realtime --domain SSA_6k_archive --outDir /mnt/HYDROLOGY_stewells/geotiff/ssa_soil_moisture_anomaly/

python /home/stewells/AfricaNowcasting/rt_code/portal_soilmoisture_convert.py --mode historical --startDate $YEAR$MONTH$DAY --endDate $YEAR$MONTH$DAY --domain SSA_6k_archive --outDir /mnt/HYDROLOGY_stewells/geotiff/ssa_soil_moisture_anomaly/




