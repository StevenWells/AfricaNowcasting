#!/usr/bin/sh
#!/bin/bash
date
source /etc/profile.d/conda.sh
conda activate py37

OUTDIR_RAW='/mnt/scratch/stewells/MTG_LI0691/raw'
OUTDIR_TIDY='/mnt/scratch/stewells/MTG_LI0691/tidy'
ARCHIVE_DIR=/mnt/prj/swift/MTG_flash_count/`date -u +%Y"/"%m`
GRA_SCRATCHDIR='/mnt/scratch/stewells/MTG_LI0691/flash_count_NRT'

# prepare areas
mkdir -p $OUTDIR_TIDY
mkdir -p $OUTDIR_RAW
mkdir -p $ARCHIVE_DIR
mkdir -p $GRA_SCRATCHDIR

collection='EO:EUM:DAT:0691'
/home/stewells/AfricaNowcasting/eumdac/eumdac download -c $collection -s `date -u -d "-1hour" "+%Y-%m-%dT%H:%M"` -y -o $OUTDIR_RAW

zfiles=`ls $OUTDIR_RAW/*.zip`

for zfile in $zfiles
do
  unzip -n $zfile -d $OUTDIR_RAW 
  rm $zfile 
  rm $OUTDIR_RAW/*TRAIL*.nc
done
# create gridded file with test_NRT_data.f90
mv $OUTDIR_RAW/W_XX*LFL*.nc $OUTDIR_TIDY
# tidy up
rm $OUTDIR_RAW/W_XX*.jpg
rm -rf $OUTDIR_RAW/quicklooks

# create gridded output
#
# vn1 makes full use of all data within 15 minute period
#
/home/stewells/AfricaNowcasting/rt_code/grid_NRT_LI_afnow
chmod a+r /mnt/scratch/stewells/MTG_LI0691/flash_count_NRT/*

# Create lightning file for portal
python /home/stewells/AfricaNowcasting/rt_code/portal_lightning_pt_convert.py --sourceDir $GRA_SCRATCHDIR 