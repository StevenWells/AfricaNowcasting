#!/bin/bash
set -ex
DOWNLOAD_DIR=/mnt/scratch/stewells/MSG_NRT/in
mkdir -p $DOWNLOAD_DIR
#vn='_d7a'
vn='_d8'
#
# recommended option of publication_after doesn't appear to work
#
#eumdac download -c EO:EUM:DAT:MSG:HRSEVIRI --publication-after `date -d "-35min" "+%Y-%m-%dT%H:%M:%S"` --chain test_NRT.yaml
#
/home/stewells/AfricaNowcasting/eumdac/eumdac download -c EO:EUM:DAT:MSG:HRSEVIRI -s `date -u -d "-23min" "+%Y-%m-%dT%H:%M"` --chain /home/stewells/AfricaNowcasting/rt_code/test_NRT$vn.yaml -o $DOWNLOAD_DIR
#
#  if more than 5 eumdac tailor jobs running, run clean-up
#
#test `/users/global/cmt/msg/eumdac/eumdac tailor list|grep HRSEVIRI|grep -v RUNNING|grep -vc QUEUED` -gt 5 && /users/global/cmt/msg/eumdac/clean_tailor_jobs.sh 
#
fname=`ls -lt /mnt/scratch/stewells/MSG_NRT/in |grep HRSEVIRI | head -1|cut -c55-`
linkname=`ls -lt /mnt/scratch/stewells/MSG_NRT/in | grep HRSEVIRI | head -1 | cut -c56-68`
##
# define output file name to be consistent with files downloaded from Leeds
#
date_string=`echo $fname|cut -c10-17`
hour_string=`echo $fname|cut -c19-22`
nc_outfile='/mnt/scratch/stewells/MSG_NRT/cut/IR_108_BT_'$date_string'_'$hour_string'.nc'
nc_outfile2='/mnt/scratch/stewells/MSG_NRT/cut/VIS_006_rad_'$date_string'_'$hour_string'.nc'
infile='/mnt/scratch/stewells/MSG_NRT/in/'$fname
outfile1='/mnt/scratch/stewells/MSG_NRT/tmp/foo.nc'
outfile2='/mnt/scratch/stewells/MSG_NRT/tmp/foo1.nc'
#
# convert channel 9 radiance in eumdac file to BT in deg C
# using formula from 
# https://www-cdn.eumetsat.int/files/2020-04/pdf_sci_bams0702_msg-calib.pdf
#
# fundamental constants:
# h=6.62607*1e-34
# c=2.997924*1e8
# kb=1.380650*1E-23
#
# SEVIRI ch9 nominal central wavenumber (m^-1) 
# mu=93065.898
# SEVIRI calibration coefficient A
# a=0.9983
# SEVIRI calibration coefficient B (K)
# b=0.627
#

LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncap2 -O -s "ir108_bt=float(1341.29/(log(1.+(0.096006/(channel_9/100000.))))-0.628068-273.15)" $infile $outfile1
#
# remove original channel 9 radiance data from output file
#
LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncks -C -O -x -v channel_1 $outfile1 $outfile2
LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncks -C -O -x -v channel_9 $outfile2 $nc_outfile
rm $outfile1 $outfile2
#
# create separate vis file and rename variable
#
LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncks -C -O -x -v channel_9 $infile $outfile1

if [ -e $nc_outfile2 ]
then
  echo "already processed"
else
LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncrename -v channel_1,vis006_rad $outfile1 $nc_outfile2 
chmod a+r $nc_outfile $nc_outfile2 
fi
#LD_LIBRARY_PATH=/home/stewells/AfricaNowcasting/pkgs/lib /home/stewells/AfricaNowcasting/pkgs/bin/ncrename -v channel_1,vis006_rad $outfile1 $nc_outfile2 

#
#ls -lt $DOWNLOAD_DIR|head
#ls -lt /mnt/scratch/stewells/MSG_NRT/cut/*.nc|head
