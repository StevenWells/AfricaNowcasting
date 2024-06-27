#!/bin/sh

# lmf_convert.sh
# Author: Steven Wells
# Date created: 30/06/2022
#
# Purpose: To extract and convert LMF.gra data into GeoTiff format
#
# Requires: convert_LWS_netcdf_v2.py in same directory
#           SEVIRILST_WA_geoloc.nc in same directory for georeferencing
#           ../ancils/shapefiles/wca_admbnda_adm1_ocha_3857.shp for masking of sea
#
# Usage:  ./lmf_convert.sh 
# Output: Set of geotiffs for each of the Image times, using same file format as .gra files, but with validity time included.
#         The .gra files are updated throughout the day (there is one for each day). This script picks up and converts ANY .gra file that
#         have been modified in the last 20 minutes
#         Raw .gra from Chris Taylor in /prj/nflics/LMF_output_probabilities. Note these have no geolocation information in them; this script adds them
# Notes; Outputs in EPSG: 3857
#         Steps::
#               1. Split .gra file into six component NetCDF files (one for each leadtime)
#               2. For each file, a) Convert to GeoTIFF
#                                 b) Create a VRT file and use it to hook up the geolocation information to the GeoTIFF
#                                 c) COnstruct the new geolocated GeoTIFF and reporoject onto EPSG:3857
#               3. Transfer all files to the SANS transfer directory
#   

# output directory (NB this directory is used to then move the files across to Lancaster SANS)
#root="/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/"
#root="/home/stewells/AfricaNowcasting/satTest/geotiff/lawis_lmf/"
date
root="/mnt/HYDROLOGY_stewells/geotiff/lawis_lmf/"
wdir="/home/stewells/AfricaNowcasting/tmp/"
# load conda environment
source /etc/profile.d/conda.sh
conda activate py37

# Move to workign directory (this script)
cd /home/stewells/AfricaNowcasting/rt_code/

# set up GDAL parameters
export GDAL_NETCDF_BOTTOMUP=NO
nodata_value=-999


# pick up files modified in the last 20 minutes
# FILES=$(find /prj/nflics/LMF_output_probabilities -regextype posix-egrep -regex ".*prob_vn2b_.*[0-9]+.gra" -type f -mmin -20)
FILES=$(find /mnt/prj/nflics/LMF_output_probabilities -regextype posix-egrep -regex ".*prob_vn2a_.*[0-9]+.gra" -type f -mmin -20)

shopt -s nullglob
for FFILE in $FILES
do
fullfile="$FFILE"


echo `date`
echo $fullfile
#fullfile=$1


# get all necessary grids as netCDFs (set of 6 files created from .gra file for each lead time - they get pulled out into separate netCDF files)
python portal_lmf_convert.py $fullfile

# set up the filenames and direcotries
filename=$(basename -- "$fullfile")
filedir=$(dirname -- "$fillfile")
filename="${filename%.*}"

#for VALIDTIME in 18 21 0 3 12 15
for VALIDTIME in prob*nc
do
suffix='.nc'




ncFile=$VALIDTIME
newfile=${VALIDTIME%"$suffix"}.tif

#tidy up and straggling temporary files prior to calculating
rm $wdir'tmp_lmf'*tif
rm $wdir'tmp_lmf'*.vrt

tmptif=$wdir'tmpfile_lmf.tif'
tmpvrt=$wdir'tmpfile2_lmf.vrt'

# COnvert each file to GeoTIFF (currently no geolocation information - need to get from SEVIRILST_WA_geoloc.nc)
gdal_calc.py --format=GTiff --type=Float32 \
	-A $ncFile \
	--calc="(A!=$nodata_value)*A +(A==$nodata_value)*A" \
	--outfile $tmptif

# Create a VRT file to link with geolocation file
gdal_translate -of VRT $tmptif $tmpvrt


# edit the VRT file to ink with geolocation file
new_section='<Metadata domain="GEOLOCATION">\n<MDI key="X_DATASET">NETCDF:"../ancils/SEVIRILST_WA_geoloc.nc":WA_lon<\/MDI>\n<MDI key="X_BAND">1<\/MDI>\
<MDI key="Y_DATASET">NETCDF:"../ancils/SEVIRILST_WA_geoloc.nc":WA_lat<\/MDI>\n<MDI key="Y_BAND">1<\/MDI>\
<MDI key="PIXEL_OFFSET">0<\/MDI>\n<MDI key="LINE_OFFSET">0<\/MDI>\n<MDI key="PIXEL_STEP">1<\/MDI>\n<MDI key="LINE_STEP">1<\/MDI>\n'
#sed -i '0,@<Metadata domain="IMAGE_STRUCTURE">@{s//'"$new_section"'@g}' tmpfile2.vrt 
#sed -i '0,@<MDI key="INTERLEAVE">BAND<\/MDI>/{s//''@g}' tmpfile2.vrt 

sed -i 's@.*"IMAGE_STRUCTURE".*@'"$new_section"'@g' $tmpvrt
sed -i 's@.*"INTERLEAVE".*@@g' $tmpvrt


# reproject  to EPSG:3857
new_nodata_value='-999'
old_nodata_value='-999'
if [ ! -f $newfile ]; then
     gdalwarp -of GTiff -geoloc -s_srs EPSG:4326 -t_srs EPSG:3857 -dstnodata "$new_nodata_value" $tmpvrt $newfile
fi


# convert the sea into a different value from missing using mask shapefile
gdal_rasterize -i -burn -998  ../ancils/shapefiles/wca_admbnda_adm1_ocha_3857.shp $newfile

# tidy up temporary files
#rm tmp*tif
#rm tmp*.vrt
rm $ncFile

# move final file to transfer directory

# get date folder
bfile=(${newfile//_/ })
ddir=${bfile[2]}
mkdir -p $root$ddir
finalout=$root$ddir/$newfile



mv $newfile $finalout


done 
# END OF LOOP over 6 LEADTIME FILES
done
# END OF LOOP OVER GRA FILES

export GDAL_NETCDF_BOTTOMUP=YES

