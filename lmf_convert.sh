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

date
# load conda environment
source /etc/profile.d/conda.sh
conda activate py37
# Move to workign directory (this script)
cd /home/stewells/AfricaNowcasting/rt_code/
# set up GDAL parameters
export GDAL_NETCDF_BOTTOMUP=NO


# fixed variables
indir="/mnt/prj/nflics/LMF_output_probabilities/"
file_prefix="prob_vn2a_"
file_suffix=".gra"
tmpdir="/home/stewells/AfricaNowcasting/tmp/"
nodata_value=-999

# default argument values
mode="realtime"
outdir="/mnt/HYDROLOGY_stewells/geotiff/lawis_lmf/"
startDate=""
endDate=""



OPTS=$(getopt -o m:d:s:e:h --long mode:,outdir:,startDate:,endDate:,help -n "date_parsing.sh" -- "$@")
if [[ $? -ne 0 ]]; then
  echo "Error parsing arguments" >&2
  exit 1
fi
# Rearrange arguments as returned by getopt
eval set -- "$OPTS"
# Parse the options
while true; do
  case "$1" in
    -m|--mode)
      mode="$2"
      shift 2 ;;
    -d|--outdir)
      outdir="$2"
      shift 2 ;;
    -s|--startDate)
      startDate="$2"
      shift 2 ;;
    -e|--endDate)
      endDate="$2"
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 [-m mode] [-d outdir] [-s startDate] [-e endDate]"
      echo "Defaults:"
      echo "  -m, --mode: $mode"
      echo "  -d, --outdir: $outdir"
      echo "  -s, --startDate: $startDate"
      echo "  -e, --endDate: $endDate"
      exit 0 ;;
    --)
      shift
      break ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1 ;;
  esac
done


# make sure output and input dir have a trailing "/"
[[ "${indir: -1}" != "/" ]] && indir="$indir/"
[[ "${outdir: -1}" != "/" ]] && outdir="$outdir/"

# get the files
if [[ "$mode" == "realtime" ]]; then
	# pick up files modified in the last 20 minutes
	# FILES=$(find /prj/nflics/LMF_output_probabilities -regextype posix-egrep -regex ".*prob_vn2b_.*[0-9]+.gra" -type f -mmin -20)
	#FILES=$(find /mnt/prj/nflics/LMF_output_probabilities -regextype posix-egrep -regex ".*prob_vn2a_.*[0-9]+.gra" -type f -mmin -20)
	FILES=($(find "$indir" -regextype posix-egrep -regex ".*prob_vn2a_.*[0-9]+.gra" -type f -mmin -20))
	echo "Files to process (realtime): ${#FILES[@]}"

elif  [[ "$mode" == "historical" ]]; then
    # sort dates out
    # Validate date format using regex
    if ! [[ "$startDate" =~ ^[0-9]{4}[0-9]{2}[0-9]{2}$ && "$endDate" =~ ^[0-9]{4}[0-9]{2}[0-9]{2}$ ]]; then
        echo "Invalid date format. Please use YYYYMMDD."
        exit 1
    fi
    # Convert dates to seconds since epoch for comparison
    start_epoch=$(date -d "$startDate" +%s 2>/dev/null)
    end_epoch=$(date -d "$endDate" +%s 2>/dev/null)

    if [ -z "$start_epoch" ] || [ -z "$end_epoch" ]; then
        echo "Error: Invalid date(s) provided."
        exit 1
    fi
    # Ensure start_date is less than or equal to end_date
    if [ "$start_epoch" -gt "$end_epoch" ]; then
        echo "Error: Start date must be earlier than or equal to end date."
        exit 1
    fi

    # Generate and print dates
    dates=()
    files=()
    current_date="$startDate"
    while [ "$(date -d "$current_date" +%s)" -le "$end_epoch" ]; do
        dates+=("$current_date")
        files+=("$indir$file_prefix$current_date$file_suffix")
        current_date=$(date -d "$current_date +1 day" +%Y%m%d)
    done
    
    # existing files
    FILES=()
    for file in "${files[@]}"; do
        if [[ -e "$file" ]]; then
            FILES+=("$file")
        fi
    done
    echo "Files to process: ${#FILES[@]}"
else 
    echo "Error incorrect option for mode. Allowed values are 'realtime' or 'historical'"
    exit 1
fi


# PROCESS EACH FILE IN LIST
shopt -s nullglob
echo $FILES
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
rm $tmpdir'tmp_lmf'*tif
rm $tmpdir'tmp_lmf'*.vrt

tmptif=$tmpdir'tmpfile_lmf.tif'
tmpvrt=$tmpdir'tmpfile2_lmf.vrt'

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
mkdir -p $outdir$ddir
finalout=$outdir$ddir/$newfile



mv $newfile $finalout


done 
# END OF LOOP over 6 LEADTIME FILES
done
# END OF LOOP OVER GRA FILES

export GDAL_NETCDF_BOTTOMUP=YES

