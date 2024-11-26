#!/bin/sh
date
source /etc/profile.d/conda.sh
conda activate py37

# move to working directory
cd /home/stewells/AfricaNowcasting/rt_code


export GDAL_NETCDF_BOTTOMUP=NO
#nodata_value=9.969209999999999e+36
nodata_value=9.96920996838687e+36 
file_prefix="LSASAF_lst_anom_Daymean_withmask_withHistClim_"
file_suffix="_1700.nc"
#nodata_value=9969209968386869046778552952102584320.0
large_value=9.00000e+10
# argument to script = full path to NETCDF file
indir="/mnt/prj/swift/SEVIRI_LST/lsta_ssa/nrt/"

# default argument values
mode="realtime"
outdir="/mnt/HYDROLOGY_stewells/geotiff/lawis_lsta/"
startDate=""
endDate=""
reprocess="false"
OPTS=$(getopt -o m:d:s:e:p:h --long mode:,outdir:,startDate:,endDate:,reprocess:,help -n "lst_convert.sh" -- "$@")
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
    -p|--reprocess)
      reprocess="$2"
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 [-m mode] [-d outdir] [-s startDate] [-e endDate] [-p reprocess]"
      echo "Defaults:"
      echo "  -m, --mode: $mode"
      echo "  -d, --outdir: $outdir"
      echo "  -s, --startDate: $startDate"
      echo "  -e, --endDate: $endDate"
      echo "  -p, --reprocess: $reprocess"
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

if [[ "$mode" == "realtime" ]]; then
# get list of files edited 
     FILES=($(find  "$indir"  -type f -mmin -15))
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
        yearmonth=${current_date:0:6}

        files+=("$indir/$yearmonth/$file_prefix$current_date$file_suffix")
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


for FFILE in "${FILES[@]}"
     do
     echo $FFILE
     fullfile="$FFILE"

     filename=$(basename -- "$fullfile")
     filedir=$(dirname -- "$fillfile")
     filename="${filename%.*}"


     #LSASAF_lst_anom_Daymean_withmask_withHistClim_20220326.nc
     curr_date=${filename: -13}
     curr_day=${curr_date:: 8}
     #root="/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/"

    # root="/mnt/HYDROLOGY_stewells/geotiff/lawis_lsta/"
     root=$outdir
     wdir="/home/stewells/AfricaNowcasting/tmp/"

     # final file on the SAN
     newfile=$root$curr_day'/LSASAF_lst_anom_Daymean_withmask_withHistClim_'$curr_date'_mask.tif'
     if [ ! -f "$newfile" ] || [ "$reprocess" = "true" ]; then

          mkdir -p $root$curr_day

          #newfile=$root'/LSASAF_lst_anom_Daymean_withmask_withHistClim_'$curr_date'_mask.tif'


          newfile_pre=$wdir'tmp_LSASAF_'$curr_date'.tif'
          newfile_ready=$wdir'tmp_LSASAF_'$curr_date'_ready.tif'
          tmptif=$wdir'tmpfile_lst.tif'
          tmpvrt=$wdir'tmpfile2_lst.vrt'

          #tidy up
          rm $wdir'tmpfile_lst'*tif
          rm $wdir'tmpfile2_lst'*.vrt
          rm $wdir'tmp_LSASAF'*.tif


          gdal_calc.py --format=GTiff --type=Float32 -A $fullfile --calc="(A!=$nodata_value)*A +(A==$nodata_value)*-999" --outfile $tmptif
          echo $fullfile


          echo "TRANSLATE"
          gdal_translate -of VRT $tmptif $tmpvrt



          # edit the VRT file
          new_section='<Metadata domain="GEOLOCATION">\n<MDI key="X_DATASET">NETCDF:"../ancils/SEVIRI_SSA_geoloc_v2.nc":SSA_LON<\/MDI>\n<MDI key="X_BAND">1<\/MDI>\
          <MDI key="Y_DATASET">NETCDF:"../ancils/SEVIRI_SSA_geoloc_v2.nc":SSA_LAT<\/MDI>\n<MDI key="Y_BAND">1<\/MDI>\
          <MDI key="PIXEL_OFFSET">0<\/MDI>\n<MDI key="LINE_OFFSET">0<\/MDI>\n<MDI key="PIXEL_STEP">1<\/MDI>\n<MDI key="LINE_STEP">1<\/MDI>\n'

          sed -i 's@.*"IMAGE_STRUCTURE".*@'"$new_section"'@g' $tmpvrt

          sed -i 's@.*"INTERLEAVE".*@@g' $tmpvrt

          echo "WARP"

          new_nodata_value='-999'
          old_nodata_value='1.175494351e-38'

          if [ ! -f $newfile_pre ]; then
               gdalwarp -of GTiff -co compress=lzw -geoloc -s_srs EPSG:4326 -t_srs EPSG:3857 -dstnodata "$new_nodata_value" $tmpvrt $newfile_pre
          fi
          #tidy up


          # flip the image -dont need!
          #python3 gdal_functions.py result.tif

          # convert the sea into a different value from missing
          gdal_rasterize -i -burn -998  ../ancils/shapefiles/ssa_landboundary_3857.shp $newfile_pre
          # get rid of any erroneous leftover nans
          gdal_calc.py -A $newfile_pre --calc="(A<$large_value)*A + (A>$large_value)*-999" --outfile $newfile_pre --NoDataValue=-999

          gdal_translate  -co compress=LZW $newfile_pre $newfile_ready
          ls $wdir
          cp -f $newfile_ready $newfile

          rm -f $newfile_ready
          rm -f $newfile_pre
          # rm tmp*.vrt

     else
          echo "Skipping: Already processed and/or reprocessing is disabled"
     fi
done
export GDAL_NETCDF_BOTTOMUP=YES
