#!/bin/sh
date
source /etc/profile.d/conda.sh
conda activate py37

# move to working directory
cd /home/stewells/AfricaNowcasting/rt_code


export GDAL_NETCDF_BOTTOMUP=NO
#nodata_value=9.969209999999999e+36
nodata_value=9.96920996838687e+36 
#nodata_value=9969209968386869046778552952102584320.0
large_value=9.00000e+10
# argument to script = full path to NETCDF file














# get list of files edited 
FILES=$(find  /mnt/prj/swift/SEVIRI_LST/lsta_ssa/nrt/202407/LSASAF*20240716*)


for FFILE in $FILES
do

fullfile="$FFILE"

filename=$(basename -- "$fullfile")
filedir=$(dirname -- "$fillfile")
filename="${filename%.*}"


#LSASAF_lst_anom_Daymean_withmask_withHistClim_20220326.nc
curr_date=${filename: -13}
curr_day=${curr_date:: 8}
#root="/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/"
#root="/home/stewells/AfricaNowcasting/satTest/geotiff/lawis_lsta/"
root="/mnt/HYDROLOGY_stewells/geotiff/lawis_lsta/"
wdir="/home/stewells/AfricaNowcasting/tmp/"

mkdir -p $root$curr_day



newfile=$root$curr_day'/LSASAF_lst_anom_Daymean_withmask_withHistClim_'$curr_date'_mask.tif'
newfile_pre=$wdir'tmp_LSASAF_'$curr_date'.tif'
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

gdal_translate  -co compress=LZW $newfile_pre $newfile

# rm tmp*tif
# rm tmp*.vrt
done

export GDAL_NETCDF_BOTTOMUP=YES
