import array
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
import xarray as xr
import datetime

# Python script to accompany lws_convert.sh
# Converts .gra LWS files into NetCDF


#LSTA grid 1436x714.
# /scratch/cmt/prob_vn2a_YYYYMMDD.gra.
# The data are written x first, then y, starting in the SW corner. There are 9 layers.
# Want to show z=8 (valid at 12Z, z=9 (valid at 15Z) and z=3 to 6 (valid at 18,21,0 and 3Z) 
# climatological probability P of a convective core given particular LSTA, and I suggest the data you create for Gemma should be 10*(P-10),



origEPSG='4326'
newEPSG='3857'

root = sys.argv[1]
filename = os.path.basename(root)
nx = 1436
ny = 714
nz=9

# hours
validTime = [18,21,0,3,12,15]

#filedate
todayDate = datetime.datetime.strptime(filename.split('_')[-1].split('.')[0],'%Y%m%d')
tomorrowDate = todayDate + datetime.timedelta(days=1)
tomorrowFile = 'prob_vn2a_'+tomorrowDate.strftime('%Y%m%d')
todayFile = 'prob_vn2a_'+todayDate.strftime('%Y%m%d')


gen_ints = array.array("f")
gen_ints.fromfile(open(root, 'rb'), os.path.getsize(root) // gen_ints.itemsize)
data=np.array(gen_ints).reshape(nz,ny,nx)
    

for ix,iMap in enumerate([2,3,4,5,7,8]):
    image = data[iMap,:]
    image_tmp = np.copy(image)
    
    image = 10*(image-10)
    image[image_tmp<0]= -999
    if validTime[ix] in [3]:
        rasFile = tomorrowFile+'_'+str(validTime[ix])+'.nc'
    elif validTime[ix] ==0:
        rasFile = filename.split('.')[0]+'_'+str(24)+'.nc'
        rasFile = todayFile+'_'+str(24)+'.nc'
    else:
        rasFile = filename.split('.')[0]+'_'+str(validTime[ix])+'.nc'
        rasFile = todayFile+'_'+str(validTime[ix])+'.nc'
    ds=xr.Dataset() 
    ds['LWS']=xr.DataArray((image),dims=['ys_mid', 'xs_mid']) 
    ds.attrs['time']='3h'
    ds.attrs['grid']="LWS"
    ds.attrs['missing']="-999"
    ##output
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path=rasFile,mode='w', encoding=enc, format='NETCDF4')
    

