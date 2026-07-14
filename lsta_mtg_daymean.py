# Interface lst_anom code
import os,sys,glob
import rasterio
import numpy as np
import datetime,array
from ctypes import *
import u_interpolate as inter
from scipy.spatial import cKDTree
from pyresample import geometry, kd_tree
from osgeo import gdal
import matplotlib.pyplot as plt
#some VMs have /prj mounted, (satdev,satprod)
# if linux, paths are kept as is (ie, /prj/....)
# if satdev, prefix with /mnt
system = 'Linux'
system = 'satdev'
dx_3km=0.026949456

# remake even if no new data
remake = True
mode = 'realtime'  #or 'historical'
# directories
SCRATCHSAVE='/mnt/scratch/stewells/MTG_LST/full_disc'

ARCHIVE_AFRICA='/mnt/prj/swift/MTG_LST/Africa'
SANDIR = '/mnt/HYDROLOGY_stewells/geotiff/ssa_lsta_mtg'



if mode=='historical':
    idate = datetime.datetime(2026,5,12,7,00)
    remake = True
else: #realtime
    idate = datetime.datetime.utcnow() 

def get_new_data(idate):
    USER='scw005'
    PWD="dnfmomd2063!"
    # arguments for wget
    not_hours=" -R \" *00??.nc, *01??.nc, *02??.nc, *03??.nc, *04??.nc, *19??.nc, *2???.nc, *.txt, *.html, *.tmp\""
    # first get list of all files today 
    today_files = glob.glob(os.path.join(SCRATCHSAVE,'PRODUCTS/MTG/MTLST/NATIVE',idate.strftime('%Y/%m/%d'),'*.nc'))
    tnow =idate
    year = tnow.year
    month = tnow.month  
    day = tnow.day
    os.makedirs(SCRATCHSAVE, exist_ok=True)
    os.system('wget --no-check-certificate -r -np -nH --user=' + USER + ' --password=' + PWD + ' -N -P '+ SCRATCHSAVE +not_hours+'  https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MTG/MTLST/NATIVE/'+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'/')
    # outputs to /mnt/scratch/stewells/MTG_LST/full_disc/PRODUCTS/MTG/MTLST/NATIVE/2026/05
    new_today_files = glob.glob(os.path.join(SCRATCHSAVE,'PRODUCTS/MTG/MTLST/NATIVE',idate.strftime('%Y/%m/%d'),'*.nc'))
    added_files = list(set(new_today_files) - set(today_files))
    return added_files



def generate_anom(idate,outDir,system):

    # set up the ctypes
    if system.lower()=='satdev':
        mount_drive = c_int(1)
    else:
        mount_drive = c_int(0)
    year = c_int(idate.year)
    month = c_int(idate.month)
    day = c_int(idate.day)
    hour = c_int(idate.hour)
    minute = c_int(idate.minute)
    outDir_bytes = outDir.encode('utf-8')
    outD = c_char_p(outDir_bytes)
    plen = c_int(len(outDir))
 
    libPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"liblst_anom.so")
    liblst = cdll.LoadLibrary(libPath)
    #liblst.lst_anom_.argtypes = [c_int, c_int, c_int, c_int, c_char_p]
    liblst.lst_anom_.restype = None    
        
    # call it
    # FORTRAN: subroutine lst_anom(iyear,imonth,iday,mount_drive,outDir)
    liblst.lst_anom_(byref(year),byref(month),byref(day),byref(hour),byref(minute),byref(mount_drive))

    # close handle to DLL
    handle =liblst._handle
    libc = CDLL(None)
    libc.dlclose.argtypes = [c_void_p]
    libc.dlclose.restype = c_int
    result = libc.dlclose(handle)
    del liblst


def get_africa_latlong():
    ifile = '/mnt/prj/swift/MTG_LST/Africa/lon_lat_3300_3670.gra'
    data = np.fromfile(ifile, dtype='float32')
    domain = {'nx_raw':3300,'ny_raw':3670,'deltax':0.03,'bytes':'d'}
    data = data.reshape((domain['nx_raw'], domain['ny_raw'],2), order='F')
    lats = data[:,:,1].T
    lons = data[:,:,0].T   
    
    return lats,lons

def make_geotiff(inDir,idate,outDir):
 #indir: location of .gra file
 #idate : date of data
 #outDir: location to save geotiff to
 #GET DATA
    fname = os.path.join(inDir,idate.strftime('mtg_lsta_0917_%Y%m%d.gra'))
    data = np.fromfile(fname, dtype='float32')
    domain = {'nx_raw':3300,'ny_raw':3670,'bytes':'d'}
    data = data.reshape((domain['nx_raw'], domain['ny_raw'], 2), order='F')
    lsta_mean  = data[:,:,1].T
    lsta_mean[lsta_mean==-999.9]=np.nan

    lats_orig,lons_orig = get_africa_latlong()   
    lats_orig[lats_orig==-91]=np.nan
    lons_orig[lons_orig==-181]=np.nan
    swath = geometry.SwathDefinition(lons=lons_orig, lats=lats_orig)
    lon_min = np.nanmin(lons_orig)
    lon_max = np.nanmax(lons_orig)
    lat_min = np.nanmin(lats_orig)
    lat_max = np.nanmax(lats_orig)

    lon_new = np.arange(lon_min, lon_max, dx_3km)
    lat_new = np.arange(lat_min, lat_max, dx_3km)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)
    target = geometry.GridDefinition(lons=lon_grid, lats=lat_grid)
    lsta_mean_new = kd_tree.resample_nearest(
        swath,
        lsta_mean,
        target,
        radius_of_influence=3000,
        fill_value=np.nan
    )
    rasFile = os.path.join(outDir, 'mtg_tmp_lsta_'+idate.strftime("%Y%m%d")+'.tif')
    reprojFile= os.path.join(outDir, 'LSASAF_lst_anom_Daymean_MTG_'+idate.strftime("%Y%m%d")+'_3857_uncompressed.tif')
    #
    #LSASAF_lst_anom_Daymean_MTG_%year%%month%%day%_3857.tif
    ##finalFile = os.path.join(outDir, 'LSASAF_lst_anom_Daymean_withmask_withHistClim_'+idate.strftime("%Y%m%d")+'_1700_mask_TEST.tif')
    finalFile = os.path.join(outDir, 'LSASAF_lst_anom_Daymean_MTG_'+idate.strftime("%Y%m%d")+'_3857.tif')

    xul = lon_min
    yul = lat_max
    origEPSG='4326'
    newEPSG='3857'
    dat_type='float32'
    #dat_type = data.dtype
    # set missing as -999
    lsta_mean_new[np.isnan(lsta_mean_new)]=-999
    transform = rasterio.transform.from_origin(xul,yul,dx_3km,dx_3km)
    rasImage = rasterio.open(rasFile,'w',driver='GTiff',
                           height=lsta_mean_new.shape[0],width=lsta_mean_new.shape[1],
                           count=1,dtype=str(dat_type),
                           crs = 'EPSG:'+str(origEPSG),
                           nodata=-999,
                           transform = transform)
    rasImage.write(np.flipud(lsta_mean_new[:]),1)
    rasImage.close()
    ds = gdal.Warp(reprojFile, rasFile, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff')
    ds = None  
    # set sea to -998
    os.system('gdal_rasterize -i -burn -998  ../ancils/shapefiles/ssa_landboundary_3857.shp '+ reprojFile)
    # compress
    os.system('gdal_translate  -co compress=LZW -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co PREDICTOR=2 '+reprojFile+' '+finalFile)
    os.system('gdaladdo -r average '+finalFile+' 2 4 8 16 32')
    os.remove(rasFile)
    os.remove(reprojFile)
    """
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].imshow(lsta_mean, origin='lower')
    ax[0].set_title("Original")

    ax[1].imshow(A_new, origin='lower')
    ax[1].set_title("Resampled (3 km)")

    plt.tight_layout()
    plt.show()
    """


if mode=='historical':
    #new_files = get_new_data(idate)
    # test a file we have
    new_files = ['/mnt/scratch/stewells/MTG_LST/full_disc/PRODUCTS/MTG/MTLST/NATIVE/2026/05/12/LSA-007_MTG_MTLST_MTG-FD_202605120700.nc']

    os.makedirs(os.path.join(ARCHIVE_AFRICA,idate.strftime("%Y/%m/%d")),exist_ok=True)
    os.makedirs(os.path.join(SANDIR,idate.strftime("%Y%m%d")),exist_ok=True)
    generate_anom(idate,ARCHIVE_AFRICA,system)
    #make_geotiff(os.path.join(ARCHIVE_AFRICA,idate.strftime("%Y/%m/%d")),idate,os.path.join(SANDIR,idate.strftime("%Y%m%d")))
else: #realtime
    new_files = get_new_data(idate)
    # doesnt matter what they are, just trigger the anomaly generation and geotiff creation for the day (will use the latest file in the directory)
    if len(new_files)>0 or remake:
        print(new_files)
        os.makedirs(os.path.join(ARCHIVE_AFRICA,idate.strftime("%Y/%m/%d")),exist_ok=True)
        os.makedirs(os.path.join(SANDIR,idate.strftime("%Y%m%d")),exist_ok=True)
        generate_anom(idate,ARCHIVE_AFRICA,system)
        make_geotiff(os.path.join(ARCHIVE_AFRICA,idate.strftime("%Y/%m/%d")),idate,os.path.join(SANDIR,idate.strftime("%Y%m%d")))
    
