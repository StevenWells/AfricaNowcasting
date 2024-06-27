import array
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
import glob
import xarray as xr
import datetime, time
import matplotlib.pyplot as plt


# Converts .gra soil moisture files into geotiff and sends them to thesans tranfer folder


# West Africa Chris Note
# /mnt/prj/swift/ASCAT_cmt/NRT_anomalies
# Beneath that there should be a directory for each month eg 202203 containing up to 2 files each day, 
# one from the morning overpasses and one from the evening. These are a # single array of 274 x 162 floats starting at 18.125W and 15.125 south with grid length of # #0.25degrees. 
# 
# SSA CHris note
#extended domain covering S Africa and Somalia (293x242 pixels rather than 274x162 pixels). The new files are in 
#/mnt/prj/swift/ASCAT_SSA/NRT_anomalies/202303/
#which is the same as previously apart from the directory name ASCAT_SSA which replaces ASCAT_cmt (SSA stands for sub-Saharan Africa).
#The old grid started at 15.125S whilst the new one starts at 35.125S. In the east, the new file runs to 54.875E whereas the old file ran to 50.125E.



# two files per day, am and pm 


# 1 list files in folder. 

# get subsetof files that have been edit in last 15 minutes (or time increment of cron job

# for each file, convert to geotiff and push to sans tranfer folder - nb need to update that tranfer file to include soil moisture


#  SCW 13 Mar 2023     Generalised coordinates by introducing domain argument which picks out parameters from domainPars dictionary


runtype=sys.argv[1]
# realtime - get list of files based on those most recently updated
# historical - get list of files based on a pattern match 
domain = sys.argv[2]
# domain will define shape of array being read in 
# should be either 'WA'=West Africa  or 'SSA' = Sub-saharam africa

# path to source
if domain=='WA':
    sourcePath = '/mnt/prj/swift/ASCAT_cmt/NRT_anomalies/'
else:
    sourcePath = '/mnt/prj/swift/ASCAT_SSA/NRT_anomalies/'


domainPars = {'WA':{'nx_raw':274,'ny_raw':162,'deltax':0.25,'xll':-18.125,'yll':-15.125},
             'SSA':{'nx_raw':293,'ny_raw':242,'deltax':0.25,'xll':-18.125,'yll':-35.125}}




# SANS transfer folder

#outPath = '/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/'
outPath = '/mnt/HYDROLOGY_stewells/geotiff/lawis_soil_moisture_anomaly/'
#outPath = '/home/stewells/AfricaNowcasting/satTest/geotiff/lawis_soil_moisture_anomaly/'

# projections
origEPSG='4326'
newEPSG='3857'

# cron fequency (minutes)
cronFreq = 15

if runtype=='historical':
    all_files=glob.glob('/mnt/prj/swift/ASCAT_SSA/NRT_anomalies/202303/ASCAT_dsm_20230313*')
else: # realtime - this has all the files that were modified since the last time the cron job was run
      #if the file was editd in that time, update it (or add if new
    all_files = []
    total_files=glob.glob(os.path.join(sourcePath,'*','ASCAT_dsm_*'))
    for f in total_files:
        modTimesinceEpoc = os.path.getmtime(f)

        modificationTime = datetime.datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
        if modificationTime > datetime.datetime.today()-datetime.timedelta(minutes=cronFreq):
            all_files.append(f)    

if len(all_files)==0:
    print("No files to process")
for root in all_files:
    #root = sys.argv[1]
    filename = os.path.basename(root)
    print(filename)
    dateStr = filename.split('_')[2]
    nx_raw=domainPars[domain]['nx_raw']
    ny_raw=domainPars[domain]['ny_raw']
    deltax=domainPars[domain]['deltax']
    xll = domainPars[domain]['xll']
    yll=domainPars[domain]['yll']
    outdir = os.path.join(outPath,dateStr)
    rasFile = filename.split('.')[0]+'.tif'
    reprojFile = filename.split('.')[0]+'_'+str(newEPSG)+'.tif'
    print(outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #sys.exit(0)
    #GET DATA
    gen_ints = array.array("f")
    gen_ints.fromfile(open(root, 'rb'), os.path.getsize(root) // gen_ints.itemsize)
    data=np.array(gen_ints).reshape(ny_raw,nx_raw)
    data= data[:]

    # lowerleft is x=-18.125 y=-15.125
    # therefore ueer left is -15.125+nrows*0.15
    xul= xll
    yul = yll+data.shape[0]*deltax
    transform = rasterio.transform.from_origin(xul,yul,deltax,deltax)
    dat_type='float64'
    #dat_type = data.dtype
    rasImage = rasterio.open(rasFile,'w',driver='GTiff',
                           height=data.shape[0],width=data.shape[1],
                           count=1,dtype=str(dat_type),
                           crs = 'EPSG:'+str(origEPSG),
                           nodata=-999.9,
                           transform = transform)
    rasImage.write(np.flipud(data[:]),1)
    rasImage.close()
    ds = gdal.Warp(reprojFile, rasFile, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff')
    ds = None  
    os.system('mv '+reprojFile+' '+outdir+'/')

    os.system('rm '+rasFile)
    #os.system('mv '+rasFile+' '+outdir)




#if do_plot:
#    cmap='Blues'
#    use_cmap = shiftedColorMap(matplotlib.cm.bwr,midpoint=0.6,name='shifted')
#    ax=plt.subplot()
#    im=ax.imshow(rasImage_pc,cmap=new_cmap)
















    

