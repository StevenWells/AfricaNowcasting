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
import argparse
import process_realtime_fns as fns   #will need to move this code into sftp_extract directory

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


# SAN transfer folder
outPath = '/mnt/HYDROLOGY_stewells/geotiff/lawis_soil_moisture_anomaly/'
###
outPathBackup ='/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/'


# get arguments
parser= argparse.ArgumentParser(description="Generate soil moisture anomaly data from .gra files")

# mode (default realtime)
parser.add_argument("--mode", choices=["realtime","historical"], default="realtime",help="Run mode (real time or historical)")

# datetimes for historical
parser.add_argument("--startDate", type=str, help="Start Date (YYYYMMDDhhmm).")
parser.add_argument("--endDate", type=str, help="Start Date (YYYYMMDDhhmm).")

# output directory, to be used if different from the SAN. Defaults to SAN
parser.add_argument("--outDir", type=str, default = outPath, help="Directory to send outputs to (defaults to SAN)")

#domain
parser.add_argument("--domain", choices=["WA","SSA","SSA_6k"], default="SSA",help="Domain: West Africa (WA) or Sub-Saharan Africa (SSA) or High Res SSA (SSA_6k)")

# load them
args = parser.parse_args()

# path to source
if args.domain=='WA':
    sourcePath = '/mnt/prj/swift/ASCAT_cmt/NRT_anomalies/'
elif args.domain == 'SSA_6k':
    sourcePath = '/mnt/scratch/cmt/sm_6km_test/'
else:
    sourcePath = '/mnt/prj/swift/ASCAT_SSA/NRT_anomalies/'


domainPars = {'WA':{'nx_raw':274,'ny_raw':162,'deltax':0.25,'xll':-18.125,'yll':-15.125,'bytes':'f'},
             'SSA':{'nx_raw':293,'ny_raw':242,'deltax':0.25,'xll':-18.125,'yll':-35.125,'bytes':'f'},
             'SSA_6k':{'nx_raw':730,'ny_raw':602,'deltax':0.1,'xll':-17.95,'yll':-35.0,'bytes':'h'}}

# backup output folder if satdev is down
toSdir = False



# projections
origEPSG='4326'
newEPSG='3857'

# cron fequency (minutes)
cronFreq = 60
# sort the dates out
if args.mode =='historical':
    if not args.startDate:
            parser.error("Start date (--startDate) not specified for historical processing")
    if not args.endDate:
        parser.error("End date (--endDate) not specified for historical processing")
    try:
        startDate = datetime.datetime.strptime(args.startDate,'%Y%m%d')
        endDate = datetime.datetime.strptime(args.endDate,'%Y%m%d')
    except:
        print("ERROR: incorrect format for dates. Need to be YYYYMMDDhhmm")
        sys.exit(0)
    if endDate < startDate:
        print("End date provided is before the start date!")
        sys.exit(0) 
    # round to nearest 15 minutes
    round_sdate = fns.roundDate(startDate)
    round_edate = fns.roundDate(endDate)
    if round_sdate != startDate:
        print("start date rounded to nearest interval matching raw data")
        startDate = round_sdate
    if round_edate != endDate:
        print("start date rounded to nearest interval matching raw data")
        endDate = round_edate
    # get list of dates (Soil moisture is daily)
    dateList = fns.generate_dates(startDate,endDate,1440)

    # get list of files
    all_files = []
    for x in dateList:
        print([x,sourcePath])
        if args.domain == 'SSA_6k':
            all_files+=glob.glob(os.path.join(sourcePath,x.strftime('ASCAT_dsm_%Y%m%d_*')))
        else:
            all_files+=glob.glob(os.path.join(sourcePath,x.strftime("%Y%m"),x.strftime('ASCAT_dsm_%Y%m%d_*')))

    #all_files=glob.glob('/mnt/prj/swift/ASCAT_SSA/NRT_anomalies/202303/ASCAT_dsm_20230313*')
else: # realtime - this has all the files that were modified since the last time the cron job was run
      #if the file was editd in that time, update it (or add if new
    all_files = []
    if args.domain == 'SSA_6k':
        total_files=glob.glob(os.path.join(sourcePath,'ASCAT_dsm_*'))
    else:
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
    nx_raw=domainPars[args.domain]['nx_raw']
    ny_raw=domainPars[args.domain]['ny_raw']
    deltax=domainPars[args.domain]['deltax']
    xll = domainPars[args.domain]['xll']
    yll=domainPars[args.domain]['yll']

    if toSdir:
        outdir= outPathBackup
    else:
        #outdir = os.path.join(outPath,dateStr)
        outdir = os.path.join(args.outDir,dateStr)
    rasFile = filename.split('.')[0]+'.tif'
    if args.domain == 'SSA_6k':
        reprojFile = filename.split('.')[0]+'_6k_'+str(newEPSG)+'.tif'
    else:
        reprojFile = filename.split('.')[0]+'_'+str(newEPSG)+'.tif'
    print(outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #sys.exit(0)
    #GET DATA
    gen_ints = array.array(domainPars[args.domain]['bytes'])
    gen_ints.fromfile(open(root, 'rb'), os.path.getsize(root) // gen_ints.itemsize)
    data=np.array(gen_ints).reshape(ny_raw,nx_raw)
    data= data[:]

    if args.domain == 'SSA_6k':
        data = data/100.
        data[data< -300] = -999

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
















    

