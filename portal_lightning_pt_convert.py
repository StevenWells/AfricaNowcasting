#
# portal_lightning_pt_convert.py
# Author: Steven Wells (UKCEH)
# 
# Convert raw .gra lightning grids from Chris Taylor to CSV format listing pixels in 
# SSA domain where lightning flashes occur
#
# Usage:  python portal_lightning_pt_convert.py <mode> <<startdate>> <<enddate>>
#         <mode> = 'realtime' or 'historical'. If missing, defaults to realtime
#         <<startdate>>, <<enddate>>: if <mode>==historical, then start and end date define the period 
#         NB. Dates will be rounded to nearest 15 minutes to align with input data
          
import os, sys
import numpy as np
import array
import matplotlib.pyplot as plt
import rasterio, time
from osgeo import gdal
import glob,datetime

# numbe rof flashes / 15 minutes - datestamp is start of 15 minutes
#origin = 35.975S, 19.975W - lower left
# 
 # deltax= 0.05 degrees
ny = 1440
nx = 1400
orig_N = -35.975+0.05*ny
raw_tdelta = 15  # minutes of raw data interval
cronFreq = 15
 #makes lightning csv file of point locations from gridded data

dataDir = '/mnt/scratch/cmt/flash_count_NRT'
#outDir = '/mnt/HYDROLOGY_stewells/lawis-west-africa/mtg_lightning'
outDir= '/home/stewells/AfricaNowcasting/dev/lightning/test_out'
#archiveDir='/mnt/prj/nflics/lightning'
archiveDir='/home/stewells/AfricaNowcasting/dev/lightning/test_out'

mode = 'realtime'

if len(sys.argv)>1: 
    if sys.argv[1] in ['realtime','historical']:
        mode = sys.argv[1]
    else:
        print("ERROR: Incorrect mode speficied. Must be either \'realtime\' or \'historical\'")
        sys.exit(0) 
# if historical then load the start and end date

def roundDate(dt,nmin=raw_tdelta):
    mins = (dt.minute // nmin)* nmin
    if dt.minute % nmin >= 7.5:
        mins+=nmin
    return dt.replace(minute=0,second=0,microsecond=0)+datetime.timedelta(minutes=mins)
def generate_dates(start,end,interval):
    dateList = []
    current = start
    delta = datetime.timedelta(minutes =interval)
    while current <= end:
        dateList.append(current)
        current+= delta
    return dateList

if mode=='historical':
    try:
        startDate = datetime.datetime.strptime(sys.argv[2],'%Y%m%d%H%M')
        endDate = datetime.datetime.strptime(sys.argv[3],'%Y%m%d%H%M')
    except:
        print("ERROR: incorrect format for dates. Need to be YYYYMMDDhhmm")
        sys.exit(0)
    if endDate < startDate:
        print("End date provided is before the start date!")
        sys.exit(0) 
    # round to nearest 15 minutes
    round_sdate = roundDate(startDate)
    round_edate = roundDate(endDate)
    if round_sdate != startDate:
         print("start date rounded to nearest interval matching raw data")
         startDate = round_sdate
    if round_edate != endDate:
         print("start date rounded to nearest interval matching raw data")
         endDate = round_edate
    # get list of dates
    dateList = generate_dates(startDate,endDate,raw_tdelta)
    # make into a file list
    fileList = [os.path.join(dataDir,x.strftime("%Y%m%d%H%M.gra")) for x in dateList]
    fileListArchive = [os.path.join(archiveDir,str(x.year),str(x.month).zfill(2),str(x.day).zfill(2),x.strftime("%Y%m%d%H%M.gra")) for x in dateList]


    # keep only those files that exist and store as dates
    exist_files = []
    for ix,ifile in enumerate(fileList):
        if os.path.exists(ifile):
            exist_files+=[ifile]
        elif os.path.exists(fileListArchive[ix]):
            exist_files+=[fileListArchive[ix]]
    # exist_files contains a list of files (and path to them, either rt or archive) that exist
    
    # get list of dates that exist
    new_files = [x.split('/')[-1].split('.')[0] for x in exist_files]
    
    

elif mode=='realtime':
    # get recent files
    new_files = []

    t0 = datetime.datetime.today()
    #total_files=glob.glob(os.path.join(dataDir,str(t0.year),str(t0.month).zfill(2),'*.gra'))
    total_files=glob.glob(os.path.join(dataDir,'*.gra'))

    for f in total_files:
        modTimesinceEpoc = os.path.getmtime(f)
        modificationTime = datetime.datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
        
        if modificationTime > datetime.datetime.today()-datetime.timedelta(minutes=cronFreq):
            # now check to see if already processed?
            ### NB dont need to do this I think as the most recent files amy get updated. 
            idate = f.split('/')[-1].split('.')[0][:8]
            itime = f.split('/')[-1].split('.')[0][8:12]
            tiffPath = os.path.join(outDir,idate[:8],'mtg_lightning_SSA_'+idate+itime+'_3857.tif')
            #os.makedirs(os.path.join(archiveDir,idate[:4],idate[4:6],idate[6:8]),exist_ok=True)
            #archiveFile = os.path.join(archiveDir,idate[:4],idate[4:6],idate[6:8],idate+itime+'.gra')
            #os.system('cp '+f+' '+archiveFile)
            ###if not os.path.exists(tiffPath):     

        #only include if not already processed
                ###new_files.append(idate+itime)
            ###elif os.path.exists(tiffPath) and overwrite:
            new_files.append(idate+itime)
    exist_files = [os.path.join(dataDir,x+'.gra') for x in new_files]

# exist_files = [list of paths to the files that exist]
# new_files = [lis of datetimes of the files that exists (As string)]

if len(new_files)==0:
     		print("No new files to process")

# sort acording to date
sorted_pairs = sorted(zip(new_files,exist_files))
new_files, exist_files = zip(*sorted_pairs)
new_files = list(new_files)
exist_files = list(exist_files)
#new_files = sorted(new_files)


for rundate in new_files:
    nHist = 4
    print("(re)processing "+rundate)
    # get last four images
    dt_now = datetime.datetime.strptime(rundate,"%Y%m%d%H%M")
    backdates = [(dt_now - datetime.timedelta(minutes=15)*x).strftime("%Y%m%d%H%M") for x in range(nHist)]
    # get the backpaths, which may be from a) live raw feed, b) archive or c) missing
    backpaths = []
    for iback, back in enumerate(backdates):
        rtFile = os.path.join(dataDir,back+'.gra')
        archiveFile = os.path.join(archiveDir,back[:4],back[4:6],back[6:8],back+'.gra')
        if os.path.exists(rtFile):
            backpaths+=[rtFile]
        elif os.path.exists(archiveFile):
            backpaths+=[archiveFile]
        else:
             backpaths+=['MISSING']


    backpaths = backpaths[::-1]      


    sob = np.zeros((ny,nx))
    for idx,ifile in enumerate(backpaths):
        if ifile=='MISSING':
             data = np.zeros((ny,nx))
        else:       # file should exist 
            gen_ints = array.array("h")
            gen_ints.fromfile(open(ifile,'rb'),os.path.getsize(ifile) // gen_ints.itemsize)
            data = np.array(gen_ints).reshape(ny,nx)

        sob[data>0] = nHist - idx
        lats  = [-35.975+0.05*x for x in range(ny)]
        lons = [-19.975 + 0.05*x for x in range(nx)]
        l_inds= np.argwhere(sob)
        targetDir = os.path.join(outDir,rundate[:8])
        if not os.path.exists(targetDir):
            os.makedirs(targetDir,exist_ok=True)
        with open(os.path.join(targetDir,"lightning_points_"+rundate+".csv"),'w') as f:
            f.write('Latitude,Longitude,Window\n')
            for point in range(len(l_inds)):
                f.write(str(round(lats[l_inds[point][0]],3))+','+str(round(lons[l_inds[point][1]],3))+','+str(round(sob[l_inds[point][0],l_inds[point][1]],0))+'\n')

    #archive the file if it is only in the rt code
    archiveFile = os.path.join(archiveDir,rundate[:4],rundate[4:6],rundate[6:8],rundate+'.gra')
    rtFile = os.path.join(dataDir,rundate+'.gra')
    if os.path.exists(rtFile) and not os.path.exists(archiveFile):
        os.makedirs(os.path.join(archiveDir,rundate[:4],rundate[4:6],rundate[6:8]),exist_ok=True)      
        os.system('cp '+rtFile+' '+archiveFile)