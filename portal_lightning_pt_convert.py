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
 #makes lightning csv file of point locations from gridded data

dataDir = '/mnt/scratch/cmt/flash_count_NRT'
outDir = '/mnt/HYDROLOGY_stewells/lawis-west-africa/mtg_lightning'
#outDir= '/home/stewells/AfricaNowcasting/dev/lightning/'
tmpDir='/home/stewells/AfricaNowcasting/dev/lightning/'
archiveDir='/mnt/prj/nflics/lightning'

# get recent files
new_files = []
cronFreq=25
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
        os.makedirs(os.path.join(archiveDir,idate[:4],idate[4:6],idate[6:8]),exist_ok=True)
        archiveFile = os.path.join(archiveDir,idate[:4],idate[4:6],idate[6:8],idate+itime+'.gra')
        os.system('cp '+f+' '+archiveFile)
        ###if not os.path.exists(tiffPath):     

	#only include if not already processed
			###new_files.append(idate+itime)
		###elif os.path.exists(tiffPath) and overwrite:
        new_files.append(idate+itime)
if len(new_files)==0:
     		print("No new files to process")
new_files = sorted(new_files)

for rundate in new_files:
        nHist = 4
        print("(re)processing "+rundate)
        # get last four images
        dt_now = datetime.datetime.strptime(rundate,"%Y%m%d%H%M")
        backdates = [(dt_now - datetime.timedelta(minutes=15)*x).strftime("%Y%m%d%H%M") for x in range(nHist)]
        backpaths  = [os.path.join(dataDir,x+'.gra') for x in backdates	][::-1]
        sob = np.zeros((ny,nx))
        for idx,ifile in enumerate(backpaths):
            gen_ints = array.array("h")
				#try:
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