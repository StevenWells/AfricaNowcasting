import rasterio
import xarray as xr
import numpy as np
from osgeo import gdal
import netCDF4 as nc
import datetime,time
import os,glob,sys,argparse
from itertools import chain

nrows = 2962
ncols=  2777
delta_y = 0.02701789935
delta_x = 0.02701729106

dataDir='/mnt/prj/swift/rain_over_africa'
tmpDir='/home/stewells/AfricaNowcasting/tmp/'
geotiffDir = '/mnt/HYDROLOGY_stewells/geotiff/ssa_africarain_precip_accum/'
backupDir = '/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data'
#geotiffDir = '/home/stewells/AfricaNowcasting/satTest/'
testDate = '202406201300'
accPeriods = [1,3,6,24,48,72]

toSdir = False


def make_geoTiff(data,rasFile,doReproj = True,origEPSG='4326',newEPSG='3857',reprojFile='test.tif',trim=False):
    nbands = len(data)
    dat_type = str(data[0].dtype)
    transform = rasterio.transform.from_origin(-20,40,delta_x,delta_y)
    gdal_options={ "PROFILE": "GeoTIFF", "COMPRESS": "DEFLATE", "ZLEVEL":  "9" }
    # rasImage = rasterio.open(rasFile,'w',driver='GTiff', **gdal_options,
    #    rasImage = rasterio.open(rasFile,'w', driver='GTiff', **gdal_options,
    #     compress='DEFLATE',zlevel='9',
    rasImage = rasterio.open(rasFile,'w', driver='GTiff',zlevel='9',
                           height=data[0].shape[0],width=data[0].shape[1],
                           count=nbands,dtype=dat_type,
                           crs = 'EPSG:'+str(origEPSG),
                           transform = transform
                        )
    for ix,Image in enumerate(data):
        rasImage.write(np.flipud(Image[:]),ix+1)
    rasImage.close()
    if trim:
    #crop parameters
        upper_left_x = -20
        upper_left_y = 26
        lower_right_x = 55
        lower_right_y = -36.5
        window = (upper_left_x,upper_left_y,lower_right_x,lower_right_y)
        rasFile2 = rasFile[:-4]+'_chop.tif'
        gdal.Translate(rasFile2, rasFile, projWin = window)
    else:
        rasFile2 = rasFile
    if doReproj:
        ds = gdal.Warp(reprojFile, rasFile2, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff',creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        ds = None

    os.system('rm '+rasFile2)


def getAccs(tnow,accPeriods,dataDir,tmpDir,geotiffDir):
    print("Generating accumulations for "+str(tnow))
    tnowStr = tnow
    tnow  = datetime.datetime.strptime(tnow,'%Y%m%d%H%M')
    
    if not os.path.exists(os.path.join(geotiffDir,tnowStr[:8])):
        os.mkdir(os.path.join(geotiffDir,tnowStr[:8]))

    # generate list of dates & files to be included in the accumualtion
    datelist = [tnow - n*datetime.timedelta(minutes=15)  for n in range(4*max(accPeriods)+1)]
    datelist_plus15 = [x+datetime.timedelta(minutes=15) for x in datelist]

    filelist = [os.path.join(dataDir,str(x.year),str(x.month).zfill(2),'MSG3'+x.strftime('%Y%m%d-S%H%M-E')+(x+datetime.timedelta(minutes=15)).strftime('%H%M')+'.nc') for x in datelist]
    # reverse to be working backwards from t0

    #filelist = filelist[::-1]
    iacc = np.zeros((2962,2777))
    #initialise total
    accArray = np.copy(iacc)
    for ix,ifile in enumerate(filelist):
        #print(ifile)
        try:
            dfile = xr.open_dataset(ifile)
            iacc =dfile.variables['posterior_mean'][:,:]
        except:
            print("Missing file "+ifile)

        if ix==0: # half the first value for accumulation
            iacc = iacc/2.0
        if ix in [x*4-1 for x in accPeriods]: # list of indices corresponding to accumulation periods
            acchr = int((ix+1)/4)
            print(str(acchr)+'hr')
            accArr_i = np.array(np.round(0.25*np.add(accArray,iacc/2.0),2))
            accArr_i[accArr_i < 1] = 0.0
            rasPath = os.path.join(tmpDir,"HSAF_precip_acc"+str(acchr)+"h_"+tnowStr+"_SSA.tif")
            if toSdir:
                rasPath_3857 = os.path.join(backupDir,"rainoverAfrica_SSA_"+tnowStr+"_acc"+str(acchr)+"h_3857.tif")
            else:
                rasPath_3857 = os.path.join(geotiffDir,tnow.strftime('%Y%m%d'),"rainoverAfrica_SSA_"+tnowStr+"_acc"+str(acchr)+"h_3857.tif")

            make_geoTiff([accArr_i],rasPath,reprojFile=rasPath_3857,trim=True)
            os.system('rm '+rasPath)


        accArray = np.add(accArray,iacc)
    return 



if __name__ == '__main__':

    parser=argparse.ArgumentParser(prog='roa_accs.py')
    parser.add_argument('--date',type=str,default=testDate,help='date of T0 file to process accumulations up to (YYYYMMDDhhmm)')
    parser.add_argument('--dataDir',type=str,default=dataDir,help='directory to process ROA files from')
    parser.add_argument('--tmpDir',type=str,default=tmpDir,help='directory to hold temporary files')
    parser.add_argument('--geotiffDir',type=str,default=geotiffDir,help='directory to save outputs geoTiffs')
    args=parser.parse_args()
    tmpDir = args.tmpDir
    dataDir= args.dataDir
    geotiffDir = args.geotiffDir
    tnow = args.date

    if not os.path.exists(dataDir):
            print("dataDir not found: "+dataDir)
            sys.exit(91)
    if not os.path.exists(tmpDir):
            print("tmpDir not found: "+tmpDir)
            sys.exit(91)           
    if not os.path.exists(geotiffDir):
            print("geotiffDir not found: "+geotiffDir)
            sys.exit(91)

    # get most recent files
    new_files = []
    cronFreq=20
    t0 = datetime.datetime.today()

    total_files=glob.glob(os.path.join(dataDir,str(t0.year),str(t0.month).zfill(2),'MSG3*nc'))
    for f in total_files:
        modTimesinceEpoc = os.path.getmtime(f)
        modificationTime = datetime.datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
        if modificationTime > datetime.datetime.today()-datetime.timedelta(minutes=cronFreq):
            # now check to see if already processed?
            idate = f.split('/')[-1].split('-')[0][4:]
            itime = f.split('/')[-1].split('-')[1][1:]
            tiffPath = os.path.join(geotiffDir,idate[:8],'rainoverAfrica_SSA_'+idate+itime+'_acc72h_3857.tif')
            if not os.path.exists(tiffPath): 
                
#               only include if not already processed
                new_files.append(idate+itime)
    if len(new_files)==0:
        print("No new files to process")
    for rundate in new_files:
        getAccs(rundate,accPeriods,dataDir,tmpDir,geotiffDir)