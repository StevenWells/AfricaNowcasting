#!/usr/bin/env python

#import matplotlib
#matplotlib.use('TkAgg')

# V2: uses different method for calculating daily mean. Aligns with cmt outputs. 

import h5py
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import datetime
import time, glob, sys
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import argparse
import os.path

# 
import array

from netCDF4 import Dataset
from datetime import datetime, timedelta
import calendar 

ancdir = "/mnt/prj/swift/SEVIRI_LST/Ancillary/"
file_lon = ancdir+"hdf5_lsasaf_msg_lon_msg-disk_4bytesprecision"
file_lat = ancdir+"hdf5_lsasaf_msg_lat_msg-disk_4bytesprecision"


dmean_method = 2
# 1 is original from v1 file 

# calculate local time zone information
local_tz = True

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def read_var(filename, varname, print_attrs=False):
   global offset,scale
   if os.path.isfile(filename):
     fid = h5py.File(filename, "r")
     var = fid[varname]

     mdi = var.attrs["MISSING_VALUE"]
     scale = var.attrs["SCALING_FACTOR"]

     var_m = np.ma.masked_equal(var, mdi)
     var_m = var_m/scale 

     return var_m, fid

def read_latlon(filename, varname, print_attrs=False):
   global offset,scale
   if os.path.isfile(filename):
     fid = Dataset(filename, "r")
     var = fid.variables[varname]


     return var, fid


def read_var_t01(filename, varname, print_attrs=False):
    fid = h5py.File(filename, "r")
    var = fid[varname]

    mdi = var.attrs["MISSING_VALUE"]
    scale = var.attrs["SCALING_FACTOR"]

    var_m = np.ma.masked_equal(var, mdi)
    var_m = (var_m) / scale

    return var_m, fid

def read_var_hist(filename, var1, var2, var3, print_attrs=False):
    fid   = h5py.File(filename, "r")
    v1    = fid[var1]
    v2    = fid[var2]
    v3    = fid[var3]

    scale1, offset1 = v1.attrs["SCALING_FACTOR"], v1.attrs["OFFSET"]
    var1 = (v1 - offset1) / scale1
    
    scale2, offset2 = v2.attrs["SCALING_FACTOR"], v2.attrs["OFFSET"]
    var2 = (v2 - offset2) / scale2

    scale3, offset3 = v3.attrs["SCALING_FACTOR"], v3.attrs["OFFSET"]
    var3 = (v3 - offset3) / scale3

    return var1, var2, var3, fid

def read_var_hist_tz(GMTdate,lonGrid,dbDir,var1,var2,var3,source='clim'):
    # get the GMT image

    if source=='clim':
        filename = datetime.strftime(GMTdate,"%m%d_%H%M")
    elif source=='stats':
        tminus15 = datetime.strftime(GMTdate - timedelta(minutes=15),"%H%M")
        tnow = datetime.strftime(GMTdate,"%H%M")
        filename = datetime.strftime(GMTdate,"%m%d_")+tnow+'-'+tminus15
    dbFile = dbDir+filename
	# initialise local arrays
    lsta_av_itz_var1=lsta_av_itz_var2=lsta_av_itz_var3=np.ones(lonGrid.shape)*-999   # to hold individual time zone data
    localGrid_var1 = localGrid_var2 =localGrid_var3 =np.ones(lonGrid.shape)*-999   # final grid of combined time zones
    try:
        fid = h5py.File(dbFile, "r")
        lsta_av_itz_var1= (fid[var1][...].astype(float)- fid[var1].attrs["OFFSET"])/fid[var1].attrs["SCALING_FACTOR"]
        lsta_av_itz_var2= (fid[var2][...].astype(float)- fid[var2].attrs["OFFSET"])/fid[var2].attrs["SCALING_FACTOR"]
        lsta_av_itz_var3= (fid[var3][...].astype(float))#- fid[var3].attrs["OFFSET"])/fid[var3].attrs["SCALING_FACTOR"]
      
    except Exception as e: # if missing file, set as missing
        print(e)
        lsta_av_itz_var1=lsta_av_itz_var2=lsta_av_itz_var3 = np.ones(lonGrid.shape)*-999

    # only use this image where TZ allows (this convers full mSG grid
    #lonList = [-7.5 - x*15 for x in range(5)][::-1] + [7.5 + x*15 for x in range(6)]
    #tzShift = [x for x in range(-4,7)]

    # finer strips
    lonList = [-1.875 - x* 3.75 for x in range(6)][::-1] + [1.875 + x*3.75 for x in range (16)]
    tzShift = [0- x*15 for x in range(6)][::-1] + [x*15 for x in range(1,16)]
    #print(lonList)
    #print(tzShift)
    for itz in range(len(lonList)-1):
        localDate = GMTdate + timedelta(minutes=tzShift[itz])
        #print([localDate,lonList[itz],lonList[itz+1]])
        
        if (localDate.hour < 7) or (localDate> datetime(localDate.year,localDate.month,localDate.day,17,0) ):                 
            continue 
        else:
            localGrid_var1 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var1,localGrid_var1)
            localGrid_var2 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var2,localGrid_var2)
            localGrid_var3 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var3,localGrid_var3)
    # 
    #plt.imshow(localGrid_var1)
    #plt.show()
    return 	localGrid_var1,localGrid_var2,localGrid_var3


def read_var_hist_tz_old(GMTdate,lonGrid,dbDir,var1,var2,var3,source='clim'):
    """
    For a given date, create the local time LST map from the relevant time window files (representing the different time zones)
    INPUTS:
    GMTdate  datetime object of the date in GMT required to extract LST climatology for
    lonGrid  grid of longitudes for the image
    dbDir    location of the climatology database
    var<i>   Name of the variables to extract from the NetCDF files
    source   'clim' for climatology or 'stats' for statistics
    RETURNS:
    localGrid_var<i> map with local time zone information for variable var<i>
    
    """
    # set up time zone bands
    # 1 hour shift for every 15 degrees on disc
    lonList = [-7.5 - x*15 for x in range(5)][::-1] + [7.5 + x*15 for x in range(5)]
    tzShift = [x for x in range(-4,6)]

    
    #print(lonList)
    #print(tzShift)




	# initialise local arrays
    lsta_av_itz_var1=lsta_av_itz_var2=lsta_av_itz_var3=np.ones(lonGrid.shape)*-999  # to hold individual time zone data
    localGrid_var1 = localGrid_var2 =localGrid_var3 =np.ones(lonGrid.shape)*-999   # final grid of combined time zones
    #prev_lst_av_var1 = prev_lst_av_var2 = prev_lst_av_var3 = np.ones(lonGrid.shape)*-999 # backup grid in case missing data (likely not to be used)
    
    # loop over time zones and read appropriate time file
    for itz in range(len(lonList)-1):		
        # get file
        localDate = GMTdate + timedelta(hours=tzShift[itz])
        if source=='clim':
            filename = datetime.strftime(localDate,"%m%d_%H%M")
        elif source=='stats':
            tminus15 = datetime.strftime(localDate - timedelta(minutes=15),"%H%M")
            tnow = datetime.strftime(localDate,"%H%M")
            filename = datetime.strftime(localDate,"%m%d_")+tnow+'-'+tminus15


        dbFile = dbDir+filename
        # exclude if local time outside 0700 - 1700
        if (localDate.hour < 7) or (localDate> datetime(localDate.year,localDate.month,localDate.day,17,0) ):                 
            continue
        else:   
            #print([filename,tzShift[itz]])     
            try:
                fid = h5py.File(dbFile, "r")
                #prev_lst_av_var1,prev_lst_av_var2,prev_lst_av_var3   = lsta_av_itz_var1,lsta_av_itz_var2,lsta_av_itz_var3
                lsta_av_itz_var1= (fid[var1][...].astype(float)- fid[var1].attrs["OFFSET"])/fid[var1].attrs["SCALING_FACTOR"]
                lsta_av_itz_var2= (fid[var2][...].astype(float)- fid[var2].attrs["OFFSET"])/fid[var2].attrs["SCALING_FACTOR"]
                lsta_av_itz_var3= (fid[var3][...].astype(float))#- fid[var3].attrs["OFFSET"])/fid[var3].attrs["SCALING_FACTOR"]
                

            except Exception as e: # if missing file, set as missing
                print(e)
                lsta_av_itz_var1=lsta_av_itz_var2=lsta_av_itz_var3 = np.ones(lonGrid.shape)*-999
                #lsta_av_itz_var1,lsta_av_itz_var2,lsta_av_itz_var3 = prev_lst_av_var1,prev_lst_av_var2,prev_lst_av_var3
                #prev_lst_av_var1,prev_lst_av_var2,prev_lst_av_var3 = lsta_av_itz_var1,lsta_av_itz_var2,lsta_av_itz_var3

                    # combine 
        #print([lonGrid.shape,lsta_av_itz_var1.shape,localGrid_var1.shape])
        localGrid_var1 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var1,localGrid_var1)
        localGrid_var2 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var2,localGrid_var2)
        localGrid_var3 = np.where((lonGrid>lonList[itz]) & (lonGrid<=lonList[itz+1]),lsta_av_itz_var3,localGrid_var3)
    


    # returns numpy array of local grid
    return 	localGrid_var1,localGrid_var2,localGrid_var3







def subset_var(var, lon, lat, bbox):
    """Returns subsections of the input arrays.  When the input arrays are on
    curvilinear grids (e.g., SEVIRI) it's not possible to return a rectangular
    array subsection that follows lon/lat bounds exactly.  Instead this returns
    the var[col0:col1, row0:row1] subsection that contains all pixels that are
    within the requested region, which is likely to also include pixels outside
    of the requested lon/lat region.

    Subsection given by bbox = (lon0, lon1, lat0, lat1).
    """

    mask = ((bbox[0] <= lon) & (lon < bbox[1]) &
            (bbox[2] <= lat) & (lat < bbox[3]))
    c, r = np.where(mask.filled(False))
    cs, rs = slice(min(c), max(c)+1), slice(min(r), max(r))

    return var[cs, rs], lon[cs, rs], lat[cs, rs]



# CODE STARTS HERE
rt_feedRoot = '/mnt/scratch/semval1/SEVIRI_LST/data/'
tmpRoot = '/mnt/scratch/stewells/SEVIRI_LST/data/tmp/'
# get todays date
dnow= datetime.now()
rtDir = os.path.join(rt_feedRoot,str(dnow.year)+str(dnow.month).zfill(2))
tmpDir = os.path.join(tmpRoot,str(dnow.year)+str(dnow.month).zfill(2))
currentFiles =glob.glob(tmpDir+'/HDF5_LSASAF_MSG_LST*_'+dnow.strftime('%Y%m%d')+'*')
liveFiles  = glob.glob(rtDir+'/HDF5_LSASAF_MSG_LST*_'+dnow.strftime('%Y%m%d')+'*')
newFnames = sorted(list(set([x.split('/')[-1] for x in liveFiles]).difference([x.split('/')[-1] for x in currentFiles])))
newFiles = [os.path.join(rtDir,x) for x in newFnames]



### Read latitude longitude file 
lon_m, fid = read_var(file_lon, "LON")
lat_m, fid = read_var(file_lat, "LAT")

#crop to africa domain
bbox = (-20., 55., -40., 40.)
mask = ((bbox[0] <= lon_m) & (lon_m < bbox[1]) &
    	(bbox[2] <= lat_m) & (lat_m < bbox[3]))

row, col = np.where(mask.filled(False))   
rows, cols = slice(min(row), max(row)+1), slice(min(col), max(col))
lats=lat_m[rows, cols]
lons=lon_m[rows, cols]

updatedMean = True if len(newFiles)>0 else False

for file in newFiles: # might be .bz so need extracting
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    os.system('cp '+file+' '+tmpDir+'/')


for file in newFiles: 
    # get file time
    fileHour= file.split('/')[-1].split('_')[-1][8:10]
    fileMins = file.split('/')[-1].split('_')[-1][10:12]
    
    print("Processing "+file.split('/')[-1])
    year  = dnow.year
    smonth = str(dnow.month).zfill(2)
    month = int(smonth)
    sday   = str(dnow.day).zfill(2)
    day = int(sday)
    hours = fileHour
    mins = fileMins
    
    date0 = datetime(year,month,day,int(hours),int(mins),00) # new file
    date1= date0-timedelta(minutes=15) # reprocess the earlier one sicne theeres now a tafter file, and do new one
    date2= date0+timedelta(minutes=15)
    #date2=datetime(year,month,day,hours,30,00)
    for d in datetime_range(date1, date2, timedelta(minutes=15)):
        print(d)
        mn = d.strftime('%m')
        tnow =  d.strftime('%H%M') 
        tbef1 =  (d - timedelta(minutes=15)).strftime('%H%M')
        tbef2 =  (d - timedelta(minutes=30)).strftime('%H%M')
        taft1 =  (d + timedelta(minutes=15)).strftime('%H%M')
        days = d.strftime('%m%d')
        outday = d.strftime('%Y%m%d')

        strings = d.strftime('%Y%m')
        now = d.strftime('%Y%m%d%H%M')
        now_15 = (d - timedelta(minutes=15)).strftime('%Y%m%d%H%M')
        now_30 = (d - timedelta(minutes=30)).strftime('%Y%m%d%H%M')
        now_45 = (d - timedelta(minutes=45)).strftime('%Y%m%d%H%M')
        nowp15 = (d + timedelta(minutes=15)).strftime('%Y%m%d%H%M')
   #     print(strings + '  and ' + now + ' and ' + now_15)
   #     print(mn, tnow, tbef1, days)
 


        datadir= '/mnt/scratch/semval1/SEVIRI_LST/data/'+strings
        tmpdir =  '/mnt/scratch/stewells/SEVIRI_LST/data/tmp/'+strings
        archivedir = '/mnt/prj/swift/SEVIRI_LST/lsta_ssa/nrt/'+strings

        hist_stat="/mnt/prj/swift/SEVIRI_LST/SEVIRI_LST_2004-2022/historic_stats_adj_3back1forward/"+mn+"/HDF5_LSASAF_MSG_LST_MSG-Disk_HistStats_"
        hist_clim="/mnt/prj/swift/SEVIRI_LST/SEVIRI_LST_2004-2022/historic_clim/"+mn+"/HDF5_LSASAF_MSG_LST_MSG-Disk_HistStats_" 
 
   #     print (datadir)

        if not os.path.exists(tmpdir):
             os.makedirs(tmpdir)
        bzProc=False
        if bzProc: 
            file1 = datadir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now+".bz2"
            file2 = datadir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now_15+".bz2"
            file3 = datadir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now_30+".bz2"
            file4 = datadir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+nowp15+".bz2"

            file_now = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now+".bz2"
            file_pre1 = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now_15+".bz2"
            file_pre2 = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+now_30+".bz2"
            file_aft = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_"+nowp15+".bz2"
            #print([file_now,os.path.exists(file_now)])
            if not os.path.exists(file_now):
               os.system('cp ' + file1 + " " +tmpdir)
            if not os.path.exists(file_pre1):
               os.system('cp ' + file2 + " " +tmpdir)
            if not os.path.exists(file_pre2):
               os.system('cp ' + file3 + " " +tmpdir)
            if not os.path.exists(file_aft):
               os.system('cp ' + file4 + " " +tmpdir)
 
 
            if os.path.exists(file_now):
               os.system('bzip2 -d ' + file_now)
            if os.path.exists(file_pre1):
               os.system('bzip2 -d ' + file_pre1)
            if os.path.exists(file_pre2):
               os.system('bzip2 -d ' + file_pre2)
            if os.path.exists(file_aft):
               os.system('bzip2 -d ' + file_aft)


        data_now = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_" + now
        data_pre1 = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_" + now_15
        data_pre2 = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_" + now_30
        data_aft1 = tmpdir + "/HDF5_LSASAF_MSG_LST_MSG-Disk_" + nowp15
        #dataanom = "/scratch/semval1/SEVIRI_LST_2020-2021/data_anom_withmask_1700/"+strings
        dataanom = "/mnt/scratch/stewells/SEVIRI_LST/data_anom_withmask_1700/"+strings
    #    print(data_now)
    #    print(data_pre1)
    #    print(data_aft1)
    #    print(file_lon)
    #    print(file_lat)
    
        if not os.path.exists(dataanom):
             os.makedirs(dataanom)




### Read in the Historic climatology file 

        if local_tz:            
            clim, std, count = read_var_hist_tz(d,lons,hist_stat,"Mean", "Stdev", "count",source='stats') # stats - think need locals
            clim_hist, std_hist, count_hist = read_var_hist_tz(d,lons,hist_clim,"Mean", "Stdev", "count")

        else:
            file_hist_clim = hist_clim+days+"_" +tnow 
            if os.path.isfile(file_hist_clim):
                clim_hist, std_hist, count_hist, fid  = read_var_hist(file_hist_clim, "Mean", "Stdev", "count")
    
            # non local tz version
            file_hist = hist_stat+days+"_" +tnow+"-"+tbef1 
            if os.path.isfile(file_hist):
                clim, std, count, fid  = read_var_hist(file_hist, "Mean", "Stdev", "count")

        



    # Extract a subsection of the full SEVIRI disc.
        #bbox = (-20., 20., 0., 20.)
#####  According to Seonaid's cloud mask,the following should be applied:
###    
#        mask = np.ma.masked_where( np.isnan(lst_t01) | np.isnan(lst_t02) | np.isnan(lst_t03) | np.isnan(lstpt01) | (count.data < 50) | (diff1 < (clim-2*std)) | (diff2 < (clim-2*std)) | (diff3 < (clim-2*std)) | (diff4 < (clim-2*std)), lst_t0 )
#        mask = np.ma.masked_where((np.isnan(lst_t01) | np.isnan(lst_t02)), lst_t0 ) 
###
###################  Applying the above mask now: 



        if os.path.isfile(data_now) :
           
            lst_t0, fid = read_var(data_now, "LST")
            lst_in_t0 , lon_ss, lat_ss = subset_var(lst_t0, lon_m, lat_m, bbox)
            mask = np.ma.masked_where( count < 50 , lst_in_t0 )
# have shifted these in, since they all rely on data_now existing
            if os.path.isfile(data_pre1) :
               lst_t01, fid = read_var(data_pre1, "LST")
               lst_in_t01 , lon_ss, lat_ss = subset_var(lst_t01, lon_m, lat_m, bbox)
               diff1 = lst_in_t0 - lst_in_t01
               mask = np.ma.masked_where( np.isnan(lst_in_t01) | (diff1 < (clim-2*std)) , mask )

            if os.path.isfile(data_pre2):
               lst_t02, fid = read_var(data_pre2, "LST")
               lst_in_t02 , lon_ss, lat_ss = subset_var(lst_t02, lon_m, lat_m, bbox)
               mask = np.ma.masked_where( np.isnan(lst_in_t02) , mask )

            if os.path.isfile(data_aft1):
               lstpt01, fid = read_var(data_aft1, "LST")
               lst_in_pt01 , lon_ss, lat_ss = subset_var (lstpt01, lon_m, lat_m, bbox)
               diff2 = lst_in_pt01 - lst_in_t0
               mask = np.ma.masked_where( np.isnan(lst_in_pt01) | (diff2 < (clim-2*std)) , mask )


            anom = mask - clim_hist
    #        anom [ anom.data > 100 ] = np.nan
            print (anom.shape)
     
    ###   def write_netcdf_data(file_in,anom):
            fout1 = dataanom+"/LSASAF_lst_anom_wrt_histclim_withmask_"+now+".nc"

            print(fout1)
            dataset= Dataset(fout1,'w')
           # x= dataset.createDimension('x',1436)
           # y= dataset.createDimension('y',714)
            x= dataset.createDimension('x',2326)
            y= dataset.createDimension('y',2599)
            lsta= dataset.createVariable('lsta',np.float32,('y','x'))

            lsta[:,:]=anom
            lsta[:,:]=np.ma.masked_where(lst_in_t0 == 0, anom)
            dataset.close()

        fid.close()

if updatedMean: # some new files were processed so recalculate the daily mean
    print("Recalculating daily mean")
    lsum = np.zeros(lons.shape)
    nsum = np.zeros(lons.shape)    
    #alltodayFiles = sorted(glob.glob("/scratch/stewells/SEVIRI_LST/data_anom_withmask_1700/"+str(year)+smonth+"/LSASAF_lst_anom*_"+str(year)+smonth+sday+"*.nc"))
    #startHr = int(alltodayFiles[0].split('/')[-1].split('_')[-1][8:10])
    #startMin =int(alltodayFiles[0].split('/')[-1].split('_')[-1][10:12])
    #endHr = int(alltodayFiles[-1].split('/')[-1].split('_')[-1][8:10])
    #endMin = int(alltodayFiles[-1].split('/')[-1].split('_')[-1][10:12])
    #date1=datetime(year,month,day,3,0,00)
    #date2=datetime(year,month,day,18,00,00)
    #date1=datetime(year,month,day,startHr,startMin,00)
    #date2=datetime(year,month,day,endHr,endMin,00)+timedelta(minutes=15)
    date1= datetime(year,month,day,0,0,00)
    date2 = datetime(year,month,day,22,30,00)
    allfiles= {}
    newvar_a = []

    # load in all the files
    for idate,d in enumerate(datetime_range(date1, date2, timedelta(minutes=15))):
            mn = d.strftime('%m')
            tnow =  d.strftime('%H%M') 
            tbef =  (d - timedelta(minutes=15)).strftime('%H%M')
            days = d.strftime('%m%d')
            outday = d.strftime('%Y%m%d')
            strings = d.strftime('%Y%m')
            now = d.strftime('%Y%m%d%H%M')
            now_15 = (d - timedelta(minutes=15)).strftime('%Y%m%d%H%M')
            #print(tnow)
            #print(strings + '  and ' + now + ' and ' + now_15)
            #print(mn, tnow, tbef, days)
            dataanom = "/mnt/scratch/stewells/SEVIRI_LST/data_anom_withmask_1700/"+strings
            #dataanom = "/prj/swift/SEVIRI_LST/SEVIRI_LST_2004-2022/sample_output/full_local/data_anom_withmask_1700/"+strings
            dmeandir = "/mnt/scratch/stewells/SEVIRI_LST/data_anom_withmask_1700/daymean/"+strings
            dmeandir='/mnt/prj/swift/SEVIRI_LST/lsta_ssa/nrt/'+strings
            fin = dataanom+"/LSASAF_lst_anom_wrt_histclim_withmask_"+now+".nc"
            #print(fin)
            if os.path.exists(fin):
              print(fin)
              fid = Dataset(fin, "r")
              var_a = fid.variables['lsta'][:,:]
			  
              hour=(idate)/4.
              lhour=lons/360.*24.+hour
              lsum = np.where((lhour>=7) & (lhour<=17) & (var_a < 1.0*10**30),lsum+var_a,lsum)
              nsum = np.where((lhour>=7) & (lhour<=17) & (var_a < 1.0*10**30),nsum+1,nsum)
              #lsum[lhour>=7 and lhour<=17 and var_a< 10^30]+=lsta
    
 
			  #allfiles[tnow]=var_a
            else:
              continue  
    # allfiles is a dictionary of the files linked to UTC time they apply to

    # loop over times in the day (would normally be the same as above but testing here for 1200
    #date1=datetime(year,month,day,7,0,00)
    #date2=datetime(year,month,day,17,15,00)
    #newvar_a = []
    #for d in datetime_range(date1, date2, timedelta(minutes=15)): #loop over local time 
    #          print(d)
    ##         make up a local timezone version 
    #          var_local =np.ones(var_a.shape)*9.96921e+36
    #          #tmp = np.ones(var_a.shape)*9.96921e+36
    #          # mask by longitude (ie only want the bit for this image corresponding to 12Local time          
    #          lonList = [-1.875 - x* 3.75 for x in range(6)][::-1] + [1.875 + x*3.75 for x in range (16)]
    #          tzShift = [0- x*15 for x in range(6)][::-1] + [x*15 for x in range(1,16)]
    #          for itz in range(len(lonList)-1):
    #            utcDate = d - timedelta(minutes=tzShift[itz])
    #           # print([lonList[itz],utcDate])
    #            if utcDate.strftime('%H%M') in allfiles.keys():
    #                 var_local = np.where((lons>lonList[itz]) & (lons<=lonList[itz+1]),allfiles[utcDate.strftime('%H%M')],var_local)
    #          newvar_a.append(var_local)
    #          #newvar_a.append(var_a)
    #          #plt.imshow(var_local)
    #          #plt.show()
    #
    #   Apply missing value mask for newvar
    #   This masking would give '0' as the resulting FillValue in the 
    #   DailyMean output file! 
    ###########################################
    #vara_msk= np.ma.masked_equal(np.dstack(newvar_a), 9.96921e+36) 
    #vara_msk= np.ma.masked_greater_equal(np.dstack(newvar_a), 9.96921e+34) 
    #print (vara_msk.shape)
    #for ij in range(vara_msk.shape[2]):
        
    ##    plt.imshow(vara_msk[:,:,ij])
    #    plt.show()
    ################################################
    ###  Due to some reason, the resulting file retains the masked value as '0'
    ###  as in the raw data.  So this has to be maksed again!
    ################################################
    #anom = np.ma.mean(vara_msk, axis=2)

    anom = np.divide(lsum,nsum,out=np.zeros_like(lsum),where=nsum!=0)
    anom[anom == 0.] =  9.96921e+36

    if not os.path.exists(dmeandir):
           os.makedirs(dmeandir)

    fout = dmeandir+"/LSASAF_lst_anom_Daymean_withmask_withHistClim_"+outday+"_1700.nc"
    print (fout)
    dayclim = h5py.File(fout, "w" )

     
    var_anom, lon_dom, lat_dom = subset_var(anom, lon_m, lat_m, bbox)
    lst_anom = dayclim.create_dataset('lst_anom_dailymean', data=anom)
    lst_anom.attrs.create('_FillValue', data=9.96921e+36) 
    lst_anom.attrs.create("units", data= "Degrees Celcius")


    dayclim.close
    fid.close
	# update archive

    #if not os.path.exists(archivedir):
    #     os.makedirs(archivedir)
      
    #os.system('cp '+fout+' '+archivedir+'/')


