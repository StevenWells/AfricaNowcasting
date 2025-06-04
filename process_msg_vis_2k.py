import rioxarray
import xarray as xr
import numpy as np
import glob,os,datetime,sys,time
from pyproj import CRS
import u_interpolate as inter
import rasterio
from osgeo import gdal

# Archive of the 10 minute reprocessed files
archiveDir = '/mnt/prj/nflics/data/vis_2k'
# liveDir
liveDir = '/mnt/scratch/cmt/FCI_tif_NRT/'
# portal output directory
portalDir = '/mnt/HYDROLOGY_stewells/geotiff/ssa_visible_2k'
reprocess_file = False
origEPSG = '4326'
newEPSG = '3857'
versionstr = '4.0.0'




# get the most recent files from the liveDir
files = glob.glob(liveDir+'*.tif')
if len(files)==0:
    print("No files to process")
    sys.exit(0)

files_to_process = []
for f in files:
    
    modTimesinceEpoc = os.path.getmtime(f)
    modificationTime = datetime.datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
    if modificationTime > datetime.datetime.today()-datetime.timedelta(minutes=10): #probably just one(the latest)
        # now check if been processed

        idate =datetime.datetime.strptime(os.path.basename(f).split('_')[1][:8]+os.path.basename(f).split('_')[1][9:13],"%Y%m%d%H%M")
        archDir = os.path.join(archiveDir,str(idate.year),str(idate.month).zfill(2),str(idate.day).zfill(2))

        if (not os.path.exists(os.path.join(archDir,'VIS_2k_'+os.path.basename(f).split('_')[1]+'_'+origEPSG+'.tif'))) or (os.path.exists(os.path.join(archDir,'VIS_2k_'+os.path.basename(f).split('_')[1]+'_'+origEPSG+'.tif')) and reprocess_file):
        
            files_to_process.append(f)


#read reprojection information
with open("/mnt/prj/nflics/MTG_vis_testbed/crs_proj_geostationary_MTG.txt", "r") as f:
    crs = CRS.from_proj4(f.read())


# process file function
def process_file(rFile):   
    da_box = read_geotiff(rFile)
    sdate = datetime.datetime.strptime(os.path.basename(rFile).split('_')[1][:8]+os.path.basename(rFile).split('_')[1][9:13],"%Y%m%d%H%M")
    #############
    xmax, xmin = np.max(da_box.x), np.min(da_box.x)
    ymax, ymin = np.max(da_box.y), np.min(da_box.y)
    # create new coordinates going from ~500m to ~2km by dividing by 4
    new_x = np.linspace(xmin.values, xmax.values, int(len(da_box.x)/4))
    new_y = np.linspace(ymin.values, ymax.values, int(len(da_box.y)/4))
    # usual interpolation routine, input is not irregular
    # THIS SHOULD BE IMPLEMENTED IN WORKFLOW TO BE RUN ONLY ONCE AND "ind, weights, shape" OUTPUT SAVED FOR REUSE
    if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_data_vis2k.npz'):
        print("reading npz weights")
        weightdata = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_data_vis2k.npz')
        inds = weightdata['inds']
        weights= weightdata['weights']
        shape=tuple(weightdata['shape'])                    
    else: # need to make it   
        print("creating weights")
        inds, weights, shape = inter.interpolation_weights(da_box.x.values, da_box.y.values, new_x, new_y, irregular_1d=False)
        #inds, weights, new_shape=uinterp.interpolation_weights(lons_mid[np.isfinite(lons_mid)], lats_mid[np.isfinite(lats_mid)],blobs_lons, blobs_lats, irregular_1d=True)
        np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_data_vis2k.npz',inds=inds,weights=weights,shape=np.array(shape))
    ##################
    # read reprojected data as numpy array
    data_2km = inter.interpolate_data(da_box.values, inds, weights, shape)
    # convert to data array for subsetting
    da_2km = xr.DataArray(data_2km,
        coords={"y": new_y, "x": new_x},
        dims=["y", "x"],
    )
    # subsetting to remove undefined edges for portal plotting
    da_2km = da_2km.sel(y=slice(-33,-10), x=slice(15,35))
    ######## da_2km should be saved in format needed for portal hosting
    nbands=1
    #rasFile= '/home/stewells/AfricaNowcasting/tmp/vis2km_4326.tif'
    
    dat_type = str(da_2km.dtype)
    archiveDirNow = os.path.join(archiveDir,str(sdate.year),str(sdate.month).zfill(2),str(sdate.day).zfill(2))
    os.makedirs(archiveDirNow,exist_ok=True)
    outFile = os.path.join(archiveDirNow,'VIS_2k_'+os.path.basename(rFile).split('_')[1]+'_'+origEPSG+'.tif')

    #transform = rasterio.transform.from_origin(-27,27,0.026949456,0.026949456)
    transform = rasterio.transform.from_origin(15,-10,0.02794187,0.02794187)
    rasImage = rasterio.open(outFile,'w',driver='GTiff',
                            height=da_2km.shape[0],width=da_2km.shape[1],
                            count=nbands,dtype=dat_type,
                            crs = 'EPSG:'+str(origEPSG),
                            transform = transform)
    rasImage.write(np.flipud(da_2km[:]),1)
    rasImage.close()
    # add metadata version
    gdaledit= '/users/hymod/stewells/miniconda2/envs/py37/bin/gdal_edit.py'
    os.system("gdal_edit.py -mo \"xmp_Version_Version="+versionstr+"\" "+outFile)

    #reprojFile = os.path.join(archiveDirNow,outFile)
    #ds = gdal.Warp(reprojFile, rasFile, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff',creationOptions=["COMPRESS=LZW"])
    #ds = None  
    #os.system('rm '+rasFile)

def fileFromTime(itime,idir,type='raw'):
    if type=='raw':
        itimeN = itime+datetime.timedelta(minutes=3)
        rfiles = glob.glob(os.path.join(idir,itime.strftime('FCIL1HRFI_%Y%m%dT%H%M*Z_')+itimeN.strftime('%Y%m%dT%H%M*Z_epct_*_FC.tif')))
        print("RAW")
        print(itime.strftime('FCIL1HRFI_%Y%m%dT%H%M*Z_')+itimeN.strftime('%Y%m%dT%H%M*Z_epct_*_FC.tif'))
        if len(rfiles)>0:
            return rfiles[0]
        else:
            return os.path.join(idir,'MISSING.dat') # dummy filename to indicate missing
    elif type=='processed':
        
        archiveDir = os.path.join(idir,str(itime.year),str(itime.month).zfill(2),str(itime.day).zfill(2))
        pfiles = glob.glob(os.path.join(archiveDir,itime.strftime('VIS_2k_%Y%m%dT%H%M*Z_'+origEPSG+'.tif')))
        if len(pfiles)>0:
            return pfiles[0]
        else:
            return os.path.join(idir,"MISSING_PROCESSES.dat")

        return os.path.join(archiveDir,itime.strftime('VIS_2k_%Y%m%dT%H%M*Z_'+origEPSG+'.tif'))

    

# read tif file function
def read_geotiff(file):
    data = rioxarray.open_rasterio(file)
    data_rio = data.rio.write_crs(crs)
    data_reprojected = data_rio.rio.reproject("EPSG:4326")

    data_reprojected = data_reprojected.sel(x=slice(13,36), y=slice(-9, -35))
    data_reprojected = data_reprojected.where(data_reprojected.values<60000, other=0)

    #normalise data between 0 and 1 for visualisation
    data_reprojected = (data_reprojected - np.min(data_reprojected)) / (np.max(data_reprojected) - np.min(data_reprojected))

    return data_reprojected.squeeze()

def create_portal_file(image,vtime,outRoot):
    tmpFile= '/home/stewells/AfricaNowcasting/tmp/vis2km_tmp_4326.tif'
    # make a percentage
    image = image*100.0
    dat_type = str(image.dtype)
    nbands = 1
    transform = rasterio.transform.from_origin(15,-10,0.02794187,0.02794187)
    rasImage = rasterio.open(tmpFile,'w',driver='GTiff',
                            height=image.shape[0],width=image.shape[1],
                            count=nbands,dtype=dat_type,
                            crs = 'EPSG:'+str(origEPSG),
                            transform = transform)
    rasImage.write(image[:],1)
    rasImage.close()
    # add metadata version
    gdaledit= '/users/hymod/stewells/miniconda2/envs/py37/bin/gdal_edit.py'
    os.system("gdal_edit.py -mo \"xmp_Version_Version="+versionstr+"\" "+tmpFile)

    # final portal file
    outDir = os.path.join(outRoot,vtime.strftime("%Y%m%d"))
    os.makedirs(outDir,exist_ok=True)
    outFile = os.path.join(outDir,'VIS_2k_'+vtime.strftime('%Y%m%d%H%M')+'_'+newEPSG+'.tif')    
    ds = gdal.Warp(outFile, tmpFile, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff',creationOptions=["COMPRESS=LZW"])
    ds = None  
    os.system('rm '+tmpFile)


# for each  new file to process
# check if the previous two timstamp files has been processed
# if they have, carry on
# if not, check if they exist in raw
#   if yes, process and mark as there 
#   if no, mark as missing

for rFile in files_to_process:
    idate =datetime.datetime.strptime(os.path.basename(rFile).split('_')[1][:8]+os.path.basename(rFile).split('_')[1][9:13],"%Y%m%d%H%M")
    req_dates= [idate - datetime.timedelta(minutes=10*x) for x in range(3)]
    for ix,req_date in enumerate(req_dates):
        processed_file = fileFromTime(req_date,archiveDir,'processed')
        raw_file = fileFromTime(req_date,liveDir,'raw')
        print([req_date,liveDir])
        if os.path.exists(processed_file):
            print("File already processed: "+processed_file)
        else:
            if os.path.exists(raw_file):
                try:
                    print("Process the file "+raw_file)
                    process_file(raw_file)
                except:
                    print("failed to process file "+raw_file )
            else:  # now raw file either
                    print("Missing file: "+raw_file)

# STAGE 2: for each new file, update the prtal geotiffs
    quarter_hours = [0, 15, 30, 45]
    # Find the largest quarter-hour mark that is less than or equal to the given minute
    past_quarter = max(q for q in quarter_hours if q <= idate.minute)
    # 15 minute window time
    validity_time = datetime.datetime(idate.year,idate.month,idate.day,idate.hour,past_quarter)
    validity_end  = validity_time + datetime.timedelta(minutes=15)
    validity_times = [ validity_time+datetime.timedelta(minutes=1)*x for x in range(16)]
    sdate_end  = idate+ datetime.timedelta(minutes=10)
    if sdate_end > validity_end:
        next_image_time = validity_time + datetime.timedelta(minutes=15)
        print("Add initial image for the next  time stamp")
        try:
            next_image = rioxarray.open_rasterio(fileFromTime(idate,archiveDir,'processed'))[0]
            print(next_image.shape)
            create_portal_file(next_image,next_image_time,portalDir)
        except Exception as e:
            print("Couldn't add initial image for the next  time stamp: no file to process")
            print(e)
    # print(validity_end)
    # reprocess the current valirty time
    # get list of 10 min files we need
    fileListT0 = []
    for iT in req_dates:
        iT_end = iT + datetime.timedelta(minutes=10)
        rdates = [iT+ datetime.timedelta(minutes=1)*x for x in range(11)]
        if len(list(set(rdates) & set(validity_times)))>0:
            if os.path.exists(fileFromTime(iT,archiveDir,'processed')):
                print("Adding "+fileFromTime(iT,archiveDir,'processed'))
                fileListT0+=[rioxarray.open_rasterio(fileFromTime(iT,archiveDir,'processed'))]
            else:
                print("Missing file: "+str(fileFromTime(iT,archiveDir,'processed')))
    if len(fileListT0)>0:
        combined_image = np.mean(fileListT0,axis=0)[0]

        create_portal_file(combined_image,validity_time,portalDir)
    else:
        print("No files to process")

        



