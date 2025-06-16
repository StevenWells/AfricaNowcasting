import xarray as xr
import tensorflow as tf
import tensorflow.keras.optimizers
import tensorflow.keras.metrics
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import date, datetime, timedelta
import netCDF4 as nc
from scipy.interpolate import griddata
#from ndays import numOfDays

import glob
import calendar
import os,sys
import rasterio, time
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds
import argparse

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
import u_interpolate as uinterp
from osgeo import gdal

# Define domain and time period
# date is supposed to change automatically


# core path (most recent data)
dataDir ="/mnt/scratch/stewells/MSG_NRT/cut/"
# Geotiff dir on the SAN for portal
SANdir = '/mnt/HYDROLOGY_stewells/geotiff/ssa_nowcast_cores_unet/'
#SANdir = '/home/stewells/'
# testdate for default historical mode date  (to duplicate date used on https://github.com/JawairiaAA/WISER_testbed_2025/ )
testdate='202501172100'

archiveDir= "/mnt/prj/nflics/cnn_cores/geotiff/"
#archiveDir= "/home/stewells/arch_tmp/"

# get the command arguments
parser= argparse.ArgumentParser(description="Generate geotiffs of Nowcast cores")

# mode (default realtime)
# mode : realtime = pick most recently added file that has not already been processed
#        historical = pick a specific date TODO: allow for range of dates
parser.add_argument("--mode", choices=["realtime","historical"], default="realtime",help="Run mode (real time or historical)")
parser.add_argument("--hDate", type=str,default=testdate, help="Historical date to process (YYYYMMDDhhmm).")
# output directory, to be used if different from the SAN. Defaults to SAN
parser.add_argument("--outDir", type=str, default = SANdir, help="Directory to send outputs to (defaults to SAN)")
# load them
args = parser.parse_args()
#get variables from arguments
mode = args.mode
outRoot = args.outDir

# lead times (hours) 
leadtimes = [1,2,4,6]



# realtime: get most recently added file
if mode=='realtime':
    total_files=glob.glob(os.path.join(dataDir,'IR_108*'))
    new_dates = []
    for f in sorted(total_files):
            
            """
            modTimesinceEpoc = os.path.getmtime(f)

            modificationTime = datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
            if modificationTime > datetime.today()-timedelta(minutes=1):
                idate =''.join(os.path.basename(f).split('_')[3:5]).split('.')[0]
                # only process if not already done so (check existence of 6hr image - the last to process)
                if not os.path.exists(os.path.join(outRoot,idate[:8],'nowcast_cores_unet_'+idate[0:8]+'_'+idate[8:12]+'_'+str(6)+'hr_3857.tif')):
                    new_dates.append(idate)    
            """
            idate =''.join(os.path.basename(f).split('_')[3:5]).split('.')[0]
            new_dates.append(idate)    
    # Choose latest date only for now
    if len(new_dates)>0:
        current_date = new_dates[-1]
    else:
        print("No data to process")
        sys.exit(0)
elif mode== 'historical':
    current_date = args.hDate
    #current_date = '202501172100'

print("Processing: "+str(current_date))
#--------------------------------
current_year = current_date[0:4]
current_month = current_date[4:6]
current_day = current_date[6:8]
t = 3 #numOfDays

# cords of interest
#start_lat = -20 #5 # 5 
#end_lat = -5 #21 #10
#start_lon = 25 #-10
#end_lon = 55 #0


# coords for each region
# [start_lat,end_lat,start_lon,end_lon,latIndex,lonIndex,timelag(hrs)]
#regPars = {'ZA':{'coords':[-20,-5,25,55,1,1],'timeLag':2.1,'a':11,'b':-25,'modelFile':'ZA_Jan_Feb_trained_model_2005_to_2019.h5'},
#           'KY':{'coords':[-8,7,26.5,47,100,1000],'timeLag':4.1,'a':-33,'b':19,'modelFile':'KY_MAM_trained_model_2005_to_2019.h5'}}

regPars = {'ZA':{'coords':[-20,-5,25,55,1,1],'timeLag':4.1,'a':11,'b':-25,'modelFile':'ZA_DJF_trained_model_2005_to_2019.h5','regrid':False},
           'KY':{'coords':[-8,7,26.5,47,100,1000],'timeLag':4.1,'a':-33,'b':19,'modelFile':'KY_MAM_trained_model_2005_to_2019.h5','regrid':False},
           'SE':{'coords':[0,21,-20,1,10,1000],'timeLag':4.1,'a':-13,'b':13,'modelFile':'SE_JAS_trained_model_2005_to_2019_0p04.h5','regrid':True}}




# get native MSG grid (core)
#coords_filename = glob.glob('/mnt/prj/Africa_cloud/geoloc/*.npz')[0]  # this is /prj/Africa_cloud/geoloc/*.npz on the Linux system
coords_filename = glob.glob('/home/stewells/AfricaNowcasting/ancils/unet/lat_lon*.npz')[0]  # this is /prj/Africa_cloud/geoloc/*.npz on the Linux system

msg_latlon = np.load(coords_filename)
mlon = msg_latlon['lon']
mlat = msg_latlon['lat']

# loop was here start

def _create_mean_filter(half_num_rows, half_num_columns, num_channels):
    """Creates convolutional filter that computes mean.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels

    :param half_num_rows: Number of rows on either side of center.  This is
        (M - 1) / 2.
    :param half_num_columns: Number of columns on either side of center.  This
        is (N - 1) / 2.
    :param num_channels: Number of channels.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of filter weights.
    """

    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)

    return np.full(
        (num_rows, num_columns, num_channels, num_channels), weight,
        dtype=np.float32
    )

def FSS_loss(target_tensor, prediction_tensor):
    
    half_window_size_px=2
    use_as_loss_function=True 
    #mask_matrix
    function_name=None
    test_mode=False
    """Fractions skill score (FSS).

    M = number of rows in grid
    N = number of columns in grid

    :param half_window_size_px: Number of pixels (grid cells) in half of
        smoothing window (on either side of center).  If this argument is K, the
        window size will be (1 + 2 * K) by (1 + 2 * K).
    :param use_as_loss_function: Boolean flag.  FSS is positively oriented
        (higher is better), but if using it as loss function, we want it to be
        negatively oriented.  Thus, if `use_as_loss_function == True`, will
        return 1 - FSS.  If `use_as_loss_function == False`, will return just
        FSS.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    weight_matrix = _create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )
       
    """Computes loss (fractions skill score).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Fractions skill score.
    """

    smoothed_target_tensor = K.conv2d(
        x=target_tensor, kernel=weight_matrix,
        padding='same', strides=(1, 1), data_format='channels_last'
    )

    smoothed_prediction_tensor = K.conv2d(
        x=prediction_tensor, kernel=weight_matrix,
        padding='same', strides=(1, 1), data_format='channels_last'
    )

    actual_mse = K.mean(
        (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
    )
    reference_mse = K.mean(
        smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
    )

    if use_as_loss_function:
        return actual_mse / reference_mse

    return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name


def spatial_filter_conv(predicted_image,half_window_size_px):
    
    #half_window_size_px=2
    weight_matrix = _create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    smoothed_predicted_image = K.conv2d(
        x=predicted_image, kernel=weight_matrix,
        padding='same', strides=(1, 1), data_format='channels_last'
    )
    return smoothed_predicted_image


from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)


def merge_geotiffs(tif_files,output_file):
    # Open all datasets
    datasets = [rasterio.open(f) for f in tif_files]

    # Determine output bounds (maximum extent)
    lefts, bottoms, rights, tops = zip(*[dataset.bounds for dataset in datasets])
    #print([lefts, bottoms, rights, tops])
    min_left, min_bottom = min(lefts), min(bottoms)
    max_right, max_top = max(rights), max(tops)
    # Calculate output transform and dimensions
    resolution = datasets[0].res[0]  # Assuming uniform resolution

 
    width = int((max_right - min_left) / resolution)
    height = int((max_top - min_bottom) / resolution)
    transform = from_bounds(min_left, min_bottom, max_right, max_top, width, height)

    # Create an empty array filled with NaN
    combined_data = np.full((height, width), np.nan, dtype='float32')


    # Place each dataset in the combined data array
    for dataset in datasets:
        window = window_from_bounds(*dataset.bounds, transform=transform)
        win_height = int(window.height)
        win_width = int(window.width)
        #data = dataset.read(out_shape=(dataset.count, win_height, win_width), window=window, resampling=Resampling.nearest)
        data = dataset.read(1)
        row_off, col_off = int(window.row_off), int(window.col_off)
        
        # Identify valid data points and place them in the combined array
        valid_mask = ~np.isnan(data)
        
        combined_data[row_off:row_off + win_height+1, col_off:col_off + win_width+1] = np.where(
            valid_mask, 
            data, 
            combined_data[row_off:row_off + win_height+1, col_off:col_off + win_width+1]
        )



    # Write output file
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": height,
        "width": width,
        "transform": transform
    })
    combined_data = combined_data.reshape((1,combined_data.shape[0],combined_data.shape[1]))
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(combined_data)
    # Close datasets
    for dataset in datasets:
        dataset.close()





regional_tifs = {str(x):[] for x in leadtimes}
for region in list(regPars.keys()):
    start_lat = regPars[region]['coords'][0]
    end_lat = regPars[region]['coords'][1]
    start_lon = regPars[region]['coords'][2]
    end_lon = regPars[region]['coords'][3]

    latInd = regPars[region]['coords'][4]
    lonIndex =regPars[region]['coords'][5]
    timeLag = regPars[region]['timeLag']
    # find core indices using one file
    lat_ind = np.where((mlat[:,latInd]>=start_lat) & (mlat[:,latInd]<=end_lat))[0]
    lon_ind = np.where((mlon[lonIndex,:]>=start_lon) & (mlon[lonIndex,:]<=end_lon))[0]
    lat = mlat[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
    lon = mlon[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]

    num_frames= 3   # 
    t0= 1  #1   
    a= regPars[region]['a']
    b= regPars[region]['b']

    if region=='ZA':
        lon_sub = lon[a:,:b]
        lat_sub = lat[a:,:b]
    elif region=='SE':
        lon_sub = lon[:,:]
        lat_sub = lat[:,:]
    else:
        lon_sub = lon[:a,b:]
        lat_sub = lat[:a,b:]

    # prepare resampling for geotiff
    #dx = 0.026949456
    #dx = 0.053898912
    dx = 0.04 # make same as SE processing
    # these are from MSG
    lat_min, lat_max= np.nanmin(lat_sub),np.nanmax(lat_sub)
    lon_min, lon_max= np.nanmin(lon_sub),np.nanmax(lon_sub)
    grid_lat = np.arange(lat_min,lat_max ,dx)
    grid_lon = np.arange(lon_min,lon_max ,dx)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon,grid_lat)
    inds, weights, new_shape=uinterp.interpolation_weights(lon_sub[np.isfinite(lon_sub)], lat_sub[np.isfinite(lat_sub)],grid_lon_mesh, grid_lat_mesh, irregular_1d=True)
    #print([region,len(grid_lat),len(grid_lon),new_shape])

    # if processing done on regular grid, set up regridding
    if regPars[region]['regrid']:
        regular_lat = np.arange(regPars[region]['coords'][0], regPars[region]['coords'][1], 0.04)
        regular_lon = np.arange(regPars[region]['coords'][2], regPars[region]['coords'][3], 0.04)
        #regular_lon, regular_lat = np.meshgrid(regular_lon,regular_lat)
        inds_regular, weights_regular, shape_regular = uinterp.interpolation_weights(lon_sub, lat_sub, regular_lon, regular_lat) # save weights for continuous use - MSG interpolation on regular. 
        regridded_tir = np.zeros((t,len(regular_lat),len(regular_lon)),dtype=float) 
        print(regridded_tir.shape)
        #inds_totiff,weights_totiff,shape_totiff = uinterp.interpolation_weights(regular_lon,regular_lat,grid_lon, grid_lat)

    #cores = np.zeros((t,len(lat[:,1]),len(lon[1,:])),dtype=float) 
    tir = np.zeros((t,len(lat[:,1]),len(lon[1,:])),dtype=float)
    


    time_core = np.zeros((t)) 
    data_adapter._is_distributed_dataset = _is_distributed_dataset
    tf.config.run_functions_eagerly(True)


    ##### Define input shape
    image_height= 512
    image_width= 512   #
    num_channels= 3    #    



    # load files
    current_date_int = datetime.strptime(current_date, '%Y%m%d%H%M')
    to_date=datetime.strptime(str(current_date), '%Y%m%d%H%M')
    to_minus_1hr_date=current_date_int-timedelta(hours=1)
    to_minus_1hr_date= to_minus_1hr_date.strftime('%Y%m%d%H%M')
    to_minus_2hr_date=current_date_int-timedelta(hours=2)
    to_minus_2hr_date= to_minus_2hr_date.strftime('%Y%m%d%H%M')

    dates_of_interest = [to_minus_2hr_date,to_minus_1hr_date,str(current_date)]
    print("T0: "+str(current_date))
    #dir_name = '/mnt/prj/nflics/real_time_data/'+current_year+'/'+current_month.zfill(2)+'/'+current_day.zfill(2)+'/' if mode=='historical' else dataDir
    dir_name = '/mnt/prj/nflics/real_time_data/'+current_year+'/'+current_month.zfill(2)+'/'+current_day.zfill(2)+'/' 


    list_of_files=[]                
    for a in range(0,len(dates_of_interest),1):
        dates_of_interest_curr = dates_of_interest[a]
        list_of_files.append(dir_name+'IR_108_BT_'+dates_of_interest_curr[0:4]+dates_of_interest_curr[4:6]+dates_of_interest_curr[6:8]+'_'+dates_of_interest_curr[8:]+'_eumdat.nc')

    # check for existing t0-2 file

    if os.path.exists(list_of_files[0]) == False:
        
        list_of_files[0]=list_of_files[2]
        list_of_files[1]=list_of_files[2] 
        
        to2_date = dates_of_interest[0]
        try:
            dir_name = '/mnt/prj/nflics/real_time_data/'+current_year+'/'+to2_date[4:6]+'/'+to2_date[6:8]+'/' 
            all_file_names = sorted(glob.glob(dir_name+"IR*.nc"));  #
            latest_to2_file = all_file_names[-4*2] 
            # check time between files 
            to_2_date=latest_to2_file[-23:-15]+latest_to2_file[-14:-10]
            to_2_datetime=datetime.strptime(str(int(to_2_date)), '%Y%m%d%H%M')
            time_difference = to_date-to_2_datetime    
            if time_difference< timedelta(hours=timeLag):
                list_of_files[0]=latest_to2_file
                list_of_files[1]=all_file_names[-4]
            else:
                list_of_files[0]=list_of_files[2]
                list_of_files[1]=list_of_files[2]  
        except:
            print("Unable to find suitable replacement for T0-2. Using T0 at all three time steps")
            list_of_files[0]=list_of_files[2]
            list_of_files[1]=list_of_files[2]  

    # read in tir data
    for l in range(0,len(list_of_files),1): 
        tir_filename = list_of_files[l]
        if os.path.exists(tir_filename):
            ds = xr.open_dataset(tir_filename).squeeze() 
            tir_temp =  ds['ir108_bt'].values  #/10000
            tir[l,:,:] = tir_temp[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]   
            # regrid if required
            if regPars[region]['regrid']:
                regridded_tir[l,:,:] = uinterp.interpolate_data(tir[l,:,:], inds_regular, weights_regular, shape_regular)  # interpolation using saved weights for MSG TIR
            ds = None 
        time_core[l] = int(dates_of_interest[l])  #int(tir_filename[-15:-3])


    num_frames= 3   # 
    t0= 1  #1   
    a= regPars[region]['a']
    b= regPars[region]['b']

    # loop over lead times here
    for leadtime in leadtimes:

        print("Processing "+str(leadtime)+'hr leadtime for '+str(region))

        if regPars[region]['regrid']:
            tir_t_0 = regridded_tir[:,:a,b:]
        else:
            if region=='ZA':  # TODO: generalise this
                tir_t_0 = tir[:,a:,:b]
            else:
                tir_t_0 = tir[:,:a,b:]

        ind_tir = np.where(tir_t_0>-0.01)
        tir_t_0[ind_tir] = 0
        tir_t_0[np.isnan(tir_t_0)] = 0
        tir_t_0 = np.round(tir_t_0/-173,4)

        # NEED TO POINT TO CORRECT MODEL FILE
        modelFile= '/home/stewells/AfricaNowcasting/ancils/unet/'+str(leadtime)+'hr_using_1hr/'+regPars[region]['modelFile']
        unet_model = tf.keras.models.load_model(modelFile, compile=False,custom_objects={'loss': FSS_loss})

        unet_model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                        loss=FSS_loss,
                        metrics=[tf.keras.metrics.Accuracy()])


        #prediction_time = time_core*0
        #for i in range(0,len(time_core)):
        #    time_core_dt = datetime.strptime(dates_of_interest[i], '%Y%m%d%H%M')
        #    prediction_time_temp = (time_core_dt+timedelta(hours=leadtime)).strftime('%Y%m%d%H%M')
        #    prediction_time[i] = int(prediction_time_temp)
        
        prediction_time = int((to_date+timedelta(hours=leadtime)).strftime('%Y%m%d%H%M'))

        # Define input shape
        image_height= len(tir_t_0[1,:,1]) #lat
        image_width= len(tir_t_0[1,1,:]) #lon
       

        num_channels= 3 #  core at t0-, core at t0-1,  
        x_pred= np.zeros((1,image_height,image_width, num_channels))
        x_pred[:,:,:,0]= tir_t_0[0,:]
        x_pred[:,:,:,1]= tir_t_0[1,:]
        x_pred[:,:,:,2]= tir_t_0[2,:]

        time_of_day_pred= np.zeros((1,image_height,image_width,1))
        #time_of_day = float(str(prediction_time[-1])[-6:])/2345
        time_of_day = float(str(prediction_time)[-6:])/2345
        time_of_day_pred[:,:,:,:]=np.round(np.sin(time_of_day*math.pi),2)

        #predicted_frames= np.round(np.squeeze(unet_model.predict([x_pred,time_of_day_pred])),2)
    
        
        predicted_frames= np.round(unet_model.predict([x_pred,time_of_day_pred]),2)
        half_window_size_px = 7 if int(leadtime)>3 else 2
        filtered_image = spatial_filter_conv(predicted_frames,half_window_size_px)

        filtered_image = np.squeeze(filtered_image[0,:,:,0])


    #resample
        if not regPars[region]['regrid']: 
            #print(filtered_image.shape)
            #print(shape_totiff)
            data_interp=uinterp.interpolate_data(filtered_image, inds, weights, new_shape)
        
          #  data_interp=uinterp.interpolate_data(filtered_image, inds_totiff, weights_totiff, shape_totiff)
        else:
            data_interp = np.copy(filtered_image) # already on fixed grid
            

    # convert to probability
        data_interp*=100.
        # save geotiff
        # temporary file in original EPSG (to be deleted once converted to 3857 for portal)
        rasFile_tmp = '/mnt/HYDROLOGY_stewells/geotiff/ssa_nowcast_cores_unet/unet_'+str(leadtime)+'hr_'+region+'.tif'
        #rasFile_tmp = '../tmp/unet_'+str(leadtime)+'hr_'+region+'.tif'

        outDir = os.path.join(outRoot,current_date[0:8])
        os.makedirs(outDir,exist_ok=True)

        #print(lon_min,lat_max)
        #print(regPars[region]['coords'][2],regPars[region]['coords'][1])
        #print(len(regular_lon[b:]), len(regular_lat[:a]))
        #print(regular_lon[b],regular_lat[a])
        if not regPars[region]['regrid']: 
            
            transform = rasterio.transform.from_origin(lon_min,lat_max,dx,dx)
        else:
            #transform = rasterio.transform.from_origin(regPars[region]['coords'][2],regPars[region]['coords'][1],dx,dx)
            transform = rasterio.transform.from_origin(regular_lon[b],regular_lat[a+1],dx,dx)

        dat_type = str(data_interp.dtype)
        rasImage = rasterio.open(rasFile_tmp,'w',driver='GTiff',
                                height=data_interp.shape[0],width=data_interp.shape[1],
                                count=1,dtype=dat_type,
                                crs = "EPSG:4326",#origEPSG
                                nodata=-999.9,
                                transform = transform                           
                            )
        #for ix,Image in enumerate(data):
        rasImage.write(np.flipud(data_interp[:]),1)
        rasImage.close()
        archDir= os.path.join(archiveDir,current_date[0:8])
        os.makedirs(archDir,exist_ok=True)
        archFile =os.path.join(archDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(leadtime)+'hr_'+region+'_4326.tif')
        try:
            os.system('cp '+rasFile_tmp+' '+archFile)
            regional_tifs[str(leadtime)]+=[archFile]
        except:
            print("Failed to write to geotiff in 4326 archive directory")
        os.system('rm '+rasFile_tmp)

# For each leadtime, merge the regional geotiffs
for lt in leadtimes:
    slt = str(lt)
    files_to_merge = regional_tifs[slt]
    fullFile_tmp =     os.path.join('/mnt/HYDROLOGY_stewells/geotiff/ssa_nowcast_cores_unet/','nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(lt)+'hr_4326.tif')
    rasFile =     os.path.join(outDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(slt)+'hr_3857.tif')
    merge_geotiffs(files_to_merge,fullFile_tmp)
    
    """
    src_files_to_mosaic = [rasterio.open(f) for f in files_to_merge]
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=-999.9)
    # Update metadata for the new combined raster
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    # Write the merged GeoTIFF
    with rasterio.open(fullFile_tmp, "w", **out_meta) as dest:
        dest.write(mosaic)
    """
    # reproject for portal
    print("A")
    print(fullFile_tmp)
    print(rasFile)
    try:
        # reproject onto EPSG:3857 for portal usage
        if os.path.exists(rasFile):
            os.system('rm '+rasFile)

        ds = gdal.Warp(rasFile, fullFile_tmp, srcSRS='EPSG:4326', dstSRS='EPSG:3857', format='GTiff',creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        ds = None 
    except Exception as e:
        print("Failed to write to output directory")
        print(e)
        
    #  archive the 3857 portal file too 
    archFile =os.path.join(archDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(slt)+'hr_3857.tif')
    try:
        os.system('cp '+rasFile+' '+archFile)
    except:
        print("Failed to write to geotiff archive directory")
    os.system('rm '+fullFile_tmp)