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
# testdate for default historical mode date  (to duplicate date used on https://github.com/JawairiaAA/WISER_testbed_2025/ )
testdate='202501172100'
archiveDir= "/mnt/prj/nflics/cnn_cores/geotiff/"

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
    for f in total_files:
            modTimesinceEpoc = os.path.getmtime(f)

            modificationTime = datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
            if modificationTime > datetime.today()-timedelta(minutes=1):
                idate =''.join(os.path.basename(f).split('_')[3:5]).split('.')[0]
                # only process if not already done so (check existence of 6hr image - the last to process)
                if not os.path.exists(os.path.join(outRoot,idate[:8],'nowcast_cores_unet_'+idate[0:8]+'_'+idate[8:12]+'_'+str(6)+'hr_3857.tif')):
                    new_dates.append(idate)    

    # Choose latest date only for now
    if len(new_dates)>0:
        current_date = new_dates[0]
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
start_lat = -20 #5 # 5 
end_lat = -5 #21 #10
start_lon = 25 #-10
end_lon = 55 #0

# get native MSG grid (core)
#coords_filename = glob.glob('/mnt/prj/Africa_cloud/geoloc/*.npz')[0]  # this is /prj/Africa_cloud/geoloc/*.npz on the Linux system
coords_filename = glob.glob('/home/stewells/AfricaNowcasting/ancils/unet/lat_lon*.npz')[0]  # this is /prj/Africa_cloud/geoloc/*.npz on the Linux system

msg_latlon = np.load(coords_filename)
mlon = msg_latlon['lon']
mlat = msg_latlon['lat']

# find core indices using one file
lat_ind = np.where((mlat[:,1]>=start_lat) & (mlat[:,1]<=end_lat))[0]
lon_ind = np.where((mlon[1,:]>=start_lon) & (mlon[1,:]<=end_lon))[0]
lat = mlat[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
lon = mlon[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]

num_frames= 3   # 
t0= 1  #1   
a= 11
b= -25

lon_sub = lon[a:,:b]
lat_sub = lat[a:,:b]

# prepare resampling
dx = 0.026949456
lat_min, lat_max= np.nanmin(lat_sub),np.nanmax(lat_sub)
lon_min, lon_max= np.nanmin(lon_sub),np.nanmax(lon_sub)
grid_lat = np.arange(lat_min,lat_max ,dx)
grid_lon = np.arange(lon_min,lon_max ,dx)
grid_lon, grid_lat = np.meshgrid(grid_lon,grid_lat)
inds, weights, new_shape=uinterp.interpolation_weights(lon_sub[np.isfinite(lon_sub)], lat_sub[np.isfinite(lat_sub)],grid_lon, grid_lat, irregular_1d=True)



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


def spatial_filter_conv(predicted_image):
    
    half_window_size_px=2
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



cores = np.zeros((t,len(lat[:,1]),len(lon[1,:])),dtype=float)
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
        dir_name = '/prj/nflics/real_time_data/'+current_year+'/'+to2_date[4:6]+'/'+to2_date[6:8]+'/' 
        all_file_names = sorted(glob.glob(dir_name+"IR*.nc"));  #
        latest_to2_file = all_file_names[-4*2] 
        # check time between files 
        to_2_date=latest_to2_file[-23:-15]+latest_to2_file[-14:-10]
        to_2_datetime=datetime.strptime(str(int(to_2_date)), '%Y%m%d%H%M')
        time_difference = to_date-to_2_datetime    
        if time_difference< timedelta(hours=2.1):
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
        ds = None 
    time_core[l] = int(dates_of_interest[l])  #int(tir_filename[-15:-3])


num_frames= 3   # 
t0= 1  #1   
a= 11
b= -25

# loop over lead times here
for leadtime in leadtimes:

    print("Processing "+str(leadtime)+'hr leadtime')
    ind = np.where(cores>0)
    cores[ind] = 1 
    cores_t_0 = cores[:,a:,:b]
    tir_t_0 = tir[:,a:,:b]
    ind_tir = np.where(tir_t_0>-0.01)
    tir_t_0[ind_tir] = 0
    tir_t_0[np.isnan(tir_t_0)] = 0
    tir_t_0 = np.round(tir_t_0/-173,4)


    modelFile= '/home/stewells/AfricaNowcasting/ancils/unet/'+str(leadtime)+'hr_using_1hr/ZA_Jan_Feb_trained_model_2005_to_2019.h5'
    print(os.path.exists(modelFile))
    unet_model = tf.keras.models.load_model(modelFile, compile=False,custom_objects={'loss': FSS_loss})

    unet_model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                    loss=FSS_loss,
                    metrics=[tf.keras.metrics.Accuracy()])


    #prediction_time = time_core*0
    #for i in range(0,len(time_core)):
    #    time_core_dt = datetime.strptime(dates_of_interest[i], '%Y%m%d%H%M')
    #    prediction_time_temp = (time_core_dt+timedelta(hours=leadtime)).strftime('%Y%m%d%H%M')
    #    prediction_time[i] = int(prediction_time_temp)
    
    prediction_time = int((to_date+timedelta(hours=1)).strftime('%Y%m%d%H%M'))

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
    
    filtered_image = spatial_filter_conv(predicted_frames)
    filtered_image = np.squeeze(filtered_image[0,:,:,0])

#resample
    data_interp=uinterp.interpolate_data(filtered_image, inds, weights, new_shape)
# convert to probability
    data_interp*=100.
    # save geotiff
    # temporary file in original EPSG (to be deleted once converted to 3857 for portal)
    rasFile_tmp = '/mnt/HYDROLOGY_stewells/geotiff/ssa_nowcast_cores_unet/unet_'+str(leadtime)+'hr_tmp.tif'
    outDir = os.path.join(outRoot,current_date[0:8])
    os.makedirs(outDir,exist_ok=True)
    rasFile =     os.path.join(outDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(leadtime)+'hr_3857.tif')

    transform = rasterio.transform.from_origin(lon_min,lat_max,dx,dx)
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
    archFile =os.path.join(archDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(leadtime)+'hr_4326.tif')
    try:
        os.system('cp '+rasFile_tmp+' '+archFile)
    except:
        print("Failed to write to geotiff in 4326 archive directory")
    try:
        # reproject onto EPSG:3857 for portal usage
        ds = gdal.Warp(rasFile, rasFile_tmp, srcSRS='EPSG:4326', dstSRS='EPSG:3857', format='GTiff',creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        ds = None 
    except:
        print("Failed to write to output directory")
    os.system('rm '+rasFile_tmp)
    
    
    archFile =os.path.join(archDir,'nowcast_cores_unet_'+current_date[0:8]+'_'+current_date[8:12]+'_'+str(leadtime)+'hr_3857.tif')
    try:
        os.system('cp '+rasFile+' '+archFile)
    except:
        print("Failed to write to geotiff archive directory")
