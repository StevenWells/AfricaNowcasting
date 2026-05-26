# real time processing of the wind forecasts
import xarray as xr
from ecmwf.opendata import Client
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio, sys
from rasterio.transform import from_origin
from skimage.measure import block_reduce
import datetime,os,re,glob


archiveRoot = '/mnt/prj/swift/ecmwf_windfc'
tmpDir = '/home/stewells/AfricaNowcasting/tmp/ecmwf_wind/'
portalRoot_wind = '/mnt/HYDROLOGY_stewells/lawis-west-africa/ecmwf_wind'   # /YYYYMMDD
portalRoot_tcwv = '/mnt/HYDROLOGY_stewells/geotiff/ssa_ecmwf_tcwv'
portalRoot_mucape = '/mnt/HYDROLOGY_stewells/geotiff/ssa_ecmwf_mucape'

clientopts = ['azure', 'aws','google','ecmwf','ecmwf-esuites']





def get_uv_file(tmpDir,queue):
    # download latest file
    now = datetime.datetime.now(datetime.timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    if now.hour < 7:
        yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        # Try today's 00 run first, fallback to yesterday's 12 run
        search_configs = [(today_str, 0), (yesterday_str, 18)]
    # If it is between 07:00 and 19:00 UTC, today's 00 run is definitely ready.
    elif now.hour < 19:
        search_configs = [(today_str, 0)]
    # If it is late in the day (after 19:00 UTC), today's 12 run is ready.
    else:
        search_configs = [(today_str, 12), (today_str, 0)]
    
    for date_str, run_hour in search_configs:
        print(f"Trying to download uv10tcwv GRIB for date {date_str} and run hour {run_hour:02d}")
        try:
            client.retrieve(
                date=date_str,
                time=run_hour,
                step=[0,6,12,18],
                type="fc",
                param=["100u","100v","tcwv","mucape"],
                target=os.path.join(tmpDir,"uv10tcwv.data.grib"),
            )
            if os.path.exists(os.path.join(tmpDir,"uv10tcwv.data.grib")):
                queue.put(True)
                return

        except Exception as e:
            print("Failed to download uv10tcwv GRIB ")
            print(e)
            continue
    print("Failed to download uv10tcwv GRIB for any of today's runs.")
    queue.put(False)

    

def get_uv700_file(tmpDir,queue):
    now = datetime.datetime.now(datetime.timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    if now.hour < 7:
        yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        # Try today's 00 run first, fallback to yesterday's 12 run
        search_configs = [(today_str, 0), (yesterday_str, 12)]
    # If it is between 07:00 and 19:00 UTC, today's 00 run is definitely ready.
    elif now.hour < 19:
        search_configs = [(today_str, 0)]
    # If it is late in the day (after 19:00 UTC), today's 12 run is ready.
    else:
        search_configs = [(today_str, 12), (today_str, 0)]
    for date_str, run_hour in search_configs:
        try:

            client.retrieve(
                date=date_str,
                time=run_hour,
                step=[0,6,12,18],
                type="fc",
                param=["u", "v"],
                levelist=[700],
                target=os.path.join(tmpDir,"uv700.data.grib"),
            )
            if os.path.exists(os.path.join(tmpDir,"uv700.data.grib")):
                queue.put(True)
                return
        except Exception as e:
            print(f"Failed to download uv700 GRIB for run time {run_hour}: {e}")
            continue
    print("Failed to download uv700 GRIB for any of today's runs.")
    queue.put(False)

    #return got_uv10,got_uv700

def read_grib2_cfgrib(file_path,var):
    try:
        # Open GRIB2 file
        print("Converting to netcdf: "+file_path)
        ds = xr.open_dataset(file_path, engine="cfgrib")    
        print("Variables in file:", list(ds.data_vars))
        print("\nCoordinates:", list(ds.coords))
        print(ds.coords["time"].values)
        print(ds.coords["valid_time"].values)
        origin = pd.to_datetime(ds.coords["time"].values)

        fname = os.path.join(archiveRoot,str(origin.year),str(origin.month).zfill(2),var+origin.strftime('_%Y%m%d%H%M.nc'))
        os.makedirs(os.path.dirname(fname),exist_ok=True)
        if not os.path.exists(fname):
            ds.to_netcdf(fname)
        # remove grid and indx file
        os.remove(file_path)
        idxFiles= glob.glob(os.path.join(tmpDir,var+'*.idx'))
        for idxFile in idxFiles:
            os.remove(idxFile)

        return ds
    except Exception as e:
        print(f"Error reading GRIB2 file: {e}")
        return None

def get_update_times(tnow):
    lead_hours = [0, 6, 12, 18]
    today_times = [
    tnow.replace(hour=h, minute=0, second=0, microsecond=0)
    for h in lead_hours
    ] 
    yesterday = tnow - datetime.timedelta(days=1)
    yesterday_times = [
        yesterday.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in lead_hours
    ]
    tomorrow = tnow + datetime.timedelta(days=1)
    tomorrow_times = [tomorrow.replace(hour=h,minute=0,second=0,microsecond=0) for h in [0,6]]
    return yesterday_times + today_times + tomorrow_times


def get_best_estimate_date(var,tnow):
    # based on data in archive, work out what the best image would be for time now
    lead_hours = [0, 6, 12, 18]
    candidates = []
    today_times = [
    tnow.replace(hour=h, minute=0, second=0, microsecond=0)
    for h in lead_hours
    ]
    yesterday = tnow - datetime.timedelta(days=1)
    yesterday_times = [
        yesterday.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in lead_hours
    ]
    
    all_times = today_times + yesterday_times
    all_files = [os.path.join(archiveRoot,str(t.year),str(t.month).zfill(2),var+t.strftime('_%Y%m%d%H%M.nc')) for t in all_times]
    all_files = [x for x in all_files if os.path.exists(x)]
    #print(all_files)
    for f in all_files:
        # extract D from filename (adjust pattern to your format)
        m = re.search(r'(\d{8})(\d{2})', f)
        D = datetime.datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H")

        for lead in lead_hours:
            valid_time = D + datetime.timedelta(hours=lead)

            candidates.append({
                "file": f,
                "lead": lead,
                "origin": D,
                "valid": valid_time,
                "dist": abs(valid_time - tnow)
            })

    # sort by distance then lead time

   
    if len(candidates)==0:
        print("No candidates found in archive for variable "+var)
        return None
    best = min(candidates, key=lambda x: (x["dist"], x["lead"]))
    return best


def update_portal_data(var,valid_time,origin,leadtime_hrs=0,speedthreshold=0,do_tcwv=False,do_mucape=False):
    # function to extract the wind and tcwv data from the appropriate netcdf files stored in the archive
    # assumes that the forecast origin and leadtime are known
    # var: 'uv700' or 'uv10tcwv'
    # valid_time: datetime object of the forecast valid time to use (for naming output file)
    # origin: datetime object of forecast origin to use
    # leadtime: leadtime in hours

    uv_file = os.path.join(archiveRoot,str(origin.year),str(origin.month).zfill(2),var+origin.strftime('_%Y%m%d%H%M.nc'))
    # convert leadtime into index of netcdf file
    leadtime = int(leadtime_hrs/6)




    ds = xr.open_dataset(uv_file)
        # 
    if var=='uv700':
        fileUnit = '700hPa'
        uvar='u'
        vvar='v'
    else:
        if 'v100' in list(ds.data_vars):
            fileUnit = '100m'
            uvar='u100'
            vvar='v100'
        else:
            fileUnit = '10m'
            uvar='u10'
            vvar='v10'
    
    domain_pars = {'nx_raw':294,'ny_raw':243,'deltax':0.25,'xll':-18.25,'yll':-35.0}
    # crop the data to the domain of interest
    lon_min = domain_pars['xll']
    lon_max = domain_pars['xll'] + domain_pars['nx_raw'] * domain_pars['deltax']
    lat_min = domain_pars['yll']
    lat_max = domain_pars['yll'] + domain_pars['ny_raw'] * domain_pars['deltax']
    #print(f"Cropping to lon: {lon_min} to {lon_max}, lat: {lat_min} to {lat_max}")
    ds_cropped = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max,lat_min))

    ncDate = ds_cropped['time'].values.astype('datetime64[s]').astype(object)

    lats = np.array(ds_cropped["latitude"].values)
    lons = np.array(ds_cropped["longitude"].values)
    lon2d, lat2d = np.meshgrid(lons, lats)
    lat_col = lat2d.flatten()
    lon_col = lon2d.flatten()
    lon_list= lons
    lat_list = lats
    ny,nx = ds_cropped[uvar][leadtime].shape
    u = ds_cropped[uvar][leadtime].values
    v = ds_cropped[vvar][leadtime].values
    if var=='uv10tcwv':
        tcw = ds_cropped['tcwv'][leadtime].values
        tcw = np.round(tcw,1)
        try: #mucale was not always there
            mucape= ds_cropped['mucape'][leadtime].values
            mucape = np.round(mucape,1)
        except:
            mucape = None

    speed = np.hypot(u, v)
    direction = np.degrees(np.arctan2(u, v))
    direction = (direction + 360) % 360
    if speedthreshold>0:
        speed[speed<speedthreshold] = -9999
        direction[speed<speedthreshold]= -9999


    df = pd.DataFrame({
        'latitude': lat_col,
        'longitude': lon_col,
        'speed':speed.flatten().round(1),
        'direction':direction.flatten().round()})
    df['direction'] = df['direction'].astype('Int64')
    minspeed = '' if speedthreshold == 0 else f"_minspeed{speedthreshold}"
    outDir = os.path.join(portalRoot_wind,valid_time.strftime('%Y%m%d'))
    os.makedirs(outDir,exist_ok=True)
   # print(os.path.join(outDir,f"ecmwf_wind_{fileUnit}_{ncDate.strftime('%Y%m%d')}{minspeed}.csv"))
    df.to_csv(os.path.join(outDir,f"ecmwf_wind_{fileUnit}_{valid_time.strftime('%Y%m%d')}_{valid_time.strftime('%H%M')}{minspeed}.csv"), index=False)

    # UV10 also containst water column, so process that too and make a geotiff
    if var=='uv10tcwv' and do_tcwv:
        pixel_height = domain_pars['deltax']
        pixel_width = domain_pars['deltax']
        ulx = lons[0] - pixel_width / 2
        uly = lats[0] + pixel_height / 2
        transform = from_origin(ulx, uly, pixel_width, abs(pixel_height))
        # GeoTIFF profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,             # two bands: U and V
            'height': ny,
            'width': nx,
            'crs': 'EPSG:4326',     # geographic coordinates
            'transform': transform
        }
        # Write to GeoTIFF
        outDir = os.path.join(portalRoot_tcwv,valid_time.strftime('%Y%m%d'))
        os.makedirs(outDir,exist_ok=True)
        with rasterio.open(os.path.join(outDir,f"ecmwf_tcwv_{valid_time.strftime('%Y%m%d')}_{valid_time.strftime('%H%M')}.tif"), 'w', **profile) as dst:
            dst.write(tcw.astype('float32'), 1)  

    if var=='uv10tcwv' and do_mucape and mucape is not None:
        pixel_height = domain_pars['deltax']
        pixel_width = domain_pars['deltax']
        ulx = lons[0] - pixel_width / 2
        uly = lats[0] + pixel_height / 2
        transform = from_origin(ulx, uly, pixel_width, abs(pixel_height))
        # GeoTIFF profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,             # two bands: U and V
            'height': ny,
            'width': nx,
            'crs': 'EPSG:4326',     # geographic coordinates
            'transform': transform
        }
        # Write to GeoTIFF
        outDir = os.path.join(portalRoot_mucape,valid_time.strftime('%Y%m%d'))
        os.makedirs(outDir,exist_ok=True)
        print(f"Writing MUCAPE GeoTIFF to "+outDir)
        with rasterio.open(os.path.join(outDir,f"ecmwf_mucape_{valid_time.strftime('%Y%m%d')}_{valid_time.strftime('%H%M')}.tif"), 'w', **profile) as dst:
            dst.write(mucape.astype('float32'), 1)  

def run_download(job_func, job_args=(), timeout=60):
    queue = mp.Queue()
    
    def wrapped():
        job_func(*job_args, queue)
    p = mp.Process(target=wrapped)
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("Timeout — killing process")
        p.terminate()
        p.join()
        return False

    if not queue.empty():
        return queue.get()

    return False




# pick best origin 
tnow = datetime.datetime.now()
# get the latest file and put in tmp dir 

for cname in clientopts:
    try:
        client = Client(cname)
        print(f"Testing client {cname}")
        uv10 = run_download(get_uv_file, job_args=(tmpDir,))
        uv700 = run_download(get_uv700_file, job_args=(tmpDir,))
        print(f"Got uv10: {uv10}, got uv700: {uv700}")
        if uv10 and uv700:
            print(f"Successfully downloaded files using client {cname}")
            break
        else:
            print(f"Failed to download files using client {cname}, trying next client if available.")
    except Exception as e:
        print(f"Error with client {cname}: {e}")
        continue


#uv10, uv700 = get_files()
#uv10, uv700 = False,False
gotFiles = {'uv10tcwv':uv10, 'uv700':uv700}

for var in ['uv700','uv10tcwv']:
    # convert to netcdf and archive if new 
    #print(var)
    if gotFiles[var]:
        dataset = read_grib2_cfgrib(os.path.join(tmpDir, var + ".data.grib"), var)
    # get the best data for today and yesterday
 
    update_times = get_update_times(tnow)
    for update_time in update_times:
        print("Checking "+var+" for data with valid time: "+str(update_time))

        best  = get_best_estimate_date(var,update_time)
        print(best)
        # TODO check if best date currently being used- if so dont need to update
        # update the portal
        if not best is None:
            print("Generating portal data")
            update_portal_data(var,update_time,best['origin'],leadtime_hrs=best['lead'],speedthreshold=0,do_tcwv=True,do_mucape=True)
    # now do tcwv
    #if var=='uv10tcwv':
    #    print("Generating tcwv geotiff")
    #    best  = get_best_estimate_date(var,tnow)
    #    if not best is None:
    #        update_portal_data(var,update_time,best['origin'],leadtime_hrs=best['lead'],speedthreshold=0,do_tcwv=True)
