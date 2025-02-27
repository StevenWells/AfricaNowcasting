# -*- coding: utf-8 -*-


import numpy as np
import util
import xarray as xr
import powerBlob_utils
import datetime as dt


def wavelet_analysis(meteosat_data, longitudes, latitudes, date, savefile, data_resolution=5):


    outt, nogood, t_thresh_size, t_thresh_cut, pix_nb, area_img = powerBlob_utils.filter_img(meteosat_data, data_resolution)

    wav = util.waveletT(outt, dataset='METEOSAT5K_vera')
    meteosat_data[nogood] = np.nan

    power_msg = powerBlob_utils.find_scales_dominant(wav, nogood, area_img, dataset='MSG')

    if power_msg is None:  # if power calculation failed
        print('Power calc fail, continue')
        #return  SRA Edit 19/09/2019 to to return all zeros if failed
        power_msg=np.zeros(np.shape(meteosat_data))

    #date = dt.datetime(year, month, day, hour, minute)

    isnan = np.isnan(meteosat_data)
    meteosat_data[isnan] = 0
    new_savet = (meteosat_data * 100).astype(np.int16)

    ds = xr.Dataset()

    blob = xr.DataArray(power_msg, coords={'time': date, 'lat': latitudes, 'lon': longitudes},
                                dims=['lat', 'lon'])  # [np.newaxis, :])
    tir = xr.DataArray(new_savet, coords={'time': date, 'lat': latitudes, 'lon': longitudes},
                               dims=['lat', 'lon'])

    ds['blobs'] = blob
    ds['tir'] = tir

    ds.attrs['radii']=(np.floor(wav['scales'] / 2. / np.float(data_resolution))).astype(np.uint8)
    ds.attrs['scales_rounded'] = np.round(wav['scales']).astype(np.uint8)
    ds.attrs['scales_original'] = wav['scales']
    ds.attrs['cutout_T'] = t_thresh_size
    ds.attrs['cutout_minPixelNb'] = pix_nb
    if savefile:
        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path=savefile, mode='w', encoding=enc, format='NETCDF4')
        print('Saved ' + savefile)


    return (ds)
