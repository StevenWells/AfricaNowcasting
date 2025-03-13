import numpy as np
import numpy.ma as ma
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.path as mpltPath
import pandas as pd
import os,datetime,array
import xarray as xr
import csv
import u_interpolate as uinterp
import run_powerBlobs
import multiprocessing
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import subprocess
import time, calendar
import sys
import rasterio
from osgeo import gdal
from scipy import ndimage
from plot_source_destination_functions import *
import netCDF4 as nc

# include Conni's ccores package
sys.path.append('/mnt/users/hymod/seodey/NFLICS/')
import ccores.cores as cores
##################################################
# Main function to create the nowcasts
###############################################
def process_realtime_v3(tnow,datadir,rt_dir,plotdir,scratchbase,lst_path,nflics_base,rt_code_input,feed,db_version):
    processClock= time.time()
    
	
    do_full_nowcast = ['wa','sadc']  # 'wa', 'sadc'
	# initialise risk
    do_risk_subdomain = {x:False for x in do_full_nowcast}	
    
	# select domains to generate risk info (currently only 'wa')
    #do_risk_subdomain['wa'] = True
	
    do_risk_months = {'wa':[],'sadc':[]}
    nowcast_months = {'wa':['01','02','03','04','05','06','07','08','09','10','11','12'],'sadc':['01','02','03','04','05','06','07','08','09','10','11','12']}
    
    # set up whether to run nowcast
    tnow_month = tnow[4:6]
    for idomain in do_full_nowcast:
        if not tnow_month in nowcast_months[idomain]:		    
            do_full_nowcast.remove(idomain)

    #set up whether to do risk calcs (only JJAS WA at the moment?
	# default is all off, this loop turns them on based on do_risk_months
    for idomain in do_full_nowcast:	
        if tnow_month in do_risk_months[idomain]:
            do_risk_subdomain[idomain] = True
		
		
    do_extended_core_calcs=False
    do_lst_adjustments =False
    do_shared_plots=True
    do_point_timeseries = True
    do_geotiff = True
    output_site_cores=True

   # OPTIMISATION OPTIONS
    opt_geotiff = True
    opt_geotiff_float32=True
    opt_geotiff_ndpls = 2

	
   # version_maj = Main product version
   # version_min = LMF on = 2, LMF off = 1, Testing/non official version = 0
   # version_submin = LMF version being used (0=None, 1 = Connis LMF, 2 = Chris LMF)

    version_maj =    {'full':4,'wa':4, 'sadc':5}
    version_min =    {'full':0,'wa':1, 'sadc':0}
    version_submin = {'full':0,'wa':1, 'sadc':0}
	
	
    #version_maj  = 3
    #version_min = 1  # LMF on = 2, LMF off = 1, Testing/non official version = 0
    #version_submin = 0  # LMF version being used (0=None, 1 = Connis LMF, 2 = Chris LMF)

    dom_suffix = {'wa':'','sadc':'_sadc'} #WA kept as is

    #-------------------------------------------------------------------------------
    # Name:     process_realtime_v3.p
    #-------------------------------------------------------------------------------
    n_inner=41     #size in grid points of the inner square used for selecting from the historical database
    nadd=20         #number of additional points eachgvhh side to add for getting historical sample
    n_search=6
    do_irregular_lts = True
    t_search=60
   
    search_freq=60
    # lead time in hours (used if irregularly spaced)
    if do_irregular_lts:
        tsearch_irr = [0,1,2,3,4,5,6,8,10,12]
	
	
    #plot_area=[8,-20,20,0]		        #ll_lat,ll_lon,urlat,urlon in degrees
	
    #plot_area_ex=[4,-23.5,20,30]         #ll_lat,ll_lon,urlat,urlon in degrees for extended region core calculations
    #plot_area_ex=[4,-23,20,32]
    #plot_area_ex=[4,-21,20,32]
    plot_area_ex = [-41,-27,27,79] #FULL DOMAIN
    plot_area = plot_area_ex  #test, set to be the same as the extended area for nowcasts
    #plot_area_sub = {'wa':[4,-23,20,32],'sadc':[-38,8,0,55]} # 
    plot_area_sub =  {'wa':[-2,-23,20,32],'sadc':[-38,8,0,55]}	# now going further south
    #nx = 1804 #CHris' data  
    nx=2268                   #Specify grid size. origional resolution data (cells not calculated)
    ny=2080
    nx_dakarstrip=164 
    ny_dakarstrip=580
    blob_dx=0.04491576 #approx 5km over WAfrica for calculation of shape_wave (non-circular Blobs from Coni's code)
    blob_dx_3km=0.026949456 # approx 3km
    #filters_real_time=range(9,23,2)
    #filters_real_time = [15]*7
	# these are defined for EVERY HOUR, even if some hours are not included in the tsearch_irr
    filters_real_time = {}
    #filters_real_time['wa'] = list(range(9,23,2))+[21]*6
    filters_real_time['wa'] =  [25]*13
    filters_real_time['sadc'] = [25]*13
    # redefine datadir
    datadir={'wa':[],'sadc':[]}
	# TODO - AUTOMATE HIST DATABASE SELECION BY MONTH
	
    dd = datetime.datetime.strptime(tnow,"%Y%m%d%H%M")
	#lastDay = (datetime.date(dd.year + int(dd.month / 12), (dd.month % 12) + 1, 1) - datetime.timedelta(days=1)).day
	
    lastDay = (datetime.date(dd.year + int(dd.month / 12), (dd.month % 12) + 1, 1) - datetime.timedelta(days=1)).day
    for dday in [(dd.replace(day=1)- datetime.timedelta(1)),dd,(dd.replace(day=lastDay) +  datetime.timedelta(1))]:

        ilastDay = (datetime.date(dday.year + int(dday.month / 12), (dday.month % 12) + 1, 1) - datetime.timedelta(days=1)).day
	    # force 28 days for feb
        ilastDay = 28 if ilastDay==29 else ilastDay
        dstring = '2004{}01to2019{}{}'.format(str(dday.month).zfill(2),str(dday.month).zfill(2),str(ilastDay))        
        datadir['wa']+=["/mnt/prj/NC_Int_CCAd/3C/seodey/data/historical_database/WCA/msg9_cell_shape_wave_rect_"+dstring+"_WCA/msg9_cell_shape_wave_rect_"]	
        datadir['sadc'] +=["/mnt/prj/NC_Int_CCAd/3C/seodey/data/historical_database/msg9_cell_shape_wave_rect_"+dstring+"_SADC_v2/msg9_cell_shape_wave_rect_"]

    # mask dirs
    mask_dir= {}
    mask_dir['sadc'] = '/mnt/prj/NC_Int_CCAd/3C/seodey/data/historical_database/Counts_2004to2019_'
    mask_dir['wa'] = '/mnt/prj/NC_Int_CCAd/3C/seodey/data/historical_database/WCA/Counts_2004to2019_'
    archiveDir= '/mnt/prj/nflics/rt_cores/'
   # datadir['sadc'] ="/prj/nflics/historical_database/date_split_SADC_v2_realtime/msg9_cell_shape_wave_rect_20041101to20191130_SADC_v2/msg9_cell_shape_wave_rect_"
   # datadir['wa'] =  "/prj/nflics/historical_database/date_split_WA_v2_realtime/msg9_cell_shape_wave_rect_20040601to20190930_WA_v2/msg9_cell_shape_wave_rect_"
     #   datadir['wa']+=["/prj/NC_Int_CCAd/3C/seodey/data/historical_database/WCA/msg9_cell_shape_wave_rect_"+dstring+"_WCA/msg9_cell_shape_wave_rect_"]	
    #datadir['sadc'] ="/prj/NC_Int_CCAd/3C/seodey/data/historical_database/msg9_cell_shape_wave_rect_20040201to20190228_SADC_v2/msg9_cell_shape_wave_rect_"
      #  datadir['sadc'] +=["/prj/NC_Int_CCAd/3C/seodey/data/historical_database/msg9_cell_shape_wave_rect_"+dstring+"_SADC_v2/msg9_cell_shape_wave_rect_"]
    # point timeseries Info
    pt_locn_names,pt_locn_locs = {},{}
    pt_locn_names['wa'] = ['Dakar','Matam','Tambacounda','Ziguinchor','Kaolack','Iles de Saloum']
    pt_locn_locs['wa'] = [[14.734,-17.468],[15.658,-13.256],[13.773,-13.667],[12.577,-16.272],[14.160,-16.07],[13.663,-16.642]]  #[lat,lom]
    pt_locn_names['sadc'] = ['Chisamba_Chipembi', 'Kabwe_Mulungushi Univ', 'Chisamba_Gart farm', 'Mufulira_Kafironda', 'Kafulafuta_Police yard', 'Kitwe_University Cumpas', 'Katete_Katete FTC', 'Chama_Chama Ftc', 'Katete_Agric Camp', 'Lusaka_Unza', 'chongwe_KKIA', 'Luangwa_Kaunga', 'Lusaka_Unza Agric', 'Chiengi_Chiengi school', 'Milenge_Milenge', 'Nakonde_Mwenzo school', 'Mafinga_Ntendele school', 'Luwingu_school Sec School', 'Chilubi Island_Chilubi school', 'Mpulungu_Mpulungu school', 'Nsenga Hill_school', 'Kabompo_Met Yard', 'Mwinilunga_Met office', 'Kalumbila_mine site', 'Maamba_mine office', 'Gwembe_Muyunbwe', 'Chirundu_Lusitu FTC', 'Monze_kanchomba Ftc', 'Kazungula_Nyawa', 'Kaoma_Met office', 'Lukulu_Council office', 'Mulobezi_Sichili school', 'Sesheke_Met office', 'Kalabo_Met office', 'Mongu_Kataba']
    pt_locn_locs['sadc'] = [[-14.928, 28.576], [-14.291, 28.567], [-14.94656, 28.08952], [-12.613, 28.179], [-13.31669, 28.75317], [-12.774, 28.207], [-14.083, 32.061], [-11.24, 33.155], [-14.1046, 31.9201], [-15.391, 28.332], [-15.319, 28.44], [-15.61973, 30.40192], [-15.39463, 28.33722], [-8.653, 29.164], [-12.123, 29.69], [-9.333, 32.755], [-10.263, 33.374], [-10.252, 29.906], [-10.769, 30.282], [-8.773, 31.117], [-9.36441, 31.24967], [-13.596, 24.208], [-11.74, 24.431], [-12.26875, 25.3051], [-17.34, 27.187], [-16.629, 27.772], [-16.1454, 26.7912], [-16.594, 27.494], [-17.19152, 25.89093], [-14.798, 24.804], [-14.342, 23.244], [-16.712, 24.952], [-17.477, 24.301], [-14.989, 22.682], [-15.445, 23.351]]
    
    # Define Nflics output version by db_version argument  1 = extended cores ON, version-dependent plots ON, shared plots ON, suffix 'v1', 
    #                               2 = extended cores OFF, version-dependednt (inc LST) plots ON, shared plots OFF, suffix 'v2'
    #nflics_output_version=int(db_version)
	
	# subdomain geoloc file
    geoloc_sub_file = {}
    pt_locn_names_full,pt_locn_locs_full = [],[]
    squares_file= {}
    for dom in do_full_nowcast:
        sqDom = dom
        if dom =='wa':
            sqDom = 'wca'

        squares_file[dom] = nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_'+sqDom+'.nc'
        # also combine the site names
        pt_locn_names_full+=pt_locn_names[dom]
        pt_locn_locs_full+=pt_locn_locs[dom]
    geotiff_outpath = get_portal_outpath('CTT',tnow,viaS = True)
    #geotiff_outpath = '/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data'
    #geotiff_outpath = '/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/ssa_test_feed'
    #geotiff_outpath = '/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/tmp'
 #   if nflics_output_version==2:
    
    do_lst_adjustments = {}
    for dom in do_full_nowcast:	
        if version_min[dom] in [0,1]:
            do_lst_adjustments[dom]=False
        else:
            do_lst_adjustments[dom] =True
    nflics_output_version = {}
    for dom in ['full','wa','sadc']:
        nflics_output_version[dom]='_'.join([str(version_maj[dom]),str(version_min[dom]),str(version_submin[dom])])
    #nflics_output_version['sadc']='_'.join([str(version_maj),str(version_min),str(version_submin)])
# For situations in which the version of the processing is different from what is currently reuqired by the portal (ie in testing)
# This will only affect the filename on the portal, where there is a hard-coded pattern match working. 
    nflics_output_version_portal = 2 #

    #-------------------------------------------------------------------------------
    # Derived variables
    #-------------------------------------------------------------------------------
    # full domain in 5km grid
    ds_grid_info=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area,nflics_base)
	
	# do sub domains
    if 'sadc' in do_full_nowcast:
        geoloc_sub_file['sadc'] = process_grid_info(nx,ny,-1,-1,blob_dx,plot_area_sub['sadc'],nflics_base)
    if 'wa' in do_full_nowcast:
        #geoloc_sub_file['wa'] = process_grid_info(1640,580,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area_sub['wa'],nflics_base)
        geoloc_sub_file['wa'] = process_grid_info(nx,ny,-1,-1,blob_dx,plot_area_sub['wa'],nflics_base)	
    grid_lims_p=ds_grid_info["grid_lims_p"].data
    # for visible being done on 3km grid
    ds_grid_info_3km=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx_3km,plot_area,nflics_base)


    if do_extended_core_calcs:
        ds_grid_info_ex=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area_ex,nflics_base)
        #ds_grid_info_ex=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area_ex,nflics_base)
        ds_grid_info_ex=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area_ex,nflics_base)
        ds_grid_info_ex_3km=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx_3km,plot_area_ex,nflics_base)
        grid_lims_ex=ds_grid_info_ex["grid_lims_p"].data

    #--------------------------------------------------------------------------
    # Read in the cell grids where available
    #-------------------------------------------------------------------------------
    # get the required dates
    dt_now=datetime.datetime.strptime(tnow,"%Y%m%d%H%M")
    #timestamps for forecast time
    if do_irregular_lts:
        use_times = [dt_now+pd.Timedelta(minutes=60*isearch) for isearch in tsearch_irr]
    else:
        use_times=pd.date_range(dt_now,dt_now+pd.Timedelta(minutes=60*n_search),freq="60min") 
	 
    time_strs=[str(time).replace("-","").replace(" ","").replace(":","") for time in use_times]
    #forecast lead time in minutes 
    if do_irregular_lts:
        t_searches = [60*isearch for isearch in tsearch_irr]
    else:
        t_searches=list(range(0,(n_search+1)*(search_freq),search_freq))
    if feed=="historical":
		
        latlons = get_dummy_ssa_latlon()
        rt_lats=latlons[0]
        rt_lons=latlons[1]
        
		# calling the SSA domain image (to be used for both WA and full SSA
        data_all = load_data(use_times[0],[nx,ny,nx_dakarstrip,ny_dakarstrip],"ssa")
		
        #grid_lims_ex=ds_grid_info_ex["grid_lims_p"].data
        #rt_lats=np.array(ds_grid_info_ex['lats_edge'][...])  
        #rt_lons=np.array(ds_grid_info_ex['lons_edge'][...])

        #if do_extended_core_calcs:
        #    data_all_ex = data_all[grid_lims_ex[0]:grid_lims_ex[2],grid_lims_ex[1]:grid_lims_ex[3]]

        grid_lims_rt=[np.where(rt_lats[:,0]>plot_area[0])[0][0],np.where(rt_lons[0,:]>plot_area[1])[0][0],
               np.where(rt_lats[:,0]<plot_area[2])[0][-1],np.where(rt_lons[0,:]<plot_area[3])[0][-1]]
        data_all=data_all[grid_lims_rt[0]:grid_lims_rt[2],grid_lims_rt[1]:grid_lims_rt[3]][:-1,]              


        #data_all=load_data(use_times[0],[nx,ny,nx_dakarstrip,ny_dakarstrip],"WAfricaDakar")[grid_lims_p[0]:grid_lims_p[2],grid_lims_p[1]:grid_lims_p[3]]
        data_all=load_data(use_times[0],[nx,ny,nx_dakarstrip,ny_dakarstrip],"WAfricaDakar")[:len(rt_lats),:]
        data_all_vis = np.zeros((data_all.shape))*np.nan
		# crop to RT domain



        if do_extended_core_calcs:
            rt_lats_ex = np.copy(rt_lats)
            rt_lons_ex = np.copy(rt_lons)
            data_all_ex = np.copy(data_all)
        grid_lims_rt=[np.where(rt_lats[:,0]>plot_area[0])[0][0],np.where(rt_lons[0,:]>plot_area[1])[0][0],
               np.where(rt_lats[:,0]<plot_area[2])[0][-1],np.where(rt_lons[0,:]<plot_area[3])[0][-1]]
			   
			   
        data_all=data_all[grid_lims_rt[0]:grid_lims_rt[2],grid_lims_rt[1]:grid_lims_rt[3]][:-1,]

    elif feed=="eumdat":
        rt_file=rt_dir+"/IR_108_BT_"+tnow[:8]+"_"+tnow[-4:]+".nc"

        rt_lats,rt_lons,data_all=get_rt_data(rt_file,ftype='ir108_bt',haslatlon=False)

        if do_extended_core_calcs:
            rt_lats_ex = np.copy(rt_lats)
            rt_lons_ex = np.copy(rt_lons)
            data_all_ex = np.copy(data_all)
      
		
        latlons = get_dummy_ssa_latlon()
        rt_lats=latlons[0,:,:]
        rt_lons=latlons[1,:,:]

		
        data_all_vis = np.zeros(data_all.shape)*np.nan
        if do_geotiff:
            # check for equivalent visible channel file
            rt_vis_file = rt_dir+"/VIS_006_rad_"+tnow[:8]+"_"+tnow[-4:]+".nc"
            if os.path.exists(rt_vis_file):
                rt_vis_lats,rt_vis_lons,data_all_vis=get_rt_data(rt_vis_file,ftype='vis006_rad',haslatlon=False)
                
                rt_vis_lons = rt_lons
                rt_vis_lats =   rt_lats  



            else:    
                data_all_vis = np.zeros(data_all.shape)*np.nan		
        #print([rt_vis_lons.shape,data_all_vis.shape])

    elif feed=="ncas":
        rt_file=rt_dir+"/IR_108_BT_"+tnow[:8]+"_"+tnow[-4:]+".nc"
        rt_lats,rt_lons,data_all=get_rt_data(rt_file)

        if do_extended_core_calcs:
            rt_lats_ex = np.copy(rt_lats)
            rt_lons_ex = np.copy(rt_lons)
            data_all_ex = np.copy(data_all)
        #print(data_all.shape) 
        #print(rt_lats[...][:,0])
        #print(rt_lats[...][:,100])
        #print(np.array(ds_grid_info['lats_mid'][...])[:,0])
        #for ij in range(rt_lons.shape[0]):
        #grid_lims_rt = [np.where(rt_lats[:,0]>plot_area[0])[0][0],0,
		#     np.where(rt_lats[:,0]>plot_area[0])[0][0]+557,1803]  
		
        latlons = get_dummy_ssa_latlon()
        rt_lats=latlons[0]
        rt_lons=latlons[1]
       # grid_lims_rt = [np.where(rt_lats[:,0]>plot_area[0])[0][0],np.where(rt_lons[1591,:]>plot_area[1])[0][0],
	#	     np.where(rt_lats[:,0]>plot_area[0])[0][0]+557,np.where(rt_lons[1591,:]>plot_area[1])[0][0]+1803]    
        #grid_lims_rt=[np.where(rt_lats[:,0]>plot_area[0])[0][0],np.where(rt_lons[0,:]>plot_area[1])[0][0],
         #        np.where(rt_lats[:,0]<plot_area[2])[0][-1],np.where(rt_lons[0,:]<plot_area[3])[0][-1]]
     
	    
		
		
        #grid_lims_rt=[np.where(rt_lats[:,0]>plot_area[0])[0][0],np.where(rt_lons[0,:]>plot_area[1])[0][0],
        #         np.where(rt_lats[:,0]<plot_area[2])[0][-1],np.where(rt_lons[0,:]<plot_area[3])[0][-1]]
            #print([ij,grid_lims_rt,grid_lims_rt[3]-grid_lims_rt[1]])   
       
        #data_all=data_all[grid_lims_rt[0]:grid_lims_rt[2],grid_lims_rt[1]:grid_lims_rt[3]][:-1,]

		
		
        data_all_vis = np.zeros(data_all.shape)*np.nan
        if do_geotiff:
            # check for equivalent visible channel file
            rt_vis_file = rt_dir+"/VIS_006_rad_"+tnow[:8]+"_"+tnow[-4:]+".nc"
            if os.path.exists(rt_vis_file):

                rt_vis_lats,rt_vis_lons,data_all_vis=get_rt_data(rt_vis_file,ftype='VIS006_rad')

            else:    
                data_all_vis = np.zeros(data_all.shape)*np.nan
            
                


    missing=[np.all(np.isnan(grid)) for grid in data_all]
    
    #-------------------------------------------------------------------------------
    # Regrid and Calculate the core information
    #-------------------------------------------------------------------------------
    #grid information for the NFLICS domain grid used for the historical database
    lats_edge=np.array(ds_grid_info['lats_edge'][...])  
    lons_edge=np.array(ds_grid_info['lons_edge'][...])
    lats_mid=np.array(ds_grid_info['lats_mid'][...])
    lons_mid=np.array(ds_grid_info['lons_mid'][...])
    blobs_lons=np.array(ds_grid_info['blobs_lons'][...])
    blobs_lats=np.array(ds_grid_info['blobs_lats'][...])
    # 3km arrays for the visual data
    blobs_lons_3km=np.array(ds_grid_info_3km['blobs_lons'][...])
    blobs_lats_3km=np.array(ds_grid_info_3km['blobs_lats'][...])
    grid_lims_full=ds_grid_info["grid_lims_p"].data
    data_all=data_all[grid_lims_full[0]:grid_lims_full[2],grid_lims_full[1]:grid_lims_full[3]]
    data_all_vis=data_all_vis[grid_lims_full[0]:grid_lims_full[2],grid_lims_full[1]:grid_lims_full[3]]

# get subdomain latslons
    lons_mid_sub, lats_mid_sub,blobs_lons_sub, blobs_lats_sub = {},{},{},{}
    lons_edge_sub, lats_edge_sub = {},{}
    pt_locn_locs_rc	, pt_locn_locs_rc_fixed = {},{}	
    pt_locn_locs_rc_full,pt_locn_locs_rc_full_fixed = [],[]
	#for dom in list(do_full_nowcast.keys()):
    for dom in do_full_nowcast:        
        lons_mid_sub[dom] = np.array(geoloc_sub_file[dom]['lons_mid'][...])
        lats_mid_sub[dom] = np.array(geoloc_sub_file[dom]['lats_mid'][...])
        lons_edge_sub[dom] = np.array(geoloc_sub_file[dom]['lons_edge'][...])
        lats_edge_sub[dom] = np.array(geoloc_sub_file[dom]['lats_edge'][...])
        blobs_lons_sub[dom] = np.array(geoloc_sub_file[dom]['blobs_lons'][...])
        blobs_lats_sub[dom] = np.array(geoloc_sub_file[dom]['blobs_lats'][...])
		
		#hndl_nc.where(hndl_nc.apply(np.isfinite)).fillna(0.0)
        #plt.imshow(lons_edge_sub[dom])
        #plt.show()
        lats_edge_sub[dom][np.isnan(lats_edge_sub[dom])] = -999
        lons_edge_sub[dom][np.isnan(lons_edge_sub[dom])] = -999
        lats_mid_sub[dom][np.isnan(lats_mid_sub[dom])] = -999
        lons_mid_sub[dom][np.isnan(lons_mid_sub[dom])] = -999
        blobs_lats_sub[dom][np.isnan(blobs_lats_sub[dom])] = -999
        blobs_lons_sub[dom][np.isnan(blobs_lons_sub[dom])] = -999

        # get poiunt locations in terms of rows and columns OF THE SUBDOMAIN THEY APPLY TO
        pt_locn_locs_rc[dom] = [[(np.abs(lats_mid_sub[dom][:,0]-pt_locn_locs[dom][iloc][0])).argmin(),\
                        (np.abs(lons_mid_sub[dom][0,:]-pt_locn_locs[dom][iloc][1])).argmin()] for iloc in range(len(pt_locn_names[dom]))]
	    # get point locations in terms of rows and columns of fixed grid array
        pt_locn_locs_rc_fixed[dom] = [[(np.abs(blobs_lats_sub[dom][:]-pt_locn_locs[dom][iloc][0])).argmin(),\
                        (np.abs(blobs_lons_sub[dom][:]-pt_locn_locs[dom][iloc][1])).argmin()] for iloc in range(len(pt_locn_names[dom]))]					

        # full domain
    pt_locn_locs_rc_full += [[(np.abs(lats_mid[:,0]-pt_locn_locs_full[iloc][0])).argmin(),\
                        (np.abs(lons_mid[0,:]-pt_locn_locs_full[iloc][1])).argmin()] for iloc in range(len(pt_locn_names_full))]
	    # get point locations in terms of rows and columns of fixed grid array
    pt_locn_locs_rc_full_fixed += [[(np.abs(blobs_lats[:]-pt_locn_locs_full[iloc][0])).argmin(),\
                        (np.abs(blobs_lons[:]-pt_locn_locs_full[iloc][1])).argmin()] for iloc in range(len(pt_locn_names_full))]	


# weights for the full domain 
    if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz'):
        print("reading npz weights")
        weightdata = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz')
        inds = weightdata['inds_ex']
        weights= weightdata['weights']
        new_shape=tuple(weightdata['shape'])
                           
    else: # need to make it   
        print("creating weights")
        inds, weights, new_shape=uinterp.interpolation_weights(lons_mid[np.isfinite(lons_mid)], lats_mid[np.isfinite(lats_mid)],blobs_lons, blobs_lats, irregular_1d=True)
        np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz',inds_ex=inds,weights=weights,new_shape=np.array(new_shape))

   # print(os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_2_data_ex_a.npz'))
# return journey...
    if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_2_data_ex_a.npz'):
        #print(
        print("reading npz weights 2")
        weight2data = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_2_data_ex_a.npz')
        inds_2 = weight2data['inds_2_ex']
        weights_2= weight2data['weights_2_ex']
        new_shape_2=tuple(weight2data['new_shape_2_ex'])
                           
    else: # need to make it   
        print("creating weights 2")
        inds_2, weights_2, new_shape_2=uinterp.interpolation_weights(blobs_lons[np.isfinite(blobs_lons)], blobs_lats[np.isfinite(blobs_lats)], lons_mid, lats_mid)
        np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_2_data_ex_a.npz',inds_2_ex=inds_2,weights_2_ex=weights_2,new_shape_2_ex=np.array(new_shape_2))
			
			
    # visual radiation only need one-way to project onto constant pixel grid
    if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_ssa_3km.npz'):
        print("reading npz 3km weights")
        weightdata3km = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_ssa_3km.npz')
        inds_3km = weightdata3km['inds_3km']
        weights_3km = weightdata3km['weights_3km']
        new_shape_3km = tuple(weightdata3km['new_shape_3km'])
    else: #nned to make it
        print("creating weights 3km")
        inds_3km, weights_3km, new_shape_3km=uinterp.interpolation_weights( lons_mid, lats_mid,blobs_lons_3km, blobs_lats_3km)
        np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_ssa_3km.npz',inds_3km=inds_3km,weights_3km=weights_3km,new_shape_3km=np.array(new_shape_3km))

# weights for the subdomain (only need in one direction for geotiff conversion)
    weightdata_sub, inds_sub, weights_sub, new_shape_sub = {},{},{},{}

    for dom in do_full_nowcast:
        if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_'+dom+'_v2.npz'):
            weightdata_sub = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_'+dom+'_v2.npz')
            inds_sub[dom] = weightdata_sub['inds']
            weights_sub[dom]= weightdata_sub['weights']
            new_shape_sub[dom]=tuple(weightdata_sub['new_shape'])      
        else: # need to make it
            print("creating weights for subdomain "+dom)
            inds_sub[dom], weights_sub[dom], new_shape_sub[dom]=uinterp.interpolation_weights(lons_mid_sub[dom][np.isfinite(lons_mid_sub[dom])], lats_mid_sub[dom][np.isfinite(lats_mid_sub[dom])],blobs_lons_sub[dom], blobs_lats_sub[dom], irregular_1d=True)
            np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_'+dom+'_v2.npz',inds=inds_sub[dom],weights=weights_sub[dom],new_shape=np.array(new_shape_sub[dom]))
    
    #identify convectiv structures for real time image
    ###inds, weights, new_shape=uinterp.interpolation_weights( lons_mid, lats_mid,blobs_lons, blobs_lats)
    ###inds_2, weights_2, new_shape_2=uinterp.interpolation_weights(blobs_lons, blobs_lats, lons_mid, lats_mid)
    

    #print(lons_mid.shape)
    #need to interpolate the dadta to a constant resolution 5km grid for the wavelet code
    data_all_keep=np.copy(data_all)
    #print(data_all.shape)
    blobs_interp = []
    #print(use_times)
    new_method=True
    if new_method:
        #print(use_times[:1])
        for i_time,date in enumerate(use_times[:1]):
            wObj = cores.dataset('METEOSAT3K_veraLS')                                   # initialises the 3km scale decomposition and defines scale range
            wObj.read_img(np.copy(data_all[:,:]), lons_mid, lats_mid, edge_smoothing=False)   # Prepares data image for wavelets. Input here: Native MSG data and native lat/lon coordinates (irregular 2d!)        
            wObj.applyWavelet(normed='scale')
            try:
                dummy, max_location = wObj.scaleWeighting(wtype='nflics3k')
                nflics3k_da = wObj.to_dataarray(date=[date], names=['cores', 'tir'])          # Returns the calculated cores and original input image in an xarray dataset as saved in the wavelet object.
                nflics3k_da['PixelNb_-40C'] = xr.DataArray(max_location['area_pixels'], coords={'storm_idx': np.arange(len(max_location['area_pixels']))}, dims=['storm_idx'])
                nflics3k_da['max_lon'] = xr.DataArray(max_location['lon'], coords={'storm_idx': np.arange(len(max_location['area_pixels']))}, dims=['storm_idx'])
                nflics3k_da['max_lat'] = xr.DataArray(max_location['lat'], coords={'storm_idx': np.arange(len(max_location['area_pixels']))}, dims=['storm_idx'])
                blobs_interp.append( nflics3k_da['cores']  )
                if i_time==0:
                    com_lat=nflics3k_da['max_lat']
                    com_lon=nflics3k_da['max_lon']
            except:
                print('Date failed,', date)
                print(np.shape(data_all))
                blobs_interp.append(np.zeros((1,np.shape(data_all)[0],np.shape(data_all)[1]))*np.nan)
                #blobs_interp.append(np.zeros(np.shape(data_all[i_tim,:,:]))*np.nan)

        blobs_interp=np.stack(blobs_interp,axis=0)[:,0,:,:]       
        blobs_interp = blobs_interp[0]
        blobmask = np.zeros(np.shape(blobs_interp))
        blobmask[blobs_interp>0]=1
        
    else:
# SWAP OUT interolate -> powerblob -> interpolate back with jsut new core calc
        data_interp=uinterp.interpolate_data(data_all, inds, weights, new_shape)
        #run wavelet code
        data_blobs_date=run_powerBlobs.wavelet_analysis(np.copy(data_interp[:,:]), blobs_lons, blobs_lats, use_times[0],
                     "",data_resolution=5)

        com_loc=np.where(data_blobs_date["blobs"].values<0)    #power maxima of each convective structure

        #interpolate back onto MSG grid
        blobs_interp=uinterp.interpolate_data(data_blobs_date["blobs"].values, inds_2, weights_2, new_shape_2)
    
   
        blobmask_interp = np.zeros(np.shape(data_blobs_date["blobs"].values))
        blobmask_interp[data_blobs_date["blobs"].values>0]=1
	
    #mask the data to leave onlythe convective structures
    usemask=np.ones(np.shape(blobs_interp))
    usemask[(blobs_interp!=0)& ~np.isnan(blobs_interp)]=0
    data_all_m=ma.masked_array(data_all,mask=usemask)
    
        #save the data to local scratch for Chris
        #toout=np.copy(data_all_m)
    dimy=np.shape(ds_grid_info["lats_mid"].data)[0]
    dimx=np.shape(ds_grid_info["lats_mid"].data)[1]
	
    # dimx and y for subdomains
    dimx_sub, dimy_sub = {},{}
    for dom in do_full_nowcast:
        dimy_sub[dom] = np.shape(geoloc_sub_file[dom]["lats_mid"].data)[0]
        dimx_sub[dom] = np.shape(geoloc_sub_file[dom]["lats_mid"].data)[1]
	
    #ma.set_fill_value(toout,9999)
    #NEED TO ADD BACK IN!
    scratchdir=os.path.join(scratchbase,str(dt_now.year),str(dt_now.month).zfill(2),\
                        str(dt_now.day).zfill(2),str(dt_now.hour).zfill(2)+str(dt_now.minute).zfill(2))
    rt_archive=os.path.join(archiveDir,str(dt_now.year),str(dt_now.month).zfill(2),\
                        str(dt_now.day).zfill(2),str(dt_now.hour).zfill(2)+str(dt_now.minute).zfill(2))

    if not os.path.exists(scratchdir): #create plot directory if it doesn't exist
        os.makedirs(scratchdir)
    if not os.path.exists(rt_archive): #create plot directory if it doesn't exist
        os.makedirs(rt_archive)    
    print("PMAX")
    print(com_lat)
    print(com_lat.shape)
    print(com_lat[:].shape)
    print(data_all_m[:].shape)
    ds=xr.Dataset()
    ds['cores']=xr.DataArray(data_all_m[:], coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
    ds.attrs['time']=tnow
    ds.attrs['grid']="NFLICS msg cutout"
    ds.attrs['missing']="nan"
    ds["Pmax_lat"]=xr.DataArray(com_lat.values,coords={'core_ind': range(0,len(com_lat.values))}) 
    ds["Pmax_lon"]=xr.DataArray(com_lon.values,coords={'core_ind': range(0,len(com_lon.values))}) 
    ##output
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path=scratchdir+"/Convective_struct_extended_"+tnow+"_000.nc",\
                 mode='w', encoding=enc, format='NETCDF4')
    ds.to_netcdf(path=rt_archive+"/Convective_struct_extended_"+tnow+"_000.nc",\
                 mode='w', encoding=enc, format='NETCDF4')
    # output the geotiffs for the portal
    if do_geotiff:
        # 1. Cloud top temperature
# interpolate to 5km here
       # plt.imshow(data_all_keep)
       # plt.show()
        
        #rasPath = geotiff_outpath+"/Observed_CTT_"+tnow+"_extended.tif"
        rasPath = get_portal_outpath('CTT',tnow)+"/Observed_CTT_"+tnow+"_extended.tif"


        #rasPath_3857 = geotiff_outpath+"/Observed_CTT_"+tnow+"_extended_3857.tif"
        rasPath_3857 = get_portal_outpath('CTT',tnow)+"/Observed_CTT_"+tnow+"_extended_3857.tif"

        data_interp = uinterp.interpolate_data(data_all_keep, inds, weights, new_shape) # interpolate onto constant grid for geotiff

        if  opt_geotiff:                  
            if opt_geotiff_float32:
                data_interp = data_interp.astype(np.float32)
            if opt_geotiff_ndpls >=0:
                data_interp = np.round(data_interp,opt_geotiff_ndpls)



        make_geoTiff([data_interp],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
        os.system('rm '+rasPath)
        # 2. visible channel
        if not np.isnan(data_all_vis).all():
            
            #rasPath= geotiff_outpath+"/ch1_X_"+tnow+"_pc.tif"
            #rasPath_3857= geotiff_outpath+"/ch1_X_"+tnow+"_pc_3857.tif"
            rasPath= get_portal_outpath('Vis',tnow)+"/ch1_X_"+tnow+"_pc.tif"
            rasPath_3857= get_portal_outpath('Vis',tnow)+"/ch1_X_"+tnow+"_pc_3857.tif"

            data_interp_vis=uinterp.interpolate_data(data_all_vis, inds_3km, weights_3km, new_shape_3km)
# getting some weird neagitves here so crop them
            data_interp_vis[data_interp_vis<0] = np.nan

            xx= data_interp_vis.astype(float)
            xx_shifted = xx -np.nanmin(xx)
            yy= xx_shifted*100/max(np.nanmax(xx_shifted),0.000001)
            if  opt_geotiff:                  
                if opt_geotiff_float32:
                    yy = yy.astype(np.float32)
                if opt_geotiff_ndpls >=0:
                    yy = np.round(yy,opt_geotiff_ndpls)

            make_geoTiff([yy],rasPath,reprojFile=rasPath_3857,extended=True,is_vis=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
            os.system('rm '+rasPath)
        # 3. Convective structures
        #rasPath = geotiff_outpath+"/Observed_ConStruct_"+tnow+"_extended.tif"
        #rasPath_3857 = geotiff_outpath+"/Observed_ConStruct_"+tnow+"_extended_3857.tif"
        rasPath = get_portal_outpath('ConStruct',tnow)+"/Observed_ConStruct_"+tnow+"_extended.tif"
        rasPath_3857 = get_portal_outpath('ConStruct',tnow)+"/Observed_ConStruct_"+tnow+"_extended_3857.tif"
# interpolate to 5km here
        blobmask_interp = uinterp.interpolate_data(blobmask, inds, weights, new_shape) 
	
        if output_site_cores:
    # ASSUMES NEW METHOD WITH BLOBS ON NATIVE GRID
            isblobList = []
            for iloc in range(len(pt_locn_names_full)):
                isblobList+=[np.ceil(blobmask_interp[pt_locn_locs_rc_full_fixed[iloc][0],pt_locn_locs_rc_full_fixed[iloc][1]])]
            site_core_csv = plotdir+"/Site_cores_"+tnow+".csv"
            write_site_cores(site_core_csv,pt_locn_names_full,pt_locn_locs_full,isblobList)
              
        sx = ndimage.sobel(blobmask_interp, axis=0, mode='constant')
        sy = ndimage.sobel(blobmask_interp, axis=1, mode='constant')
        #sx = ndimage.sobel(blobmask, axis=0, mode='constant')
        #sy = ndimage.sobel(blobmask, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        sob_filter = 2  # original was 2
        sob[sob<=sob_filter] = 0
        sob[sob>sob_filter] = 1
        make_geoTiff([sob],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
        os.system('rm '+rasPath)
        # 4. past cores
        use_core_times=pd.date_range(dt_now,dt_now-pd.Timedelta(minutes=60*6),freq="-15min") 
        past_cores=[]
        past_times=[]
        scratchRoot = '/'.join(scratchdir.split('/')[:4])
        #scratchRoot = scratchdir
        for core_time in list(use_core_times):        
            scratchdir_cs=os.path.join(scratchRoot,'nflics_current',str(core_time.year),str(core_time.month).zfill(2),\
                       str(core_time.day).zfill(2),str(core_time.hour).zfill(2)+str(core_time.minute).zfill(2))
            #print(scratchdir_cs)
            scratchfile=scratchdir_cs+"/Convective_struct_extended_"+str(core_time).replace("-"," ").replace(":","").replace(" ","")[:-2]+"_000.nc"
            #print(scratchfile)
            #print(scratchfile)


            if os.path.exists(scratchfile):      
                #print("FOUND")         
                try:
                    core_ds=xr.open_dataset(scratchfile)   
                    past_cores.append(core_ds["cores"].data)
                #print(np.nanmax(core_ds["cores"].data))
                    past_times.append(core_ds.attrs["time"])
                    core_ds.close()
                except:
                    past_cores.append(np.zeros(np.shape(lats_mid))*np.nan)
                    past_times.append(np.nan)
            else:  
                past_cores.append(np.zeros(np.shape(lats_mid))*np.nan)
                past_times.append(np.nan)   
    
        past_cores=np.array(past_cores)

        past_cores[np.where(past_cores<0)]=1
        past_cores[np.where(np.isnan(past_cores))]=0
        
        # combine into 1 np array
        allPastCores = np.zeros(sob.shape) # ConStruct from before
        for icore in range(1,np.shape(past_cores)[0],2)[::-1]:
            ipast_Core = uinterp.interpolate_data(past_cores[icore,...], inds, weights, new_shape)            
             #apply sobel filter to hollow out
            sx = ndimage.sobel(ipast_Core, axis=0, mode='constant')
            sy = ndimage.sobel(ipast_Core, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            sob_filter = 2  # original was 2
            sob[sob<=sob_filter] = 0
            sob[sob>sob_filter] = 1
             #add to main array  
           # print([sob.shape,allPastCores.shape])
            allPastCores[sob==1] = (icore+1)/4.  # number of minutes, since this is plotting every half hour
        #shift onto the same grid as the ConStruct grid
        #get_portal_outpath('ConStruct',tnow)
        #rasPath = geotiff_outpath+"/PastCores_"+tnow+".tif"
        #rasPath_3857 = geotiff_outpath+"/PastCores_"+tnow+"_3857.tif"
        rasPath = get_portal_outpath('PastCores',tnow)+"/PastCores_"+tnow+".tif"
        rasPath_3857 = get_portal_outpath('PastCores',tnow)+"/PastCores_"+tnow+"_3857.tif"
        #print("PASTCORES")
        #print(rasPath_3857)
        make_geoTiff([allPastCores],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)    
        os.system('rm '+rasPath)   


    print(''.join(["NFLICS core time: ",str((time.time()-processClock))]))      
   
    








    if do_extended_core_calcs:
        pext= extendedCoreCalc(ds_grid_info_ex,data_all_ex,rt_lats_ex,rt_lons_ex,plot_area_ex,use_times,tnow,scratchdir,do_geotiff,geotiff_outpath,data_all_vis,ds_grid_info_ex_3km)
        pext.start()

    #-------------------------------------------------------------------------------
    # Calculate the commune core coverage and update daily and seasonal files (if needed)
    #-------------------------------------------------------------------------------
    #load in the commune_grid information
    ds_dakar=xr.open_dataset("/mnt/prj/nflics/geoloc_grids/dakar_grid_v2.nc")

    #load in "Anticedent_conditions.csv to give weighting function, WET, DRY
    ant_cond=pd.read_csv("/mnt/prj/nflics/RT_code_v2_input/Antecedent_conditions.csv",index_col=0).transpose().to_dict(orient="list")

    #load in distributions relating rg_hist to cores
    ds_rg_hist=xr.open_dataset("/mnt/prj/nflics/RT_code_v2_input/dakar_rain_hazard_v2.nc")

    
    if dt_now.month<6:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==6)[0][0]].decode() 
    elif dt_now.month>9:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==9)[0][0]].decode() 
    else:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==dt_now.month)[0][0]].decode() 

    #calculate if a core covers each commune
    cores_commune=[]
    #if there are no cores then don't bother looping
    if np.nanmin(usemask)==1:
        cores_commune=[0]*len(ds_dakar.coords["communes"])
    else:
        for i_commune,commune in enumerate(ds_dakar.coords["communes"]):
            msg_pts=ds_dakar["commune_msg_pt"][np.where(ds_dakar.coords["communes"]==commune)]
            msg_x=[int(s.split(",")[0]) for s in msg_pts.data[0].decode().split("_")]
            msg_y=[int(s.split(",")[1]) for s in msg_pts.data[0].decode().split("_")]
            # shift for Dakar on WA
            #msg_x = [x+55 for x in msg_x]
            #msg_y = [y+144 for y in msg_y]
            # shift for Dakar on full MSG
            msg_x = [x+11 for x in msg_x]
            msg_y = [y+1508 for y in msg_y]
            cores_commune.append(np.amax((~data_all_m[msg_y,msg_x].mask).astype(int)))
            #cores_commune.append(np.amin(use_mask[msg_y,msg_x]))
    ############
    #Processing of daily data

    #load in file dependent on time where a new NFLICS day startes (defined in ant_cond["Day_start"])
    if int(tnow[-4:])<int(ant_cond["Day_start"][0]): 
        yesterday=dt_now-pd.Timedelta(hours=24)
        daily_dir=os.path.join("/",*plotdir.split("/")[:-3],str(yesterday.month).zfill(2),str(yesterday.day).zfill(2))
    else:
        daily_dir=os.path.join("/",*plotdir.split("/")[:-1])
   
    daily_file=os.path.join(daily_dir,"Day_cores_"+"".join(daily_dir.split("/")[-3:])+".csv")
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)

    if not os.path.exists(daily_file): #if this is the first time in the day then need to copy template accross
        os.system("cp /mnt/prj/nflics/RT_code_v2_input/Day_cores_YYYYMMDD.csv "+daily_file)

    #read -> update -> save
    daily_cores=pd.read_csv(daily_file)             #read in daily cores file
    daily_cores[tnow[-4:]]=cores_commune            #fill in this time
    daily_cores.to_csv(daily_file,index=False)      #save the updated core file

    #format for use later in the code
    daily_cores_arr=np.array(daily_cores)[:,2:] #remove the CCRCA and Geolocation index columns
    daily_cores_arr[daily_cores_arr<0]=np.nan
    daily_cores_sites=np.array(daily_cores["Geolocation index"])

    ############
    #Processing of season data

    #load in season file
    season_file=os.path.join("/",*plotdir.split("/")[:-3],"Season_cores_total_"+daily_dir.split("/")[-3]+".csv")   #number of cores per day
    season_file_ante=os.path.join("/",*plotdir.split("/")[:-3],"Season_cores_ante_"+daily_dir.split("/")[-3]+".csv")#rain amount associated with 
                                                                                                             #  this number of cores            
    if not os.path.exists(season_file): #if this is the first time in the day then need to copy template accross
        os.system("cp /mnt/prj/nflics/RT_code_v2_input/Season_cores_total_YYYY.csv "+season_file)

    if not os.path.exists(season_file_ante): #if this is the first time in the day then need to copy template accross
        os.system("cp /mnt/prj/nflics/RT_code_v2_input/Season_cores_total_YYYY.csv "+season_file_ante)

    season_cores=pd.read_csv(season_file)
    season_cores_anti=pd.read_csv(season_file_ante)

    #day in season where the current nowcast falls. e.g. 01 June = Day 1, 10 July= Day 40
    day_in_season=(dt_now-pd.Timedelta(minutes=int(ant_cond["Day_start"][0][:2])*60+int(ant_cond["Day_start"][0][2:]))-datetime.datetime(dt_now.year,5,31,0,0)) 

    #update based on yesterdays code if we are at the last time to be processed in 24h period
    #exception is the first day in season (1 June) which is incomplete
    season_cores_anti_vec=[]
    
    if dt_now.month<6:
        print("Fix for months outside of JJAS")
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==6)].data[0]
    elif dt_now.month>9:
        print("Fix for months outside of JJAS")
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==9)].data[0]
    else:
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==dt_now.month)].data[0]
    if day_in_season.seconds==24*60*60-15*60 and day_in_season.days>0:
        print("updating daily file with yesterdays cores")
        cores_update=np.nansum(daily_cores_arr,axis=1)
        season_cores[str(day_in_season.days)]=cores_update
        cores_update[np.where(cores_update>=core_cutoff)]=core_cutoff-1  #no valid statistics beyond this poin
        if np.amax(cores_update)==0: #no point in looping if there are no cores
            season_cores_anti_vec=[0]*len(cores_update)
        #if there were some cores we need to do the actual calculation
        else:
            for site,site_cores in zip(np.array(season_cores["Geolocation index"]),cores_update): 
                if site_cores >0:
                    site_cores_anti=ds_rg_hist["mean_vec_"+month_assoc].data[np.where(ds_rg_hist.coords["cores_range"]==(site_cores-1))][0]
                    season_cores_anti_vec.append(site_cores_anti)
                else:
                    season_cores_anti_vec.append(0)

        #edit the dataframe for output to csv
        season_cores_anti[str(day_in_season.days)]=season_cores_anti_vec

    season_cores.to_csv(season_file,index=False)                #save the updated season core file
    season_cores_anti.to_csv(season_file_ante,index=False)      #save the updated season anti core file

# TO BE PUTI NTO FUNCTION BASED ON SUB DOMAIN 

    #if do_full_nowcast:
    if len(do_full_nowcast)>0: # any subdomains to make nowcasts for
        #format for use later in the code
        #cores to use for anticedent condition upto and NOT INCLUDING today.            
        anti_cores=np.array(season_cores_anti)[:,2:(day_in_season.days+1)] #remove CCRCA and Geolocation columns, then read from 1 to day_in_season-1
        anti_cores[np.where(anti_cores==-999)]=np.nan

        today_cores=np.nansum(daily_cores_arr,axis=1)
        today_cores[np.where(today_cores>=core_cutoff)]=core_cutoff-1
        #-------------------------------------------------------------------------------
        # plot the cores up to time now and LST data if it exists
        #-------------------------------------------------------------------------------
        use_core_times=pd.date_range(dt_now,dt_now-pd.Timedelta(minutes=60*6),freq="-15min") 
        past_cores=[]
        past_times=[]
        for core_time in list(use_core_times):
            scratchdir=os.path.join(scratchbase,str(core_time.year),str(core_time.month).zfill(2),\
                            str(core_time.day).zfill(2),str(core_time.hour).zfill(2)+str(core_time.minute).zfill(2))

            scratchfile=scratchdir+"/Convective_struct_extended_"+str(core_time).replace("-"," ").replace(":","").replace(" ","")[:-2]+"_000.nc"
            #print(scratchfile)
            if os.path.exists(scratchfile):
                try:
                    core_ds=xr.open_dataset(scratchfile)   
                    past_cores.append(core_ds["cores"].data)
                    past_times.append(core_ds.attrs["time"])
                    core_ds.close()
                except:
                    past_cores.append(np.zeros(np.shape(lats_mid))*np.nan)
                    past_times.append(np.nan)   
            else:
                past_cores.append(np.zeros(np.shape(lats_mid))*np.nan)
                past_times.append(np.nan)   
			
    
        past_cores=np.array(past_cores)
        past_cores[np.where(past_cores<0)]=10
        past_cores[np.where(np.isnan(past_cores))]=0


        #load in the lst data if it exists
        valid_lsta=True
        if dt_now.hour>=11:
            if os.path.exists(os.path.join(lst_path,tnow[:6])):
                try:                
                    ds_lst=xr.open_dataset(os.path.join(lst_path,tnow[:6],"LSASAF_lst_anom_Daymean_withmask_withHistClim_"+tnow[:8]+".nc"))
				
                    dat_lst=ds_lst["lst_anom_dailymean"].data
                    dat_lst[np.where(dat_lst>100)]=np.nan
                    if dt_now.hour<17:
                        lst_time_lab="0700 to "+tnow[-4:]+"UTC"
                    else:
                        lst_time_lab="0700-1700UTC"
                except:
                    dat_lst=np.zeros(np.shape(lats_mid))*np.nan
                    valid_lsta=False
            else:
                dat_lst=np.zeros(np.shape(lats_mid))*np.nan
                valid_lsta=False

        if dt_now.hour<11:#use yesterday            
            yesterday=str(dt_now-pd.Timedelta(hours=24)).replace("-"," ").replace(":","").replace(" ","")
            if os.path.exists(os.path.join(lst_path,yesterday[:6])):
                try:
                    ds_lst=xr.open_dataset(os.path.join(lst_path,yesterday[:6],"LSASAF_lst_anom_Daymean_withmask_withHistClim_"+yesterday[:8]+".nc"))
                    dat_lst=ds_lst["lst_anom_dailymean"].data
                    dat_lst[np.where(dat_lst>100)]=np.nan
                    lst_time_lab="0700-1700UTC (Day-1)"
                except:
 
                    dat_lst=np.zeros(np.shape(lats_mid))*np.nan
                    valid_lsta=False
            else:
                dat_lst=np.zeros(np.shape(lats_mid))*np.nan
                valid_lsta=False

        #load in grid information
        lst_geoloc=xr.open_dataset("/mnt/prj/nflics/SEVIRI_LST/SEVIRILST_WA_geoloc.nc")


        if not os.path.exists(plotdir): #create plot directory if it doesn't exist
            os.makedirs(plotdir)
        

       
         
        if do_shared_plots:
            plotfile=plotdir+"/LSTA_past_cores_"+tnow+"_000.png"  #DO NOT CHANGE - HARD CODED IN GUI
            
            if ~np.isnan(np.nanmax(dat_lst)):
               # plot_slice_lst(dat_lst,lst_geoloc["WA_lat"].data,lst_geoloc["WA_lon"].data,past_cores,lats_edge,lons_edge,\
               #         lats_mid,lons_mid,plotfile,'l',[8,-18,20,0],-10,\
               #            r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T anom. ($^\circ$C)"),cmap="PuOr_r",\
               #          	cell_col="cyan",use_vmin=-12,use_vmax=12,use_extend="both",\
               #             use_title="LST av. anom. ("+lst_time_lab+") from climatology  \n & identified convective structures")
                            
                plot_slice_lst(dat_lst,lst_geoloc["WA_lat"].data,lst_geoloc["WA_lon"].data,past_cores,lats_edge,lons_edge,\
                        lats_mid,lons_mid,plotfile,'l',plot_area,-10,\
                           r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T anom. ($^\circ$C)"),cmap="PuOr_r",\
                         	cell_col="cyan",use_vmin=-12,use_vmax=12,use_extend="both",\
                            use_title="LST av. anom. ("+lst_time_lab+") from climatology  \n & identified convective structures")                           
                            
            else:
			
			# THIS NEEDS FIXING - below cuurently not working...
                print("TODO: fix lst png")
            
                            
             #   plot_slice_lst(data_all_keep,lats_mid,lons_mid,past_cores,lats_edge,lons_edge,\
             #           lats_mid,lons_mid,plotfile,'l',plot_area,-10,\
             #              r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T ($^\circ$C)"),cmap="hot",\
             #            	cell_col="cyan",use_vmin=-80,use_vmax=-10,use_extend="min",\
             #               use_title="Observed cloud top temperautre  \n & identified convective structures")                            

        #-------------------------------------------------------------------------------
        # for the origin time plot the cells from the MSG image
        #-------------------------------------------------------------------------------

        def nowcast_subdomain(domain):	
           # FUNCTION TO DO NOWCASTS (AND POSSIBLY RISK IF APPROPRIATE)
            #print("Processing nowcasts for "+domain)


            # initial step - plot convective structures over domain
            plotfile=plotdir+"/Convective_struct_"+tnow+"_000"+dom_suffix[domain]+".png"  #DO NOT CHANGE - HARD CODED IN GUI

            #plot_slice_cells_blobs(data_all_keep,data_all_m.mask[...],lats_edge,lons_edge,lats_mid,lons_mid,\
            #              plotfile,'l',[8,-18,20,0],-10,\
            #               r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T ($\circ$C)"),cmap="hot",\
            #             	cell_col="cyan",use_vmin=-80,use_vmax=-10,use_extend="min",\
            #                use_title="Observed cloud top temperature \n & identified convective structures")

          #  plot_slice_cells_blobs(data_all_keep,data_all_m.mask[...],lst_geoloc["WA_lat"].data,lst_geoloc["WA_lon"].data,lats_mid,lons_mid,\
          #                plotfile,'l',plot_area_sub[domain],-10,\
          #                 r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T ($\circ$C)"),cmap="hot",\
          #               	cell_col="cyan",use_vmin=-80,use_vmax=-10,use_extend="min",\
          #                  use_title="Observed cloud top temperature \n & identified convective structures")
                            
            plot_slice_cells_blobs(data_all_keep,data_all_m.mask[...],lats_edge,lons_edge,lats_mid,lons_mid,\
                          plotfile,'l',plot_area_sub[domain],-10,\
                           r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n T ($\circ$C)"),cmap="hot",\
                         	cell_col="cyan",use_vmin=-80,use_vmax=-10,use_extend="min",\
                            use_title="Observed cloud top temperature \n & identified convective structures")				
							

            plt.close()


            if do_risk_subdomain[domain]:
                #-------------------------------------------------------------------------------
                # Polygon grids
                #-------------------------------------------------------------------------------
                grid_poly_ds=xr.open_dataset(nflics_base+'/shape_files/wca_admbnda_adm1_ocha/wca_admbnda_adm1_ocha'+str(plot_area_sub[domain]).replace(" ","").replace(",","_").replace("-","n")[1:-1]+'.nc')
                grid_poly=grid_poly_ds['poly_grid'].data
                poly_names=grid_poly_ds['names_list'].coords['names'].data
                poly_vals=grid_poly_ds['names_list'].data  
   
            #-------------------------------------------------------------------------------
            # Calculate which database files are needed
            #-------------------------------------------------------------------------------
		    #ds_grid=process_squares(ds_grid_info,n_inner,inds,weights,new_shape,nflics_base)
		
		
	
	        # 20240109 add in new function for SA
            #define com_loc using Seonaids new version
            # with added constraint that if not in subdomain then ignire it
            com_loc=((np.zeros((np.shape(com_lat.data))).astype(int),np.zeros((np.shape(com_lon.data))).astype(int)) )

            i=0
            #print(domain)
            #print(plot_area_sub[domain])
            #print(com_lat.data)
            #print(com_lon.data)
            for lat_loc,lon_loc in zip(com_lat.data,com_lon.data):        
            #print(i,lat_loc,lon_loc)
                if ((lat_loc not in lats_mid_sub[domain]) or (lon_loc not in lons_mid_sub[domain])):
                #if ((lat_loc < plot_area_sub[domain][0]) or (lat_loc > plot_area_sub[domain][2]) or (lon_loc <  plot_area_sub[domain][1]) or (lon_loc > plot_area_sub[domain][3])):
                    continue
                #print([lon_loc,lat_loc])
                #print(np.where((lats_mid_sub[domain]==lat_loc) & (lons_mid_sub[domain]==lon_loc))[1])
                #print(np.where(lats_mid_sub[domain]==lat_loc))
                #print(np.where(lons_mid_sub[domain]==lon_loc))	
                #print([np.nanmax(lons_mid_sub[domain])])			
            # lats_mid etc need to be the SUBDOMAIN ONES

                com_loc[1][i,]=np.where((lats_mid_sub[domain]==lat_loc) & (lons_mid_sub[domain]==lon_loc))[1]
                com_loc[0][i,]=np.where((lats_mid_sub[domain]==lat_loc) & (lons_mid_sub[domain]==lon_loc))[0]
                i=i+1
        
		    # (re)definition of com_loc for the specific domain 
		    # ds_grid_info has lats_mid etc for subdomains in it - these are used below
		    # needed to be loaded in for the particualr domain
		    # NEW process_grid_info function to replace/edit - extra option for SSA
		    # datadir needs to be the subdomain datadir
			
  
            print("squares for domains")	
            sqDom = domain
            if domain=='wa':
                sqDom='wca'    
            ds_grid = process_squares_msg(ds_grid_info,n_inner,nflics_base,squares_file[domain],sqDom.upper())
            load_squares=np.unique(ds_grid["squares_native"].data[com_loc])  
            load_squares=load_squares[~np.isnan(load_squares)]
            #if(domain =='sadc'): 
            #    print("squares for sadc")	    
            #    ds_grid = process_squares_msg(ds_grid_info,n_inner,nflics_base,squares_file[domain],'SADC')
            #    load_squares=np.unique(ds_grid["squares_native"].data[com_loc])  
            #    load_squares=load_squares[~np.isnan(load_squares)]	
				
		
           # else: #(WA)	
            # use this still for the WA
           #     ds_grid=process_squares(ds_grid_info,n_inner,inds,weights,new_shape,nflics_base,squares_file[domain])
            #    load_squares=np.unique(ds_grid["squares_native"].data[com_loc]) 
            #    ###load_squares=np.unique(ds_grid["squares_5km"].data[com_loc])
            #    load_squares=load_squares[~np.isnan(load_squares)]
   

            selected_squares=np.zeros(np.shape(ds_grid["squares_native"].data))*np.nan
            for f in load_squares:
                selected_squares[np.where(ds_grid["squares_native"].data==f)]=f
            #print(selected_squares)
            use_rect=np.take(ds_grid["rect_id"].data,(load_squares.astype("int")-1))
            #print(use_rect)
        
            #use_rect=[i.decode('utf-8') for i in use_rect]

            print("Calculating nowcast for source locations:", use_rect)
            #-------------------------------------------------------------------------------
            # load in frequency files from historical database, take makimum over different rect and plot
            #-------------------------------------------------------------------------------
            missing_database_vec=[]
            do_risk=0
            if (len(use_rect)>0):
                do_risk=1
                ####do_risk=0
                if (tnow[-2:]=="00"):
                    load_time=tnow[-4:]
                else:  #times on 15, 30 and 45 minutes past the hour use nowcast files for the NEXT hour ()
                    if (int(tnow[-4:-2])+1)==24:
                        load_time="0000"
                    else:
                        load_time=str(int(tnow[-4:-2])+1).zfill(2)+"00"
            
                dat_rect_t=[]
                dat_poly_t=[]
                dat_rect_pt_ts = [[] for x in range(len(pt_locn_names[domain]))] # 6 locations 
                dat_rect_fixed_pt_ts = [[] for x in range(len(pt_locn_names[domain]))] # 6 locations on fixed pixel grid
                pt_ts_filters = []
                dat_rect_max_comb =[]
                #print([use_times,t_searches])		
                for use_time,i_search in zip(use_times,t_searches):   
                    missing_database_vec_t=[]
                    dat_rect=[]
                    # tmp holders
                    dat_rect_m1,dat_rect_m2,dat_rect_m3 = [],[],[]
                    missing_database_vec_t_m1,missing_database_vec_t_m2,missing_database_vec_t_m3 = [],[],[]			
                    #rect_rt_loop(dat_rect,use_rect,datadir,i_search,load_time,filters_real_time,missing_database_vec_t,nadd,lats_mid,domain,poly_val_freq)
					# NEED TO  SORT DATADIR
					
                    dat_rect_max_m1, missing_database_vec_t_m1,poly_val_freq_m1=rect_rt_loop(dat_rect_m1,use_rect,datadir[domain][0],i_search,load_time,filters_real_time[domain],missing_database_vec_t_m1,nadd,lats_mid_sub[domain],do_risk_subdomain[domain])
                    dat_rect_max_m2, missing_database_vec_t_m2,poly_val_freq_m2=rect_rt_loop(dat_rect_m2,use_rect,datadir[domain][1],i_search,load_time,filters_real_time[domain],missing_database_vec_t_m2,nadd,lats_mid_sub[domain],do_risk_subdomain[domain])
                    dat_rect_max_m3, missing_database_vec_t_m3,poly_val_freq_m3=rect_rt_loop(dat_rect_m3,use_rect,datadir[domain][2],i_search,load_time,filters_real_time[domain],missing_database_vec_t_m3,nadd,lats_mid_sub[domain],do_risk_subdomain[domain])
                    #print(domain)
                    #plt.imshow(dat_rect_max_m1)
                    #plt.show()
                    #plt.imshow(dat_rect_max_m2)
                    #plt.show()
                    #plt.imshow(dat_rect_max_m3)
                    #plt.show()
				#combine 3 months of climatologies
                    ###dat_rect_max=0.5*(1-(dt_now.day/lastDay))*dat_rect_max_m1+0.5*dat_rect_max_m2+0.5*(dt_now.day/lastDay)
                    if (np.all(dat_rect_max_m1<0) and np.all(dat_rect_max_m2<0) and np.all(dat_rect_max_m3<0)): #all
                        dat_rect_max = dat_rect_max_m2
                    elif (np.all(dat_rect_max_m1<0) and np.all(dat_rect_max_m2<0)): #1 and 2
                        dat_rect_max = dat_rect_max_m3   
                    elif (np.all(dat_rect_max_m1<0) and np.all(dat_rect_max_m3<0)): #1 and 3
                        dat_rect_max = dat_rect_max_m2					
                    elif (np.all(dat_rect_max_m2<0) and np.all(dat_rect_max_m3<0)): #1 and 2
                        dat_rect_max = dat_rect_max_m1				
                    elif np.all(dat_rect_max_m1<0) : #1 
                        dat_rect_max = 0.5*dat_rect_max_m2+0.5*dat_rect_max_m3			
                    elif np.all(dat_rect_max_m2<0) : #2
                        dat_rect_max = 0.5*dat_rect_max_m1+0.5*dat_rect_max_m3																
                    elif np.all(dat_rect_max_m3<0) : #3		
                        dat_rect_max = 0.5*dat_rect_max_m1+0.5*dat_rect_max_m2								
                    else:
                        dat_rect_max=0.5*(1-(dt_now.day/lastDay))*dat_rect_max_m1+0.5*dat_rect_max_m2+0.5*(dt_now.day/lastDay)*dat_rect_max_m3
                    #if domain =='sadc':
                    #    plt.imshow(dat_rect_max)
                    #    plt.show()
		            # poly val_freq use the one from the actual day??
                    poly_val_freq = poly_val_freq_m2
		
		
        #mask based on number of points        
                    mask=rt_mask_nowcast(mask_dir[domain],dt_now,lastDay,i_search)
                    dat_rect_max[np.where(mask==1)]=-998.0
                    """
					BEEN PUT INTO A FUNCTION RECT_RT_LOOP
                    for i_rect,rect in enumerate(use_rect):
                        timeStampStart= time.time()		  		    
                        loaddir=datadir[domain]+rect+"_"+str(nadd)+"/Data_clim_freq_rectrect_"+rect+"_"+str(nadd)+"_search_"+str(i_search)+"_"+\
                           "refhours_"+load_time+"00_"+load_time+"00.nc"
                        #print([os.path.exists(loaddir),loaddir])
                        if os.path.exists(loaddir):
                            ds=xr.open_dataset(loaddir)
                            
                            if ds.attrs["ngrids"]>=20:
														
                                flt_ind=np.where(ds.flt.data==filters_real_time[domain][int(i_search/60)])[0][0]
                                dat_rect.append(ds["freq"].data[flt_ind,:,:]) 
                                
                                if do_risk_subdomain[domain]:
                                    if(i_rect)==0:  
                                        poly_val_freq=ds.attrs["poly_val_freq"]
                                    else:
                                        poly_val_freq=np.maximum(poly_val_freq,ds.attrs["poly_val_freq"])
                            else:  
                                missing_database_vec_t.append(loaddir.split("/")[-1])
                                fill_missing_ds=np.zeros(np.shape(lats_mid_sub[domain]))-999
                                dat_rect.append(fill_missing_ds)
                                if do_risk_subdomain[domain]:
                                    if(i_rect)==0:
                                        poly_val_freq=-999
                                    else:
                                        poly_val_freq=poly_val_freq
                            ds.close()
                        else:
                            missing_database_vec_t.append(loaddir.split("/")[-1])
                            fill_missing_ds=np.zeros(np.shape(lats_mid_sub[domain]))-999
                            dat_rect.append(fill_missing_ds)
                            if do_risk_subdomain[domain]:							
                                if(i_rect)==0:
                        	        poly_val_freq=-999
                                else:
                        	        poly_val_freq=poly_val_freq 
                   # print(''.join(["time: ",str((time.time()-timeStampStart))]))   
                #combine grids for this time taking the maximum over the different source locations selected
                    dat_rect=np.stack(dat_rect,axis=0)

                    dat_rect_max = np.nanmax(dat_rect,axis=0)
                    """
					
					

                 # only apply after 11AM
                    if dt_now.hour>=11 and do_lst_adjustments[domain] and valid_lsta: 
                        print("Applying LST adjustments")
                        # get LST grid on NFLICS domain (copied from Seonaid's historical runs - using same geolocs file)
                        ###dat_lst_cut= dat_lst[::-1,][290:702,2:717]
                        dat_lst_cut = dat_lst[::-1][146:702,:1436]
                    			
                        # derive the probabilities
                         #lsta_pval_pre = pcalc_run(dt_now.hour,dat_lst_cut,os.path.join(nflics_base,'lsta_pdist'))
					    #lsta_pval = pcalc_run(dt_now.hour,dat_lst_cut,os.path.join(nflics_base,'lsta_pdist'))
                    
                        lsta_pval=np.ones(np.shape(lats_mid)) #array of ones on MSG grid
                        lsta_pval[:,53:1489] = pcalc_run(dt_now.hour,dat_lst_cut,os.path.join(nflics_base,'lsta_pdist'))

                        if i_search==0:
                            plot_lsta_adj=plotdir+"/LSTA_scaling_v"+str(nflics_output_version[domain])+"_"+tnow+".png"
                    #plot_slice_cells_blobs(lsta_pval,lsta_pval,lats_edge,lons_edge,lats_mid,lons_mid,\
                    #      plot_lsta_adj,'l',[8,-18,20,0],-10,\
                    #       r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n Scaling factor"),cmap="hot",\
                    #     	cell_col="none",use_vmin=-80,use_vmax=-10,use_extend="min",\
                    #        use_title="LSTA Adjustment factors")
                    #plt.close()
			
                            maxLSTval = int(np.ceil(np.nanmax(lsta_pval)))
                            plot_lst_adjustment_factors(lsta_pval,lats_edge,lons_edge,lats_mid,lons_mid,plot_lsta_adj,'l',\
                                plot_area,r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n Scaling factor"),cmap="hot",\
                                use_vmin=0,use_vmax=maxLSTval,use_extend="min",use_title="LSTA Adjustment factors")
						#plot_lst_adjustment_factors(lsta_pval,lats_edge,lons_edge,lats_mid,lons_mid,plot_lsta_adj,'l',\
                        #    [8,-18,20,0],r'{}'.format(str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC \n \n Scaling factor"),cmap="hot",\
                        #    use_vmin=0,use_vmax=maxLSTval,use_extend="min",use_title="LSTA Adjustment factors")    
                            
                            
                            plt.close()
                     # convert any nans to 1, such that they wont alter the probabilites when multiplied
                        lsta_pval[np.isnan(lsta_pval)] = 1.
                    	
                            
                   
                         # multiple the nowcast array by the scaling factors
                        dat_rect_max[:,:]=np.multiply(dat_rect_max[:,:],lsta_pval)
                        # get polygon-averaged scaling factors and apply to poly_val_freq
                        if do_risk_subdomain[domain]:						
                            poly_avg_scaling_factors = get_poly_avgs(plot_area,nflics_base,lsta_pval)    
                            tmp = [-1]*len(ds.attrs["poly_ind_freq"])
                            for ind,p in enumerate(ds.attrs["poly_ind_freq"]):
                                tmp[ind] = np.rint(float(poly_val_freq[ind])*poly_avg_scaling_factors[p]) if p in poly_avg_scaling_factors.keys() else poly_val_freq[ind]
                            poly_val_freq = np.array(([int(x) for x in tmp])) 



                # any probabilites greater than 100%, set to 100%
                #dat_rect[0,:,:][dat_rect[0,:,:]>100] = 100.    
                    # END OF LSTA IF
                    dat_rect_max[:,:][dat_rect_max[:,:]>100] = 100.    
                    for iloc in range(len(pt_locn_names[domain])):
                        dat_rect_pt_ts[iloc].append(dat_rect_max[pt_locn_locs_rc[domain][iloc][0],pt_locn_locs_rc[domain][iloc][1]])
                    pt_ts_filters.append((filters_real_time[domain][int(i_search/60)]*5/(2*3))-1) 
                    dat_rect_max_comb.append(dat_rect_max)
                    #plot the probability grids
				
                
				
                    if do_geotiff:
                        # make a constant 5km pixel version of the probability grids
                        #print(dat_rect_max.shape)

                        dat_rect_geotiff_interp=uinterp.interpolate_data(dat_rect_max, inds_sub[domain], weights_sub[domain], new_shape_sub[domain])
                        dat_rect_geotiff_interp_grid =dat_rect_geotiff_interp[:] 

                        #rasPath = geotiff_outpath+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+dom_suffix[domain]+".tif"
                        #rasPath_3857 = geotiff_outpath+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+"_3857"+dom_suffix[domain]+".tif"
                        rasPath = get_portal_outpath('Nowcast',tnow)+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+dom_suffix[domain]+".tif"
                        rasPath_3857 = get_portal_outpath('Nowcast',tnow)+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+"_3857"+dom_suffix[domain]+".tif"


                        if  opt_geotiff:                  
                            if opt_geotiff_float32:
                                dat_rect_geotiff_interp_grid = dat_rect_geotiff_interp_grid.astype(np.float32)
                            if opt_geotiff_ndpls >=0:
                                dat_rect_geotiff_interp_grid = np.round(dat_rect_geotiff_interp_grid,opt_geotiff_ndpls)



                        make_geoTiff([dat_rect_geotiff_interp_grid],rasPath,reprojFile=rasPath_3857,doReproj=False,is_nowcast=True,subdomain=domain,v_maj=version_maj[domain],v_min=version_min[domain],v_submin=version_submin[domain])
                        if do_point_timeseries and len(pt_locn_names[domain])>0:
                            for iloc in range(len(pt_locn_names[domain])):
                                dat_rect_fixed_pt_ts[iloc].append(dat_rect_geotiff_interp_grid[pt_locn_locs_rc_fixed[domain][iloc][0],pt_locn_locs_rc_fixed[domain][iloc][1]])
                            pointplot_tsfile='' # dont need for geotiff/portal
							
                          
                    

                		#os.system('cp '+pointplot_csvfile+' '+geotiff_outpath)
					
                    #os.system('rm '+rasPath)
				
                    plotfile=plotdir+"/Nowcast_v"+str(nflics_output_version[domain])+dom_suffix[domain]+"_"+tnow+"_"+str(i_search).zfill(3)+".png"	#DO NOT CHANGE - HARD COADED IN GUI
               # plot_slice_cells_blobs(np.nanmax(dat_rect,axis=0),np.nanmax(dat_rect,axis=0),\
               #          lats_edge,lons_edge,lats_mid,lons_mid,plotfile,'l',[8,-18,20,0],-40,\
               #           str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC+"+str(int(i_search/60)).zfill(1)+"h \n \n Prob. (%)",cmap="CMRmap_r",\
               #           cell_col="none",use_vmin=0,use_vmax=100,use_extend="neither",
               #             sizes=[(filters_real_time[int(i_search/60)]*5/(2*3))-1],\
               #             use_title=r'{}'.format("Probability of convective structures (%)\n Black square: spatial scale for probability"))   
                    plot_slice_cells_blobs(dat_rect_max,dat_rect_max,\
                         lats_edge_sub[domain],lons_edge_sub[domain],lats_mid_sub[domain],lons_mid_sub[domain],plotfile,'l',plot_area_sub[domain],-40,\
                          str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC+"+str(int(i_search/60)).zfill(1)+"h \n \n Prob. (%)",cmap="CMRmap_r",\
                          cell_col="none",use_vmin=0,use_vmax=100,use_extend="neither",
                            sizes=[(filters_real_time[domain][int(i_search/60)]*5/(2*3))-1],\
                            use_title=r'{}'.format("Probability of convective structures (%)\n Black square: spatial scale for probability"))   
               
                #plot_slice_cells_blobs(dat_rect_max,dat_rect_max,\
                #         lats_edge,lons_edge,lats_mid,lons_mid,plotfile,'l',[8,-18,20,0],-40,\
                #          str(use_times[0])[:-9]+"\n "+tnow[-4:]+"UTC+"+str(int(i_search/60)).zfill(1)+"h \n \n Prob. (%)",cmap="CMRmap_r",\
                #          cell_col="none",use_vmin=0,use_vmax=100,use_extend="neither",
                #            sizes=[(filters_real_time[int(i_search/60)]*5/(2*3))-1],\
                #            use_title=r'{}'.format("Probability of convective structures (%)\n Black square: spatial scale for probability"))                             
                    plt.close() 
                    if do_risk_subdomain[domain]:					
                        dat_poly_t.append(poly_val_freq)                                                            
                    missing_database_vec.append(missing_database_vec_t)
                pointplot_tsfile=plotdir+"/Nowcast_timeseries_v"+str(nflics_output_version[domain])+"_"+tnow+dom_suffix[domain]+".png"
                pointplot_csvfile=plotdir+"/Nowcast_timeseries_v"+str(nflics_output_version[domain])+"_"+tnow+dom_suffix[domain]+".csv"
            
                
			
           # if do_geotiff:
           #     rasPath = geotiff_outpath+"/Nowcast_"+tnow+".tif"
           #     rasPath_3857 = geotiff_outpath+"/Nowcast_"+tnow+"_3857.tif"
           #     make_geoTiff(dat_rect_max_comb,rasPath,reprojFile=rasPath_3857)


			    # output as netcdf
                dat_rect_max_comb = np.stack(dat_rect_max_comb,axis=0)
                ds_nc=xr.Dataset()
                ds_nc['Probability']=xr.DataArray(dat_rect_max_comb[::], coords={'leadtime':range(10),'ys_mid': range(dimy_sub[domain]) , 'xs_mid': range(dimx_sub[domain])},dims=['leadtime','ys_mid', 'xs_mid']) 
               #ds_nc['Probability']=xr.DataArray(dat_rect_max_comb[::], coords={'leadtime':range(7),'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['leadtime','ys_mid', 'xs_mid']) 

                ds_nc.attrs['time']=tnow
                ds_nc.attrs['grid']="NFLICS nowcast"
                ds_nc.attrs['missing']="nan"
           #      ##output
                comp = dict(zlib=True, complevel=5)
                enc = {var: comp for var in ds_nc.data_vars}
                ds_nc.to_netcdf(path=plotdir+"/Nowcast_"+tnow+"_000.nc",mode='w', encoding=enc, format='NETCDF4')
			

                if do_geotiff and do_point_timeseries and len(pt_locn_names[domain])>0:
                    writeType='w' if first_pt_ts[1] else 'a'

                   # pointplot_csvfile_fixed_grid=geotiff_outpath+"/Nowcast_timeseries_v"+str(nflics_output_version_portal)+"_"+tnow+".csv"
                    pointplot_csvfile_fixed_grid=get_portal_outpath('Nowcast_ts',tnow)+"/Nowcast_timeseries_v"+str(nflics_output_version_portal)+"_"+tnow+".csv"

                    plt_nflics_ts(str(use_times[0])[:-9]+' '+tnow[-4:]+" UTC",dat_rect_fixed_pt_ts,pt_locn_names[domain],pt_locn_locs[domain],pointplot_tsfile,pointplot_csvfile_fixed_grid,pt_ts_filters,nhrs=9,fixed=True)
                    first_pt_ts[1] = False
			
                if do_point_timeseries and len(pt_locn_names[domain])>0:
                    writeType='w' if first_pt_ts[0] else 'a'
                    #plt_nflics_ts(str(use_times[0])[:-9]+' '+tnow[-4:]+" UTC",dat_rect_pt_ts,pt_locn_names,pt_locn_locs,pointplot_tsfile,pointplot_csvfile,pt_ts_filters)
                    plt_nflics_ts(str(use_times[0])[:-9]+' '+tnow[-4:]+" UTC",dat_rect_fixed_pt_ts,pt_locn_names[domain],pt_locn_locs[domain],pointplot_tsfile,pointplot_csvfile,pt_ts_filters,nhrs=9)
                    first_pt_ts[0] = False
            #if do_geotiff:
            #    os.system('cp '+pointplot_csvfile+' '+geotiff_outpath)
            #-------------------------------------------------------------------------------
            #write out file of polygon values into csv file
            #-------------------------------------------------------------------------------
                if do_risk_subdomain[domain]:
                    dat_poly=np.stack(dat_poly_t,axis=0)
     
                    outcsv=plotdir+"/Nowcast_v"+str(nflics_output_version[domain])+"_"+tnow+"_polygons.csv"
                    with open(outcsv, 'w') as f:
                        writer=csv.writer(f)
                        to_out=["Admin. Boundary".encode("utf-8")]+t_searches
                        writer.writerow(to_out)
                        for poly in ds.attrs["poly_ind_freq"]:
                            poly_name=poly_names[np.where(poly_vals==poly)]
                            ncst=dat_poly[:,np.where(ds.attrs["poly_ind_freq"]==poly)[0][0]]
                            writer.writerow([str(poly_name[0]).encode("utf-8")]+list(ncst))

            #print(dat_poly)
                n_missing=[len(x) for x in missing_database_vec]
                if len(n_missing)>0:
                    if max(n_missing)==len(use_rect):
                        do_risk=0
                    else:
                        pass
                else:
                    pass

                if min(n_missing)>0:
                    print(missing_database_vec)
                else:
                    pass
            else:
                print("no rect to process")
                # make a dummy file
                try:
                    dummy_arr = np.ones((2,2))*-9999
                    for use_time,i_search in zip(use_times,t_searches):  
                        rasPath = get_portal_outpath('Nowcast',tnow)+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+dom_suffix[domain]+".tif"
                        rasPath_3857 = get_portal_outpath('Nowcast',tnow)+"/Nowcast_"+tnow+"_"+str(i_search).zfill(3)+"_3857"+dom_suffix[domain]+".tif"
                        make_geoTiff([dummy_arr],rasPath,reprojFile=rasPath_3857,doReproj=False,is_nowcast=True,subdomain=domain,v_maj=version_maj[domain],v_min=version_min[domain],v_submin=version_submin[domain])
                except Exception as e:
                    print(e)  


                pass
            print(['DO RISK',do_risk])
            if do_risk==1 and do_risk_subdomain[domain]:   
                #-------------------------------------------------------------------------------
                #flood risk calculations
                #-------------------------------------------------------------------------------
                #first calculate the flood hazard informaion (risk matrix column)
                #convective core hazard from polygon values at specified region polygon
                #load in Haxard_mapping.csv
                #prob(rain|core) per commune in Hazard_mapping.csv
                #   1.todays cores from day file already loaded in "today_cores" 
                #   2.read in distribution file -> select distibution to use based on todays cores
                #   3.anticedent conditions (three options - a using cores, b WET, c DRY)
                #       a) use information from season file and weighting function (Andicedent_conditions.csv) to estimate
                #       b) use "WET" value
                #       c) use "DRY" value
                # for each anticedent condition
                #   4.use distribution to map Thresh (from Anticedent_conditions.csv)-anticedent amount to probability per commune
                #   5.map to risk matrix row
            
                hazard_mapping=pd.read_csv(rt_code_input+"/Hazard_mapping.csv").to_dict(orient="list")
                vul_mapping=pd.read_csv(rt_code_input+"/Vulnerability.csv").to_dict(orient="list")
                m,n=[int(ant_w) for ant_w in ant_cond["Weighting"][0].split("_")]
                anti_vec=np.array(1/np.power(np.arange(day_in_season.days)+1,m/n)[::-1][:-1]) #upto and including YESTERDADY
            #print([anti_vec,day_in_season.days,m,n,m/n])
            #the anticedent water amounts
                anti_core_rain=[]           #needs to be calculated
                anti_dry=[int(ant_cond["Dry"][0])]*len(today_cores) #fixed per site
                anti_wet=[int(ant_cond["Wet"][0])]*len(today_cores) #fixed per site

                #the probability values (all need to be calculatd)
                prob_core_rain=[]
                prob_dry=[]
                prob_wet=[]
                mean_vals=ds_rg_hist["mean_vec_"+month_assoc]
                fits_all=ds_rg_hist["fits_all_"+month_assoc] 

            #regional probability for cores
                prob_nflics_all=np.amax(dat_poly,axis=0)
                prob_core_nflics=[]     #core probability for each commune
                for i_site,region in enumerate(hazard_mapping['wca_admbnda_adm1_ocha shapefile ']):   
                    #print([i_site,anti_cores[i_site,:].shape,anti_vec.shape])
                    anti_core_rain.append(np.nansum(anti_cores[i_site,:]*anti_vec))
                    prob_core_rain.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_core_rain[-1]))
                    prob_dry.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_dry[i_site]))
                    prob_wet.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_wet[i_site]))
                  # SCW May 2023 Had to remove encode for this to work - has the netcdf files changed?
                  # prob_core_nflics.append(prob_nflics_all[np.where(ds.attrs["poly_ind_freq"]==np.where(poly_names==region.encode())[0][0])][0])
                    prob_core_nflics.append(prob_nflics_all[np.where(ds.attrs["poly_ind_freq"]==np.where(poly_names==region)[0][0])][0])

                prob_core_nflics=np.array(prob_core_nflics)
                prob_core_rain=np.array(prob_core_rain)
                prob_wet=np.array(prob_wet)
                prob_dry=np.array(prob_dry)

                #flood hazard row in risk matrix (bottom to top)
                risk_row_core=np.zeros(np.shape(prob_core_nflics)).astype(int)
                risk_row_wet=np.zeros(np.shape(prob_core_nflics)).astype(int)
                risk_row_dry=np.zeros(np.shape(prob_core_nflics)).astype(int)
                haz_l=np.array(hazard_mapping['Low likelihood thresh'])
                haz_m=np.array(hazard_mapping['Medium likelihood thresh'])
                haz_h=np.array(hazard_mapping['High likelihood thresh'])

                risk_row_core[np.where((prob_core_nflics*prob_core_rain/100.>=haz_l) & (prob_core_nflics*prob_core_rain/100.<haz_m))]=1
                risk_row_core[np.where((prob_core_nflics*prob_core_rain/100.>=haz_m) & (prob_core_nflics*prob_core_rain/100.<haz_h))]=2
                risk_row_core[np.where(prob_core_nflics*prob_core_rain/100.>=haz_h)]=3

                risk_row_wet[np.where((prob_core_nflics*prob_wet/100.>=haz_l) & (prob_core_nflics*prob_wet/100.<haz_m))]=1
                risk_row_wet[np.where((prob_core_nflics*prob_wet/100.>=haz_m) & (prob_core_nflics*prob_wet/100.<haz_h))]=2
                risk_row_wet[np.where(prob_core_nflics*prob_wet/100.>=haz_h)]=3

                risk_row_dry[np.where((prob_core_nflics*prob_dry/100.>=haz_l) & (prob_core_nflics*prob_dry/100.<haz_m))]=1
                risk_row_dry[np.where((prob_core_nflics*prob_dry/100.>=haz_m) & (prob_core_nflics*prob_dry/100.<haz_h))]=2
                risk_row_dry[np.where(prob_core_nflics*prob_dry/100.>=haz_h)]=3


                #vulnerability column for each catchment. Note that this is for a SUBSET of sites

                vul_l=np.array(vul_mapping['Minor impacts thresh'])
                vul_m=np.array(vul_mapping['Significant impacts thresh'])
                vul_h=np.array(vul_mapping['Severe impacts thresh'])
                vul_sort=np.array(vul_mapping['Estimated population affected (2009 populaion)'])
                vul_geo=np.array(vul_mapping['Geolocation index'])
                haz_geo=np.array(hazard_mapping['Geolocation index'])
                vul_col=np.zeros(np.shape(vul_geo)).astype(int)

                vul_col[np.where((vul_sort>=vul_l) & (vul_sort<vul_m))]=1
                vul_col[np.where((vul_sort>=vul_m) & (vul_sort<vul_h))]=2
                vul_col[np.where(vul_sort>=vul_h)]=3


                vul_col_all=np.zeros(np.shape(haz_geo)).astype(int)-999
                vul_pop=np.zeros(np.shape(haz_geo)).astype(int)-999

                for v_geo,vul_c,vul_s in zip(vul_geo,vul_col,vul_sort):
                    vul_col_all[np.where(haz_geo==v_geo)]=vul_c
                    vul_pop[np.where(haz_geo==v_geo)]=vul_s.round(0)

                toout={'Commune shapefile CCRCA':hazard_mapping['Commune shapefile CCRCA'],\
                        'Geolocation index':hazard_mapping['Geolocation index'],\
                        'Prob(core)':prob_core_nflics,\
                        'Antecedent past-cores':np.array(anti_core_rain).round(0),'Antecedent wet':anti_wet,'Antecedent dry':anti_dry,\
                        'Prob(rain|past-cores)':prob_core_rain.round(0),'Prob(rain|wet)':prob_wet.round(0),'Prob(rain|dry)':prob_dry.round(0),\
                        'Risk row past-cores':risk_row_core,'Risk row wet':risk_row_wet,'Risk row dry':risk_row_dry,\
                         'Risk col':vul_col_all, 'Population at risk':vul_pop}

            #output information

                outpath=plotdir+"/Risk_nowcast_v"+str(nflics_output_version[domain])+"_"+tnow+"_000.csv"  #DO NOT CHANGE - HARD CODED IN GUI
               # outpath2 = geotiff_outpath+"/Risk_nowcast_v"+str(nflics_output_version_portal)+"_"+tnow+"_000.csv"
                outpath2 = get_portal_outpath('Risk',tnow)+"/Risk_nowcast_v"+str(nflics_output_version_portal)+"_"+tnow+"_000.csv"

                #outpath="/prj/nflics/RT_code_v2_input/test_out.csv"
                pd.DataFrame(toout).to_csv(outpath,index=False)
                if do_geotiff:
#                   os.system('cp '+outpath+' '+geotiff_outpath)
                    os.system('cp '+outpath+' '+outpath2)            


                #plot the risk and vulnerability information
                outpath_vul=os.path.join("/",*plotdir.split("/")[:-5],"metadata","vulnerability.png") #DO NOT CHANGE - HARD CODED IN GUI

                if not os.path.exists(outpath_vul): 
                    vul0_lab=str(min(vul_mapping['Minimal impacts thresh']))
                    vul1_lab=str(min(vul_mapping['Minor impacts thresh']))
                    vul2_lab=str(min(vul_mapping['Significant impacts thresh']))
                    vul3_lab=str(min(vul_mapping['Severe impacts thresh']))

                    use_ylab=["Minimal\n ("+vul0_lab+"-"+vul1_lab+")","Minor\n ("+vul1_lab+"-"+vul2_lab+")",\
                          "Moderate\n ("+vul2_lab+"-"+vul3_lab+")","Severe\n (>"+vul3_lab+")"]
                    if do_shared_plots:
                        plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,vul_col_all,outpath_vul,"vul",\
                            "Surface water flooding 2009 (population impacted)", 'Impact \n (population)',use_ylab)
                else:
                    pass
                risk_matrix=np.array(([0,0,0,0],[0,0,1,1],[1,1,2,2],[1,2,2,3],[np.nan,np.nan,np.nan,np.nan])).transpose()  #row then column

                outpath_prob="/mnt/prj/nflics/"  #DO NOT CHANGE - HARD CODED IN GUIvul_col_all)
                vul_col_all[vul_col_all<0]=-1
                toplt_risk_core=[risk_matrix[r,c] for r,c in zip(risk_row_core, vul_col_all)]
                toplt_risk_wet=[risk_matrix[r,c] for r,c in zip(risk_row_wet, vul_col_all)]
                toplt_risk_dry=[risk_matrix[r,c] for r,c in zip(risk_row_dry, vul_col_all)]
                #print(plotdir+"/Risk_estimated_v"+str(nflics_output_version[domain])+"_"+tnow+"_000.png")
#            plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_core,plotdir+"/Risk_estimated_v"+str(nflics_output_version)+"_"+tnow+"_000.png",\
 #                           "risk_ante","Flood Risk \n given estimated surface conditions",'Flood Risk')
 #           plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_wet,plotdir+"/Risk_wet_v"+str(nflics_output_version)+"_"+tnow+"_000.png",\
 #                           "risk_wet","Flood Risk \n given wet surface ("+ant_cond["Wet"][0]+"mm)",'Flood Risk')
 #           plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_dry,plotdir+"/Risk_dry_v"+str(nflics_output_version)+"_"+tnow+"_000.png",\
 #                          "risk_dry","Flood Risk \n given dry surface ("+ant_cond["Dry"][0]+"mm)",'Flood Risk')   
                plot_slice_risk(ds_dakar,geoloc_sub_file[domain],grid_poly_ds,toplt_risk_core,plotdir+"/Risk_estimated_v"+str(nflics_output_version[domain])+"_"+tnow+"_000.png",\
                            "risk_ante","Flood Risk \n given estimated surface conditions",'Flood Risk')
                plot_slice_risk(ds_dakar,geoloc_sub_file[domain],grid_poly_ds,toplt_risk_wet,plotdir+"/Risk_wet_v"+str(nflics_output_version[domain])+"_"+tnow+"_000.png",\
                            "risk_wet","Flood Risk \n given wet surface ("+ant_cond["Wet"][0]+"mm)",'Flood Risk')
                plot_slice_risk(ds_dakar,geoloc_sub_file[domain],grid_poly_ds,toplt_risk_dry,plotdir+"/Risk_dry_v"+str(nflics_output_version[domain])+"_"+tnow+"_000.png",\
                           "risk_dry","Flood Risk \n given dry surface ("+ant_cond["Dry"][0]+"mm)",'Flood Risk') 
						   
						   
              #  if int(nflics_output_version)==1: #also output 'unversioned' filename for temporary back compatibility
              #      plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_core,plotdir+"/Risk_estimated_"+tnow+"_000.png",\
              #              "risk_ante","Flood Risk \n given estimated surface conditions",'Flood Risk')
              #      plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_wet,plotdir+"/Risk_wet_"+tnow+"_000.png",\
              #              "risk_wet","Flood Risk \n given wet surface ("+ant_cond["Wet"][0]+"mm)",'Flood Risk')
              #      plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_dry,plotdir+"/Risk_dry_"+tnow+"_000.png",\
              #             "risk_dry","Flood Risk \n given dry surface ("+ant_cond["Dry"][0]+"mm)",'Flood Risk')   



                                                                                                                 
            else:
                pass  
        # END OF FUNCTION
        first_pt_ts = [True,True] # nflicsdir, geotiff dir
        for dom in do_full_nowcast:
            print("Nowcast processing for domain "+dom)
            nowcast_subdomain(dom)
		
    plt.close("all")
    if do_extended_core_calcs:
        # prevent the funnction from ending before the parallel work is completed, to prevent overlap with next file - no loss in time
        pext.join()                                     
###########################################################
#  function to calculate the rain distributions given coefficients 
###########################################################
def tofit_shift(x,fits,shift):
    if shift>=x:
        y=100
    else:
        a=fits[0]
        b=fits[1]
        c=fits[2]
        y = a/(b+(x-shift))*np.exp(-c*(x-shift))
    return(y)
                                                             
###########################################################
#  function to read in the UKCEH SFTP data from Chilbolton 
###########################################################
def get_rt_data(infile,ftype='IR108_BT',haslatlon=True):
    import netCDF4 as nc
    print(ftype)
    rt_dataset=None
    while rt_dataset is None:
        try:
            rt_dataset = nc.Dataset(infile,"r")
        except Exception as e:
            print(e)
            print("File not comleted download. Retrying in 5s")
            time.sleep(5)
            pass
    #rt_dataset=nc.Dataset(infile,"r")
    
    if haslatlon:
        rt_data=np.array(rt_dataset[ftype][0,...]).transpose()[::-1,]
        rt_lats=np.array(rt_dataset['lat_2d'][0,...]).transpose()[::-1,]
        rt_lons=np.array(rt_dataset['lon_2d'][0,...]).transpose()[::-1,]  
    else:
        rt_data=np.array(rt_dataset[ftype][...])
        rt_lons = np.ones(rt_data.shape)*-999
        rt_lats = np.ones(rt_data.shape)*-999
    print("got it")
    return(rt_lats,rt_lons,rt_data)



###########################################################
#  function to get the NFLICS grid information 
#(if grid_file exists it is loaded in if not it is created)
###########################################################
def process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area,nflics_base):

    grid_file=nflics_base+"/geoloc_grids/nxny"+str(nx)+'_'+str(ny)+'_nxnyds'+\
                str(nx_dakarstrip)+str(ny_dakarstrip)+'_blobdx'+str(blob_dx)+\
                '_area'+str(plot_area).replace(" ","").replace(",","_").replace("-","n")[1:-1]+'.nc'
				
				
    print(grid_file)
    if not os.path.exists(grid_file):
        #geolocations(combine west africa domain ("wa") with dakar strip ("ds"))
        lats_edge_wa,lons_edge_wa=get_geoloc_grids(nx,ny,True,"WAfrica")   #lats and lons are interpolated to be the edges of each pixel
        lats_mid_wa,lons_mid_wa=get_geoloc_grids(nx,ny,False,"WAfrica")   #lats and lons are interpolated to be the edges of each pixel
        lats_edge_ds,lons_edge_ds=get_geoloc_grids(nx_dakarstrip,ny_dakarstrip,True,"DakarStrip")   #lats and lons are interpolated to be the edges of each pixel
        lats_mid_ds,lons_mid_ds=get_geoloc_grids(nx_dakarstrip,ny_dakarstrip,False,"DakarStrip")   #lats and lons are interpolated to be the edges of each pixel

        lats_edge=np.concatenate((lats_edge_ds,lats_edge_wa[:,1:-1]),axis=1)    #join together with the NFLICS strip
        lons_edge=np.concatenate((lons_edge_ds,lons_edge_wa[:,1:-1]),axis=1)
        lats_mid=np.concatenate((lats_mid_ds,lats_mid_wa),axis=1)
        lons_mid=np.concatenate((lons_mid_ds,lons_mid_wa),axis=1)

        #calculate the positions of a sub-grid based on lat and long specified in plot_area
        grid_lims_p=[np.where(lats_mid[:,0]>plot_area[0])[0][0],np.where(lons_mid[0,:]>plot_area[1])[0][0],
	           np.where(lats_mid[:,0]<plot_area[2])[0][-1],np.where(lons_mid[0,:]<plot_area[3])[0][-1]]

        lats_edge=lats_edge[grid_lims_p[0]:grid_lims_p[2]+1,grid_lims_p[1]:grid_lims_p[3]+1]  #CHECK!!!
        lons_edge=lons_edge[grid_lims_p[0]:grid_lims_p[2]+1,grid_lims_p[1]:grid_lims_p[3]+1]  #CHECK!!!
        lats_mid=lats_mid[grid_lims_p[0]:grid_lims_p[2],grid_lims_p[1]:grid_lims_p[3]]
        lons_mid=lons_mid[grid_lims_p[0]:grid_lims_p[2],grid_lims_p[1]:grid_lims_p[3]]

        #grid of gridpoint areas
        area=get_area_grid(lats_edge,lons_edge,plot_area)
        dfc_pt,dfc_km=get_dfc(lats_edge,lons_edge,plot_area)

        #constant resolution ~5km grid for the blobs calculation
        blobs_lons=np.arange(plot_area[1],plot_area[3],blob_dx)
        blobs_lats=np.arange(plot_area[0],plot_area[2],blob_dx)

        ds=xr.Dataset() #create dataset to save to netcdf for future use
        dimy=np.shape(lats_mid)[0]
        dimx=np.shape(lats_mid)[1]
        ds['lats_edge']=xr.DataArray(lats_edge, coords={'ys_edge': range(dimy+1) , 'xs_edge': range(dimx+1)},dims=['ys_edge', 'xs_edge']) 
        ds['lons_edge']=xr.DataArray(lons_edge, coords={'ys_edge': range(dimy+1) , 'xs_edge': range(dimx+1)},dims=['ys_edge', 'xs_edge']) 
        ds['lats_mid']=xr.DataArray(lats_mid, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['lons_mid']=xr.DataArray(lons_mid, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['area']=xr.DataArray(area, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['dfc_km']=xr.DataArray(dfc_km, coords={'ys_edge': range(dimy+1) , 'xs_edge': range(dimx+1)},dims=['ys_edge', 'xs_edge']) 
        ds['dfc_pt']=xr.DataArray(dfc_pt, coords={'ys_edge': range(dimy+1) , 'xs_edge': range(dimx+1)},dims=['ys_edge', 'xs_edge']) 
        ds['grid_lims_p']=xr.DataArray(grid_lims_p, coords={'coord':['ll_lat','ll_lon','ur_lat','ur_lon'] },dims=['coord'])
        ds['plot_area']=xr.DataArray(plot_area, coords={'coord':['ll_lat','ll_lon','ur_lat','ur_lon'] },dims=['coord'])
        ds['blobs_lons']=xr.DataArray(blobs_lons, coords={'xs': range(len(blobs_lons)) },dims=['xs'])
        ds['blobs_lats']=xr.DataArray(blobs_lats, coords={'ys': range(len(blobs_lats)) },dims=['ys'])
        ds.attrs['nx']=nx
        ds.attrs['ny']=ny
        ds.attrs['nx_dakarstrip']=nx_dakarstrip
        ds.attrs['ny_dakarstrip']=ny_dakarstrip
        ds.attrs['blob_dx']=blob_dx
        #output
        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path=grid_file,\
                     mode='w', encoding=enc, format='NETCDF4')
    else:   #load in the data already created
        ds=xr.open_dataset(grid_file)  
        ds.close()
    #return(lats_edge,lons_edge,lats_mid,lons_mid,blobs_lats,blobs_lons,grid_lims_p,area,dfc_km,dfc_pt)
    return(ds)

#-------------------------------------------------------------------------------
# Function to load in msg data for a given date. 
#-------------------------------------------------------------------------------
#def load_data(date,nx,ny,is_deg):

#	filename='/mnt/prj/amma/cmt/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
#		   str(date).replace('-',"").replace(' ','').replace(':','')[:-2]
#	if is_deg:
#		filename=filename+'_546_182'

#	filename=filename+'.gra'
#	if(os.path.isfile(filename)):
#		gen_ints = array.array("B")
#		gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)

#		#fh=open(filename,"rb")
#		#gen_ints.read(fh,nx*ny) 
#		data=np.array(gen_ints).reshape(ny,nx)-173  #data is rows, columns
#		#fh.close()
#	else:
#		data=np.zeros([ny,nx])*np.nan
#	return(data)
def load_data(date,nx_ny,domain): 
   # print(domain)  
    if domain=='ssa':
        print("sub-saharan africa")
        nx=2268
        ny=2080
        #print([nx,ny])
        filename='/mnt/prj/Africa_cloud/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                       str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
        print(filename)	
        if(os.path.isfile(filename)):
            gen_ints = array.array("B")
            gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)
            data_strip=np.array(gen_ints).reshape(ny,nx)#-173  #data is rows, columns
        else:
            data_strip=np.zeros([ny,nx])*np.nan
        data=data_strip.astype(int)-173
            #plt.imshow(data)
            #plt.show()
	
	
    elif domain !="WAfricaDakar":

        nx=nx_ny[0]
        ny=nx_ny[1]
        if domain == "WAfrica":
            if os.uname()[1][0:4]=='wllf':    #local linux: use copy on network
                filename='/mnt/prj/amma/cmt/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                   str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
            else:
                filename='/group_workspaces/jasmin4/cehhmf/hmf/NFLICS/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                   str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
        elif domain=="DakarStrip":
            #if os.uname()[1][0:4]=='wllf':    #local linux: use copy on network
               # filename='/mnt/prj/amma/cmt/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                 #  str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
            #else:
            filename='/group_workspaces/jasmin4/cehhmf/hmf/NFLICS/msg/DakarStrip/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                   str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
        elif domain=="WAfricaDakarExt":
            #if os.uname()[1][0:4]=='wllf':    #local linux: use copy on network
               # filename='/mnt/prj/amma/cmt/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                 #  str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
            #else:
            filename='/gws/nopw/j04/cehhmf/hmf/NFLICS/msg/WAfricaDakar/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                   str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'

        if(os.path.isfile(filename)):
            gen_ints = array.array("B")
            gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)

            #fh=open(filename,"rb")
            #gen_ints.read(fh,nx*ny) 
            data=np.array(gen_ints).reshape(ny,nx)-173  #data is rows, columns
            #fh.close()
        else:
            data=np.zeros([ny,nx])*np.nan
    else:
	

        if date.year <2020:
            nx=nx_ny[0]
            ny=nx_ny[1]
            #first load in the WAfrica data
            filename='/group_workspaces/jasmin4/cehhmf/hmf/NFLICS/msg/WAfrica/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                       str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
            if(os.path.isfile(filename)):
                gen_ints = array.array("B")
                gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)
                data=np.array(gen_ints).reshape(ny,nx)-173  #data is rows, columns
            else:
                data=np.zeros([ny,nx])*np.nan

            filename='/group_workspaces/jasmin4/cehhmf/hmf/NFLICS/msg/DakarStrip/ch9/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                       str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
                    
            nx=nx_ny[2]
            ny=nx_ny[3]
            if(os.path.isfile(filename)):
                gen_ints = array.array("B")
                gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)
                data_strip=np.array(gen_ints).reshape(ny,nx)-173  #data is rows, columns
            else:
                data_strip=np.zeros([ny,nx])*np.nan
            data=np.concatenate((data_strip,data),axis=1)
        else:
            print("concatenated domain")
            nx=nx_ny[0]+nx_ny[2]
            ny=nx_ny[1]
            #first load in the WAfrica data
            filename='/mnt/prj/amma/cmt/msg/WAfrica/ch9_X/'+str(date.year)+'/'+str(date.month).zfill(2)+'/'+\
                       str(date).replace('-',"").replace(' ','').replace(':','')[:-2]+'.gra'
            print(filename)	
            if(os.path.isfile(filename)):
                gen_ints = array.array("B")
                gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)
                #gen_ints=np.fromfile(open(filename, 'rb'), dtype='i1',count=-1)            
                #rint(gen_ints.itemsize)            
                #print(np.shape(gen_ints))
                data_strip=np.array(gen_ints).reshape(ny,nx)#-173  #data is rows, columns
            else:
                data_strip=np.zeros([ny,nx])*np.nan
            data=data_strip.astype(int)-173
    return(data)

###########################################################
#  function to read in the squares (source location) data
#  creates this if it doesn't exist
###########################################################

def process_squares(ds_grid_info,n_inner,inds,weights,new_shape,nflics_base,squares_file):
    xadd=55
    yadd=144
    if not os.path.exists(squares_file):
	    # Calculate which database files are needed
    #-------------------------------------------------------------------------------	
        print("calculating squares file")
        inds_squares=np.indices(np.shape(ds_grid_info["lats_mid"].data))
        inds_select=(inds_squares[1][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner],inds_squares[0][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner])
        xs=inds_select[0].astype(float)
        ys=inds_select[1].astype(float)
        
        if np.shape(ds_grid_info["lats_mid"])[1] >715:
            xs=xs+xadd%n_inner
            ys=ys+yadd%n_inner
            toplt=np.copy(ds_grid_info["dfc_km"].data)
            toplt[toplt<0]=np.nan
            dfc_km_deg=np.copy(ds_grid_info["dfc_km"].data)[(n_inner+xadd%n_inner):-n_inner:n_inner,(n_inner+yadd%n_inner):-n_inner:n_inner]
        else:
            toplt=np.copy(ds_grid_info["dfc_km"].data)
            toplt[toplt<0]=np.nan
            dfc_km_deg=np.copy(ds_grid_info["dfc_km"].data)[n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner]
        dfc_km_deg[np.where(dfc_km_deg<0)]=np.nan

        xs[np.where(np.isnan(dfc_km_deg)==True)]=np.nan
        ys[np.where(np.isnan(dfc_km_deg)==True)]=np.nan

        xs=xs[~np.isnan(xs)].astype("int")
        ys=ys[~np.isnan(ys)].astype("int")

       # ys[np.where((xs==465) & (ys==62))]= ys[np.where((ys==465) & (ys==62))]-20
        #ys[np.where((xs==506) & (ys==62))]= ys[np.where((ys==506) & (ys==62))]-20
        
        #plt.contou(f(toplt)
        #plt.colorbar()
        #plt.scatter(xs.flatten(),ys.flatten(),marker=".",color="red")
        #plt.savefig()

        square_dat=np.zeros(np.shape(ds_grid_info["lats_mid"].data))*np.nan
        i_square=1
        
        if np.shape(ds_grid_info["lats_mid"])[1] >715:
            print("WA DOMAIN")
            start=1
            for x,y in zip(xs,ys):
               
                if (y==62):
                    print(x,y)
                #if (x==164+xadd and y==164+yadd):
                #    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                #    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x-20+int(n_inner/2)]=i_square

                #    i_square=i_square+1           
                if (start==1):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                    print("First",x,y)
                    start=0
                elif ((y==62) and ( x==506 or x==998 or x==1039 or x==1080)):
                    y=y-10
                    print("edditing y",x,y, "& S ext")
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1 
                elif ((y==62) and (x==465)):
                    y=y-10
                    print("edditing y",x,y, "& SW ext")
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y-31+int(n_inner/2),x-21-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1     
                                                            
                elif ((x==260 and y==267) or (y==62 and x==957)):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1  
                    print("S ext",x,y)
                elif ((yold!=y) and (x!=178 and y!=390 and x!=547 and x!=1121)): 
                    if ((x==342 and y==144)or(x==301 and y==185)or (x==220 and y==267)):
                        square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-21-int(n_inner/2): x+int(n_inner/2)]=i_square
                        square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-21-int(n_inner/2): x-20+int(n_inner/2)]=i_square
                        i_square=i_square+1
                        print("S&W ext",x,y)
                    else:
                        print("first")
                        square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-21-int(n_inner/2): x+int(n_inner/2)]=i_square
                        i_square=i_square+1 
                        print("Wext",x,y)                      
                elif ((y==103 and (x==465 or x==506 or x==998 or x==1039 or x==1080 )) or (y==62 and (x==547 or x==670 or x==711))):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1   
                    print("S ext",x,y)                         
                else: #normal case
                    #print(x,y)
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                xold=x
                yold=y
                
        else:
            for x,y in zip(xs,ys):
                print(x,y)
                if (x==164 and y==164):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x-20+int(n_inner/2)]=i_square
                    i_square=i_square+1
                elif (x== 164 and y!=246) or (x== 205 and (y<130 or y>328)) :
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                else:
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1

        #plt.imshow(square_dat,cmap="flag")
        #plt.show()    
        #plt.savefig(nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v2.png',dpi=400)
        #plt.close()
        #plot_slice_squares(square_dat%10,np.array(ds_grid_info["lats_edge"][...]),\
        #                np.array(ds_grid_info["lons_edge"][...]),np.array(ds_grid_info["lats_mid"][...]),
        #                np.array(ds_grid_info["lons_mid"][...]),nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v3.png',\
        #               'i',ds_grid_info["plot_area"].data,cmap="tab20b",use_vmin=-1,use_vmax=10)

        

       # plt.show()
        #plt.close()
        
        #fix interpolation issues
        
        
        square_dat_5km=np.rint(uinterp.interpolate_data(square_dat, inds, weights, new_shape))
        tofix=np.where(np.diff(square_dat_5km,axis=0)>0)
        fily=tofix[0]+2
        filx=tofix[1]
        filval=(fily,filx)    
        square_dat_5km[tofix]=square_dat_5km[filval]        
        #
        tofix2=np.where(np.diff(square_dat_5km,axis=1)>1)
        fily2=tofix2[0]
        filx2=tofix2[1]+2
        filval2=(fily2,filx2)    
        square_dat_5km[tofix2]=square_dat_5km[filval2]   
        
        tofix2=np.where(np.diff(square_dat_5km,axis=1)<0)
        fily2=tofix2[0]
        filx2=tofix2[1]-1
        filval2=(fily2,filx2)    
        square_dat_5km[tofix2]=square_dat_5km[filval2]   
        
        
        
        if np.shape(ds_grid_info["lats_mid"])[1] >715: #wa domain
            square_dat_5km[334,192]=7
            square_dat_5km[332,274]=9
        else:
            square_dat_5km[136,292]=56

        #plt.imshow(square_dat_5km%10,cmap="flag",vmin=-1,vmax=10)
        #plt.show()
        #plt.savefig(nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_5km_v2.png',dpi=400)
        #plt.close()
        

       
        rect_ids=[]
        add_inner=(n_inner-1)/2

        for midx,midy in zip(xs,ys):
            if np.shape(ds_grid_info["lats_mid"])[1] >715: #wa domain
                if ((midy==62) and ( midx==506 or midx==998 or midx==1039 or midx==1080)):
                    midy=midy-10
                else:
                    pass
            else:
                pass                       
            rect_pt=[max(int(midy-add_inner),0),max(int(midx-add_inner),0),\
             min(int(midy+add_inner),ds_grid_info.attrs["ny"]),\
             min(int(midx+add_inner),ds_grid_info.attrs["nx"]+ds_grid_info.attrs["nx_dakarstrip"])]  
            rect_ids.append(str(rect_pt[0]).zfill(3)+"_"+str(rect_pt[1]).zfill(3)+"_"+\
                    str(rect_pt[2]).zfill(3)+"_"+str(rect_pt[3]).zfill(3))

        ds=xr.Dataset() #create dataset to save to netcdf for future use
        dimy=np.shape(ds_grid_info["lats_mid"].data)[0]
        dimx=np.shape(ds_grid_info["lats_mid"].data)[1]
        dimy_5km=np.shape(square_dat_5km)[0]
        dimx_5km=np.shape(square_dat_5km)[1]
        nsquares=np.nanmax(square_dat).astype(int)
        square_inds=np.arange(1,nsquares+1).astype(int)

        ds['squares_native']=xr.DataArray(square_dat, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['squares_5km']=xr.DataArray(square_dat_5km, coords={'ys_5km': range(dimy_5km) , 'xs_5km': range(dimx_5km)},dims=['ys_5km', 'xs_5km']) 
        ds['xs_mid']=xr.DataArray(xs, coords={'squares': square_inds },dims=['squares'])
        ds['ys_mid']=xr.DataArray(ys, coords={'squares': square_inds },dims=['squares'])
        ds['rect_id']=xr.DataArray(rect_ids, coords={'squares': square_inds},dims=['squares'])

        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path=squares_file,mode='w', encoding=enc, format='NETCDF4')    
    else:
        #print("loading squares file")
        ds=xr.open_dataset(squares_file)
        #print("loaded squares file")
    #print(ds)
    return(ds)

def process_squares_msg(ds_grid_info,n_inner,nflics_base,squares_file,domain):
  
    #yadd=0
    if not os.path.exists(squares_file):
    
	    # Calculate which database files are needed
    #-------------------------------------------------------------------------------	
        lats_edge=np.array(ds_grid_info['lats_edge'][...])  
        lons_edge=np.array(ds_grid_info['lons_edge'][...])
        lats_mid=np.array(ds_grid_info['lats_mid'][...])
        lons_mid=np.array(ds_grid_info['lons_mid'][...])

        lats_edge[np.isnan(lats_edge)]=-999
        lons_edge[np.isnan(lons_edge)]=-999
        lats_mid[np.isnan(lats_mid)]=-999
        lons_mid[np.isnan(lons_mid)]=-999 
         
        print("calculating squares file")
        inds_squares=np.indices(np.shape(ds_grid_info["lats_mid"].data))
        #inds_select=(inds_squares[1][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner],inds_squares[0][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner])
        inds_select=(inds_squares[1][n_inner::n_inner,n_inner::n_inner],inds_squares[0][n_inner::n_inner,n_inner::n_inner])
        
        xs=inds_select[0].astype(float)
        ys=inds_select[1].astype(float)
        
        toplt=np.copy(lons_mid)
        #toplt[toplt<0]=np.nan
        #dfc_km_deg=np.copy(ds_grid_info["dfc_km"].data)[n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner]
        #dfc_km_deg[np.where(dfc_km_deg<0)]=np.nan

        #xs[np.where(np.isnan(dfc_km_deg)==True)]=np.nan #SRA 17042023
        #ys[np.where(np.isnan(dfc_km_deg)==True)]=np.nan

        xs=xs[~np.isnan(xs)].astype("int")
        ys=ys[~np.isnan(ys)].astype("int")

        #print(xs)
        #print(ys)
       # ys[np.where((xs==465) & (ys==62))]= ys[np.where((ys==465) & (ys==62))]-20
        #ys[np.where((xs==506) & (ys==62))]= ys[np.where((ys==506) & (ys==62))]-20
        
        #plt.imshow(toplt,vmin=-37,vmax=6)
        #plt.colorbar()
        #plt.scatter(xs.flatten(),ys.flatten(),marker=".",color="red")
        #plt.savefig()
        #plt.show()

        square_dat=np.zeros(np.shape(ds_grid_info["lats_mid"].data))*np.nan
        i_square=1
        
        if domain =="WA":
            print("WA DOMAIN")
            xadd=55
            yadd=144
            start=1
            i=0
            for x,y in zip(xs,ys):
                
                #if (y==62):
                #    print(x,y)
                #if (x==164+xadd and y==164+yadd):
                #    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                #    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x-20+int(n_inner/2)]=i_square

                #    i_square=i_square+1
                if (y==62 and x<260)or(y==103  and   x<178)or (y<308 and x== 55)or (x>900 and y==513) or (x>1400 and y==472):
                    print("Not using square", x,y)   
                    xs[i]=-99
                    ys[i]=-99 
                elif (start==1):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                    print("First",x,y)
                    start=0
                elif ((y==62) and ( x==506 or x==998 or x==1039 or x==1080)):
                    y=y-10
                    print("edditing y",x,y, "& S ext")
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1 
                elif ((y==62) and (x==465)):
                    y=y-10
                    print("edditing y",x,y, "& SW ext")
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y-31+int(n_inner/2),x-21-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1     
                                                            
                elif ((x==260 and y==267) or (y==62 and x==957)):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1  
                    print("S ext",x,y)
                elif ((yold!=y) and (x!=178 and y!=390 and x!=547 and x!=1121)): 
                    if ((x==342 and y==144)or(x==301 and y==185)or (x==220 and y==267)):
                        square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-21-int(n_inner/2): x+int(n_inner/2)]=i_square
                        square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-21-int(n_inner/2): x-20+int(n_inner/2)]=i_square
                        i_square=i_square+1
                        print("S&W ext",x,y)
                    else:
                        print("first")
                        square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                        i_square=i_square+1 
                        print("Wext",x,y)                      
                elif ((y==103 and (x==465 or x==506 or x==998 or x==1039 or x==1080 )) or (y==62 and (x==547 or x==670 or x==711))):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1   
                    print("S ext",x,y)                         
                else: #normal case
                    #print(x,y)
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                xold=x
                yold=y
                i=i+1
            xs=xs[xs>0]
            ys=ys[ys>0]
        elif domain=="NFLICS":
            for x,y in zip(xs,ys):
                print(x,y)
                if (x==164 and y==164):
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    square_dat[y-21-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x-20+int(n_inner/2)]=i_square
                    i_square=i_square+1
                elif (x== 164 and y!=246) or (x== 205 and (y<130 or y>328)) :
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                else:
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                    
        elif domain=="SADC":
            print(domain)
            print(len(xs),len(ys))
            i=0
            for x,y in zip(xs,ys):
                if ((y==21+20 and x>=580+20) or (y==62+20 and x>=640+20) or (y==103+20 and x>=680+20) or (y==144+20 and x>=720+20) ):
                    print("Not using square 1", x,y)  
                    xs[i]=-99
                    ys[i]=-99 
                elif ((y==185+20 and x>=760+20) or (y==226+20 and x>=861)):
                    print("Not using square 2", x,y)  
                    xs[i]=-99
                    ys[i]=-99  
                elif ((y==287 and x>=1060) or (y==328 and x>=1105) or (y==369 and x>=1140) or (y==410 and x>=1185)):
                    print("Not using square 3", x,y)  
                    xs[i]=-99
                    ys[i]=-99                                         
                elif ((y<=1100 and x==21+20) or (y<=1000 and x==62+20) or (y<=450 and x==123) or y<=160 and x==164 ):
                    print("Not using square 4", x,y)  
                    xs[i]=-99
                    ys[i]=-99 
                elif ((y==697 and x==902) or (y==697 and x==943)) :
                    print("Not using square 5", x,y)  
                    xs[i]=-99
                    ys[i]=-99   
                elif ((y>=870 and x>=1220) or (y>=920 and x>=1180 and y<=1140)) :
                    print("Not using square 6", x,y)  
                    xs[i]=-99
                    ys[i]=-99                                                  
                else:
                    square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                    i_square=i_square+1
                i=i+1
            xs=xs[xs>0]
            ys=ys[ys>0]
        else:
            #print(len(xs),len(ys))
            for x,y in zip(xs,ys):  
                square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                i_square=i_square+1
        
        
               
        #plt.imshow(square_dat,cmap="flag")
       # plt.show()    
        #plt.savefig(nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v2.png',dpi=400)
       # plt.close()
        #plot_slice_squares(square_dat%10,np.array(ds_grid_info["lats_edge"][...]),\
        #                np.array(ds_grid_info["lons_edge"][...]),np.array(ds_grid_info["lats_mid"][...]),
        #                np.array(ds_grid_info["lons_mid"][...]),nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v3.png',\
        #               'i',ds_grid_info["plot_area"].data,cmap="tab20b",use_vmin=-1,use_vmax=10)

        
       # plt.show()
        #plt.close()
        use_title=""+str(n_inner)+" ("+str(len(xs))+")"
        outfile="/gws/nopw/j04/cehhmf/hmf/NFLICS/plots/square_plots/"+domain+"_ninner"+str(n_inner)+"_total"+str(len(xs))+".png"
        plot_slice_squares_empty(square_dat,lats_edge,lons_edge,lats_mid, lons_mid, outfile,\
                        'l',[-40,0,0,60],[],0,use_title,-100,15,1,cmap="Greys")
       
        rect_ids=[]
        add_inner=(n_inner-1)/2                
        for midx,midy in zip(xs,ys):                     
            rect_pt=[max(int(midy-add_inner),0),max(int(midx-add_inner),0),\
             min(int(midy+add_inner),ds_grid_info.attrs["ny"]),\
             min(int(midx+add_inner),ds_grid_info.attrs["nx"]+ds_grid_info.attrs["nx_dakarstrip"])]  
            rect_ids.append(str(rect_pt[0]).zfill(3)+"_"+str(rect_pt[1]).zfill(3)+"_"+\
                    str(rect_pt[2]).zfill(3)+"_"+str(rect_pt[3]).zfill(3))
        
        ds=xr.Dataset() #create dataset to save to netcdf for future use
        dimy=np.shape(ds_grid_info["lats_mid"].data)[0]
        dimx=np.shape(ds_grid_info["lats_mid"].data)[1]
        nsquares=np.nanmax(square_dat).astype(int)
        square_inds=np.arange(1,nsquares+1).astype(int)

        ds['squares_native']=xr.DataArray(square_dat, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['xs_mid']=xr.DataArray(xs, coords={'squares': square_inds },dims=['squares'])
        ds['ys_mid']=xr.DataArray(ys, coords={'squares': square_inds },dims=['squares'])
        ds['rect_id']=xr.DataArray(rect_ids, coords={'squares': square_inds},dims=['squares'])

        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path=squares_file,mode='w', encoding=enc, format='NETCDF4')    
    else:
        #print("loading squares file")
        ds=xr.open_dataset(squares_file)
        #print("loaded squares file")
    #print(ds)
    return(ds)

def process_squares_orig(ds_grid_info,n_inner,inds,weights,new_shape,nflics_base):
    squares_file=nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v2.nc'

    if not os.path.exists(squares_file):
	    # Calculate which database files are needed
    #-------------------------------------------------------------------------------	
        print("calculating squares file")
        inds_squares=np.indices(np.shape(ds_grid_info["lats_mid"].data))
        inds_select=(inds_squares[1][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner],inds_squares[0][n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner])
        xs=inds_select[0].astype(float)
        ys=inds_select[1].astype(float)
        toplt=np.copy(ds_grid_info["dfc_km"].data)
        toplt[toplt<0]=np.nan
        dfc_km_deg=np.copy(ds_grid_info["dfc_km"].data)[n_inner:-n_inner:n_inner,n_inner:-n_inner:n_inner]
        dfc_km_deg[np.where(dfc_km_deg<0)]=np.nan

        xs[np.where(np.isnan(dfc_km_deg)==True)]=np.nan
        ys[np.where(np.isnan(dfc_km_deg)==True)]=np.nan

        xs=xs[~np.isnan(xs)].astype("int")
        ys=ys[~np.isnan(ys)].astype("int")

        #plt.contourf(toplt)
        #plt.colorbar()
        #plt.scatter(xs.flatten(),ys.flatten(),marker=".",color="red")
        #plt.savefig()

        square_dat=np.zeros(np.shape(ds_grid_info["lats_mid"].data))*np.nan
        i_square=1

        for x,y in zip(xs,ys):
            if x== 164 and y!=246:
                square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-20-int(n_inner/2): x+int(n_inner/2)]=i_square
                i_square=i_square+1
            else:
                square_dat[y-1-int(n_inner/2):y+int(n_inner/2),x-1-int(n_inner/2): x+int(n_inner/2)]=i_square
                i_square=i_square+1

        plt.imshow(square_dat,cmap="flag")
        #plt.show()    
        plt.savefig(nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_v2.png',dpi=400)
        plt.close()
        #fix interpolation issues
        square_dat_5km=np.rint(uinterp.interpolate_data(square_dat, inds, weights, new_shape))
        tofix=np.where(np.diff(square_dat_5km,axis=0)>0)
        filx=tofix[0]+2
        fily=tofix[1]
        filval=(filx,fily)
    
        square_dat_5km[tofix]=square_dat_5km[filval]        
        square_dat_5km[136,292]=56

        plt.imshow(square_dat_5km,cmap="flag")
        plt.savefig(nflics_base+"/geoloc_grids/msg_rect_ALLhr_ninner"+str(n_inner)+'_5km_v2.png',dpi=400)
        plt.close()
        
        rect_ids=[]
        add_inner=(n_inner-1)/2

        for midx,midy in zip(xs,ys):
            rect_pt=[max(int(midy-add_inner),0),max(int(midx-add_inner),0),\
             min(int(midy+add_inner),ds_grid_info.attrs["ny"]),\
             min(int(midx+add_inner),ds_grid_info.attrs["nx"])]  
            rect_ids.append(str(rect_pt[0]).zfill(3)+"_"+str(rect_pt[1]).zfill(3)+"_"+\
                    str(rect_pt[2]).zfill(3)+"_"+str(rect_pt[3]).zfill(3))

        ds=xr.Dataset() #create dataset to save to netcdf for future use
        dimy=np.shape(ds_grid_info["lats_mid"].data)[0]
        dimx=np.shape(ds_grid_info["lats_mid"].data)[1]
        dimy_5km=np.shape(square_dat_5km)[0]
        dimx_5km=np.shape(square_dat_5km)[1]
        nsquares=np.nanmax(square_dat).astype(int)
        square_inds=np.arange(1,nsquares+1).astype(int)

        ds['squares_native']=xr.DataArray(square_dat, coords={'ys_mid': range(dimy) , 'xs_mid': range(dimx)},dims=['ys_mid', 'xs_mid']) 
        ds['squares_5km']=xr.DataArray(square_dat_5km, coords={'ys_5km': range(dimy_5km) , 'xs_5km': range(dimx_5km)},dims=['ys_5km', 'xs_5km']) 
        ds['xs_mid']=xr.DataArray(xs, coords={'squares': square_inds },dims=['squares'])
        ds['ys_mid']=xr.DataArray(ys, coords={'squares': square_inds },dims=['squares'])
        ds['rect_id']=xr.DataArray(rect_ids, coords={'squares': square_inds},dims=['squares'])

        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path=squares_file,mode='w', encoding=enc, format='NETCDF4')    
    else:
        #print("loading squares file")
        ds=xr.open_dataset(squares_file)
        #print("loaded squares file")
    #print(ds)
    return(ds)

###########################################################
#  function to plot the NFLICS maps 
###########################################################

def plot_lst_adjustment_factors(data,lats,lons,lats_mid,lons_mid,outfile,res,plot_lims,use_cbr_title,cmap="hot",use_vmin=0,use_vmax=2,use_extend="min",use_title=""):
    plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
    np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

    minlat=plot_lims_p[0]
    maxlat=plot_lims_p[2]
    minlon=plot_lims_p[1]
    maxlon=plot_lims_p[3]

    fig, ax1 = plt.subplots()
    m = Basemap(projection='merc',ax=ax1,lat_0=0.,lon_0=0., resolution=res,
		    llcrnrlon=plot_lims[1],llcrnrlat=plot_lims[0],
		    urcrnrlon=plot_lims[3],urcrnrlat=plot_lims[2])
    X, Y = m(lons,lats)
    Xmid,Ymid=m(lons_mid,lats_mid)
    by=0.05
    bounds = np.arange(use_vmin,use_vmax+by, by)

    use_cmap=plt.get_cmap(cmap,len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds,len(bounds), use_cmap)
    
    
    #pc=m.contourf(X,Y,data,cmap='hot',levels=np.arange(15,70,5))
    #pc = m.pcolormesh(X,Y,data,cmap='hot',vmin=-80,vmax=cell_thresh)
    pc = m.pcolormesh(X,Y,data,cmap=use_cmap,norm=norm,vmin=use_vmin,vmax=use_vmax)

    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    cbr=plt.colorbar(pc,extend=use_extend,pad=0.08,shrink=0.49,aspect=10)        
    cbr.ax.set_title(use_cbr_title,fontsize=10)
    m.drawparallels(np.arange(0,25,5),labels=[1,0,0,1])
    m.drawmeridians(np.arange(-15,35,5),labels=[1,0,0,1])
    #cr=m.contour(Xmid,Ymid,blobs,colors='cyan',linewidths=2,linestyles="solid",levels=range(14,220,15))
    plt.title(use_title,pad=1,fontsize=10)   
    fig.set_size_inches(6,4)  
    plt.tight_layout(pad=1)

    if type(outfile)== str:
	    plt.savefig(outfile,dpi=400)
    else:
	    plt.show()
	    plt.close()

	
def plot_slice_cells_blobs(data,blobs,lats,lons,lats_mid,lons_mid,outfile,res,plot_lims,cell_thresh,use_cbr_title,sizes=False,\
	cmap="hot",cell_col="cyan",use_vmin=-80,use_vmax=-30,use_extend="min",use_title=""):


    

# chop full image to size
    plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
       np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

    plot_lims_p_blobs=[np.where(lats_mid[:,0]>plot_lims[0])[0][0],np.where(lons_mid[0,:]>plot_lims[1])[0][0],
       np.where(lats_mid[:,0]<plot_lims[2])[0][-1],np.where(lons_mid[0,:]<plot_lims[3])[0][-1]]

    #print(plot_lims_p)
    #print(plot_lims_p_blobs)
    data = data[plot_lims_p[0]:plot_lims_p[2],plot_lims_p[1]:plot_lims_p[3]]
    lats_data = lats[plot_lims_p[0]:plot_lims_p[2],plot_lims_p[1]:plot_lims_p[3]]
    lons_data = lons[plot_lims_p[0]:plot_lims_p[2],plot_lims_p[1]:plot_lims_p[3]]
    blobs= blobs[plot_lims_p_blobs[0]:plot_lims_p_blobs[2],plot_lims_p_blobs[1]:plot_lims_p_blobs[3]] 
    lats_blobs = lats_mid[plot_lims_p_blobs[0]:plot_lims_p_blobs[2],plot_lims_p_blobs[1]:plot_lims_p_blobs[3]]
    lons_blobs = lons_mid[plot_lims_p_blobs[0]:plot_lims_p_blobs[2],plot_lims_p_blobs[1]:plot_lims_p_blobs[3]]
   # data = data[plot_lims_p[0]:plot_lims_p[0]+lats.shape[0],plot_lims_p[1]:plot_lims_p[1]+lons.shape[1]] 
   # blobs= blobs[plot_lims_p[0]:plot_lims_p[0]+lats_mid.shape[0],plot_lims_p[1]:plot_lims_p[1]+lons_mid.shape[1]] 

    f=plt.figure()
    ax2 = f.add_subplot(111, projection=ccrs.PlateCarree())    
    ax2.set_extent([plot_lims[1],plot_lims[3],plot_lims[0],plot_lims[2]],crs=ccrs.PlateCarree())  

    by=5
    bounds = np.arange(use_vmin,use_vmax+by, by)

    use_cmap=plt.get_cmap(cmap,len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds,len(bounds), use_cmap)

    pc2=plt.pcolormesh(lons_data, lats_data, data, transform=ccrs.PlateCarree(),cmap=use_cmap,norm=norm)
    ax2.coastlines() 
    
    cbr=plt.colorbar(pc2,extend=use_extend,pad=0.08,shrink=0.49,aspect=10)        
    cbr.ax.set_title(use_cbr_title,fontsize=10)

    xl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='grey', linestyle='dashed')
    plt.grid(which='minor')
    
    xl.top_labels = False
    xl.right_labels = False
    xl.xlines = True
    xl.ylines=True
    
    # Countries
    ax2.add_feature(cartopy.feature.LAKES, linestyle='-', linewidth=0.25,edgecolor='navy',zorder=0)
    ax2.add_feature(cartopy.feature.RIVERS, linestyle='-', linewidth=0.25,edgecolor='navy')
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='--', linewidth=0.5,edgecolor="black")    

  
    if not cell_col=="none":
    	#cr=plt.contour(lons_mid,lats_mid,blobs,colors=cell_col,linewidths=2,linestyles="solid",levels=[0.5])

        cr=plt.contour(lons_blobs,lats_blobs,blobs,colors=cell_col,linewidths=2,linestyles="solid",levels=[0.5])        
    #if not cell_col=="none":
    #	cbr.add_lines(cr)
        
    plt.title(use_title,pad=1,fontsize=10)   
    f.set_size_inches(6,4)  
    plt.tight_layout(pad=1)

    if type(outfile)== str:
	    plt.savefig(outfile,dpi=400)
    else:
	    plt.show()
	    plt.close()

def OLD_plot_slice_cells_blobs(data,blobs,lats,lons,lats_mid,lons_mid,outfile,res,plot_lims,cell_thresh,use_cbr_title,sizes=False,\
	cmap="hot",cell_col="cyan",use_vmin=-80,use_vmax=-30,use_extend="min",use_title=""):

    plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
       np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

    minlat=plot_lims_p[0]
    maxlat=plot_lims_p[2]
    minlon=plot_lims_p[1]
    maxlon=plot_lims_p[3]


    fig, ax1 = plt.subplots()
    m = Basemap(projection='merc',ax=ax1,lat_0=0.,lon_0=0., resolution=res,
		    llcrnrlon=plot_lims[1],llcrnrlat=plot_lims[0],
		    urcrnrlon=plot_lims[3],urcrnrlat=plot_lims[2])
    X, Y = m(lons,lats)
    Xmid,Ymid=m(lons_mid,lats_mid)
    by=5
    bounds = np.arange(use_vmin,use_vmax+by, by)

    use_cmap=plt.get_cmap(cmap,len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds,len(bounds), use_cmap)

    data = data[plot_lims_p[0]:plot_lims_p[2],plot_lims_p[1]:plot_lims_p[3]] 
    blobs= blobs[plot_lims_p[0]:plot_lims_p[2],plot_lims_p[1]:plot_lims_p[3]] 

    #pc=m.contourf(X,Y,data,cmap='hot',levels=np.arange(15,70,5))
    #pc = m.pcolormesh(X,Y,data,cmap='hot',vmin=-80,vmax=cell_thresh)
    #pc = m.pcolormesh(X,Y,data,cmap=use_cmap,norm=norm,vmin=use_vmin,vmax=use_vmax)
    pc = m.pcolormesh(X,Y,data,cmap=use_cmap,norm=norm,vmin=use_vmin,vmax=use_vmax,shading='auto')
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    cbr=plt.colorbar(pc,extend=use_extend,pad=0.08,shrink=0.49,aspect=10)        
    cbr.ax.set_title(use_cbr_title,fontsize=10)
    m.drawparallels(np.arange(0,25,5),labels=[1,0,0,1])
    m.drawmeridians(np.arange(-15,35,5),labels=[1,0,0,1])
    #cr=m.contour(Xmid,Ymid,blobs,colors='cyan',linewidths=2,linestyles="solid",levels=range(14,220,15))
    if not cell_col=="none":
    	cr=m.contour(Xmid,Ymid,blobs,colors=cell_col,linewidths=2,linestyles="solid",levels=[0.5])
    #print(["SIZES:",sizes])
    if sizes:
        
        cx=1700#650
        cy=470#370
        
        cmap = mpl.cm.get_cmap('gray', 10)    # 11 discrete colors

        for i_size,rsize in enumerate(sizes):
            rsize=int(rsize)
            #print(rsize)
            poly=mpl.patches.Polygon([(X[cy-rsize,cx-rsize],Y[cy-rsize,cx-rsize]),
                                      (X[cy+rsize,cx-rsize],Y[cy+rsize,cx-rsize]),
                                      (X[cy+rsize,cx+rsize],Y[cy+rsize,cx+rsize]),
                                      (X[cy-rsize,cx+rsize],Y[cy-rsize,cx+rsize])],
                                       fill=False,edgecolor=cmap(i_size))
            ax=plt.gca()
            ax.add_patch(poly) 


    #m.contour(Xmid,Ymid,cell_data,levels=[0.05],colors=['purple'],linewidths=1)
    #if not cell_col=="none":
    #	cbr.add_lines(cr)
    plt.title(use_title,pad=1,fontsize=10)   
    fig.set_size_inches(6,4)  
    plt.tight_layout(pad=1)

    if type(outfile)== str:
	    plt.savefig(outfile,dpi=400)
    else:
	    plt.show()
	    plt.close()

###########################################################
#  function to plot the risk maps
###########################################################
def plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,vectoplt,outfile,plot_type="risk",use_title="",use_cbr_title="",use_ylab=""):

    toplt=np.zeros(np.shape(ds_dakar["dakar_lons"]))-999
    toplt[np.where(ds_dakar["dakar_communes"]>=0)]=0
    for c,dat_toplt in zip(ds_dakar["commune_ind"],vectoplt):
        toplt[np.where(ds_dakar["dakar_communes"]==c)]=dat_toplt
    toplt_m=ma.masked_values(toplt,-999)
    toplt_poly_dakar_m=ma.masked_values(grid_poly_ds["poly_grid"],-999)
    commune_edges=ds_dakar["dakar_communes"].data
    commune_edges[np.where(~np.isfinite(commune_edges))]=-1
    #sort the colour map
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(204/255, 255/255, 153/255), (255/255, 255/255, 50/255),(254/255,151/255,0), (180/255, 0, 0)]
    cmap=LinearSegmentedColormap.from_list("risk_matrix",colors,4)
    fig,ax=plt.subplots(figsize=(6, 4))
    m = Basemap(projection='merc', ax=ax, lat_0=0.,lon_0=0., resolution='l',
            llcrnrlon=-17.55,llcrnrlat=14.6,urcrnrlon=-17.1,urcrnrlat=14.9)
    X, Y = m(ds_grid_info["lons_edge"].data,ds_grid_info["lats_edge"].data)
    Xt, Yt = m(ds_dakar["dakar_lons"].data,ds_dakar["dakar_lats"].data)
    #pc = m.pcolormesh(X,Y,np.zeros(np.shape(grid_poly_ds["poly_grid"])),cmap='Blues',vmin=-4,vmax=80,lw=2)
    pc = m.pcolormesh(X,Y,toplt_poly_dakar_m,cmap='Greys',vmin=295,vmax=317,lw=2)
    pc = m.pcolormesh(Xt,Yt,toplt_m,vmin=0,vmax=4,cmap=cmap,lw=2)

    cbr=plt.colorbar(pc,extend="neither",pad=0.1,shrink=0.6,aspect=10) 
    cbr.set_ticks([0.5,1.5,2.4,3.5])
    if plot_type=="vul":
        cbr.ax.set_yticklabels(use_ylab)
    else:
        cbr.ax.set_yticklabels(["V. Low","Low","Medium","High"])
    cbr.ax.set_title(use_cbr_title,fontsize=10)
    pc = m.contour(Xt,Yt,commune_edges,levels=np.arange(-1.5,50,0.5),colors="black")
    plt.title(use_title,pad=0,fontsize=10)
    ax.text(0.0, 0.0, 'Region for convective structure probability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="darkgrey")
    ax.text(0.0, -0.2, 'Click on Commune to see additional information and Flood Risk matrix', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
    #ax.text(0.0, 0.0, 'Region for convective structure probability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="darkgrey")ax.text(0.3, -0.1, 'Estimated surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
    if plot_type=="vul":
        ax.text(0.0, -0.1, 'Vulnerability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black",weight="bold")
        ax.text(0.3, -0.1, 'Estimated surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.7, -0.1, 'Wet surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(1.0, -0.1, 'Dry surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
    elif plot_type=="risk_ante":
        ax.text(0.0, -0.1, 'Vulnerability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.3, -0.1, 'Estimated surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black",weight="bold")
        ax.text(0.7, -0.1, 'Wet surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(1.0, -0.1, 'Dry surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
    elif plot_type=="risk_wet":
        ax.text(0.0, -0.1, 'Vulnerability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.3, -0.1, 'Estimated surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.7, -0.1, 'Wet surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black",weight="bold")
        ax.text(1.0, -0.1, 'Dry surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
    elif plot_type=="risk_dry":
        ax.text(0.0, -0.1, 'Vulnerability', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.3, -0.1, 'Estimated surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(0.7, -0.1, 'Wet surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black")
        ax.text(1.0, -0.1, 'Dry surface', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,color="black",weight="bold")
    fig.set_size_inches(6,4)  
    plt.tight_layout(pad=1)
    ax.set_position((0.02, 0.1684, 0.6758, 0.6988))
    if type(outfile)== str:
        plt.savefig(outfile,dpi=400)
    else:
        plt.show()
        plt.close()
###########################################################
#  function to plot the LST and core maps
###########################################################
def plot_slice_lst(data_lst,data_lats,data_lons,blobs,lats,lons,lats_mid,lons_mid,outfile,res,plot_lims,cell_thresh,use_cbr_title,sizes=False,\
	cmap="hot",cell_col="cyan",use_vmin=-80,use_vmax=-30,use_extend="min",use_title=""):

    plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
       np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

    minlat=plot_lims_p[0]
    maxlat=plot_lims_p[2]
    minlon=plot_lims_p[1]
    maxlon=plot_lims_p[3]

    fig, ax1 = plt.subplots()
    m = Basemap(projection='merc',ax=ax1,lat_0=0.,lon_0=0., resolution=res,
		    llcrnrlon=plot_lims[1],llcrnrlat=plot_lims[0],
		    urcrnrlon=plot_lims[3],urcrnrlat=plot_lims[2])
    X, Y = m(data_lons,data_lats)
    Xmid,Ymid=m(lons_mid,lats_mid)
    by=2
    bounds = np.arange(use_vmin,use_vmax+by, by)


    use_cmap=plt.get_cmap(cmap,len(bounds))

    norm = mpl.colors.BoundaryNorm(bounds,len(bounds), use_cmap)
    m.drawlsmask(land_color="darkgrey")
    pc = m.pcolormesh(X,Y,data_lst,cmap=use_cmap,norm=norm,vmin=use_vmin,vmax=use_vmax)

    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)

    cbr=plt.colorbar(pc,extend=use_extend,pad=0.08,shrink=0.49,aspect=10)        
    cbr.ax.set_title(use_cbr_title,fontsize=10)
    m.drawparallels(np.arange(0,25,5),labels=[0,0,0,1])
    m.drawmeridians(np.arange(-15,35,5),labels=[0,0,0,1])

    use_lwd=np.repeat(np.arange(1,0.1,-0.1),4)
    use_colors = plt.cm.cool(np.linspace(0,1,np.shape(blobs)[0]+4))

    if not cell_col=="none":
        for i_time in range(0,np.shape(blobs)[0],1)[::-1]: 
            cr=m.contour(Xmid,Ymid,blobs[i_time,...],colors=[use_colors[i_time+4]],linewidths=0.8,linestyles="solid",levels=[0.5])
        cr=m.contour(Xmid,Ymid,blobs[i_time,...],colors=[cell_col],linewidths=0.9,linestyles="solid",levels=[0.5])

    xpos=-0.08
    ax1.text(xpos, 0.9, 'T-0h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[0])
    ax1.text(xpos, 0.8, 'T-1h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[8])
    ax1.text(xpos, 0.7, 'T-2h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[12])
    ax1.text(xpos, 0.6, 'T-3h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[16])
    ax1.text(xpos, 0.5, 'T-4h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[20])
    ax1.text(xpos, 0.4, 'T-5h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[24])    
    ax1.text(xpos, 0.3, 'T-6h', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color=use_colors[28])
    ax1.text(-0.13, 0.0, 'Cloud \n mask', va='bottom', ha='left',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color="darkgrey")
    if use_vmin>-50:
        ax1.text(1.05, 0.75, 'Dry', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color="Brown")
        ax1.text(1.05, 0.15, 'Wet', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax1.transAxes,color="Purple")
    plt.title(use_title,pad=1,fontsize=10)  
    fig.set_size_inches(6,4)
    plt.tight_layout(pad=1)

    if type(outfile)== str:
	    plt.savefig(outfile,dpi=400)
    else:
	    plt.show()
	    plt.close()

###########################################################
#  function to generate the polygon-averaged scaling factors
###########################################################
def get_poly_avgs(plot_area,base,scaling_factor_grid):
  
    #get information from pre-created grid of polygons
    poly_str=str(plot_area).replace(" ","").replace(",","_").replace("-","n")[1:-1]
    #print(poly_str)
    #print(base+'shape_files/wca_admbnda_adm1_ocha/wca_admbnda_adm1_ocha'+poly_str+'.nc')
    grid_poly_ds=xr.open_dataset(base+'shape_files/wca_admbnda_adm1_ocha/wca_admbnda_adm1_ocha'+poly_str+'.nc')
    grid_poly=grid_poly_ds['poly_grid'].data
    poly_names=grid_poly_ds['names_list'].coords['names'].data
    poly_vals=grid_poly_ds['names_list'].data   
    grid_poly_ds.close()

    poly_scaling = {}
    for reg_ind in np.unique(grid_poly):     #Loop over regions.
        poly_mask=np.ones(np.shape(scaling_factor_grid))  
        if reg_ind>=0: # valid index            
            poly_mask[np.where(grid_poly==reg_ind)]= 0
            poly_scaling[reg_ind]=np.nanmean(scaling_factor_grid[np.where(poly_mask==0)])
            
    return poly_scaling



###########################################################
#  class to run the core calculations isolated from the main code
###########################################################
class extendedCoreCalc(multiprocessing.Process):
    def __init__(self,ds_grid_info_ex,data_all_ex,rt_lats_ex,rt_lons_ex,plot_area_ex,use_times,tnow,scratchdir,do_geotiff,plotdir,vis_data,ds_grid_info_ex_3km):
        super(extendedCoreCalc,self).__init__()
        self.ds_grid_info_ex=ds_grid_info_ex
        self.ds_grid_info_ex_3km=ds_grid_info_ex_3km
        self.data_all_ex=data_all_ex
        self.scratchdir=scratchdir
        self.use_times = use_times
        self.rt_lons_ex=rt_lons_ex
        self.rt_lats_ex=rt_lats_ex
        self.plot_area_ex = plot_area_ex
        self.tnow = tnow
        self.do_geotiff = do_geotiff
        self.plotdir = plotdir
        self.vis_data = vis_data
    def run(self):

        #rt_lats_ex,rt_lons_ex,data_all_ex=get_rt_data(rt_file)
        grid_lims_rt_ex=[np.where(self.rt_lats_ex[:,0]>self.plot_area_ex[0])[0][0],np.where(self.rt_lons_ex[0,:]>self.plot_area_ex[1])[0][0],
        np.where(self.rt_lats_ex[:,0]<self.plot_area_ex[2])[0][-1],np.where(self.rt_lons_ex[0,:]<self.plot_area_ex[3])[0][-1]]
        self.data_all_ex=self.data_all_ex[grid_lims_rt_ex[0]:grid_lims_rt_ex[2],grid_lims_rt_ex[1]:grid_lims_rt_ex[3]][:-1,]
        self.vis_data = self.vis_data[grid_lims_rt_ex[0]:grid_lims_rt_ex[2],grid_lims_rt_ex[1]:grid_lims_rt_ex[3]][:-1,]
        
        missing_vis = [np.all(np.isnan(grid)) for grid in self.vis_data]
        missing_ex = [np.all(np.isnan(grid)) for grid in self.data_all_ex]
        lats_edge_ex=np.array(self.ds_grid_info_ex['lats_edge'][...])  
        lons_edge_ex=np.array(self.ds_grid_info_ex['lons_edge'][...])
        lats_mid_ex=np.array(self.ds_grid_info_ex['lats_mid'][...])
        lons_mid_ex=np.array(self.ds_grid_info_ex['lons_mid'][...])
        blobs_lons_ex=np.array(self.ds_grid_info_ex['blobs_lons'][...])
        blobs_lats_ex=np.array(self.ds_grid_info_ex['blobs_lats'][...])
        blobs_lons_ex_3km=np.array(self.ds_grid_info_ex_3km['blobs_lons'][...])
        blobs_lats_ex_3km=np.array(self.ds_grid_info_ex_3km['blobs_lats'][...])

        inds_ex, weights_ex, new_shape_ex=uinterp.interpolation_weights( lons_mid_ex, lats_mid_ex,blobs_lons_ex, blobs_lats_ex)
        inds_ex_3km, weights_ex_3km, new_shape_ex_3km=uinterp.interpolation_weights( lons_mid_ex, lats_mid_ex,blobs_lons_ex_3km, blobs_lats_ex_3km)
        inds_2_ex, weights_2_ex, new_shape_2_ex=uinterp.interpolation_weights(blobs_lons_ex, blobs_lats_ex, lons_mid_ex, lats_mid_ex)
        data_all_keep_ex=np.copy(self.data_all_ex)
        data_interp_ex=uinterp.interpolate_data(self.data_all_ex, inds_ex, weights_ex, new_shape_ex)

        
        #run wavelet analysis
        data_blobs_date_ex=run_powerBlobs.wavelet_analysis(np.copy(data_interp_ex[:,:]), blobs_lons_ex, blobs_lats_ex, self.use_times[0],
                     "",data_resolution=5)
        com_loc_ex=np.where(data_blobs_date_ex["blobs"].values<0)    #power maxima of each convective structure
        blobs_interp_ex=uinterp.interpolate_data(data_blobs_date_ex["blobs"].values, inds_2_ex, weights_2_ex, new_shape_2_ex)
        blobmask_interp_ex = np.zeros(np.shape(data_blobs_date_ex["blobs"].values))
        blobmask_interp_ex[data_blobs_date_ex["blobs"].values>0]=1
        
        usemask_ex=np.ones(np.shape(blobs_interp_ex))
        usemask_ex[(blobs_interp_ex!=0)& ~np.isnan(blobs_interp_ex)]=0
        data_all_m_ex=ma.masked_array(self.data_all_ex,mask=usemask_ex)
        dimy_ex=np.shape(self.ds_grid_info_ex["lats_mid"].data)[0]
        dimx_ex=np.shape(self.ds_grid_info_ex["lats_mid"].data)[1]
        ds_ex=xr.Dataset()
        ds_ex['cores']=xr.DataArray(data_all_m_ex[:], coords={'ys_mid': range(dimy_ex) , 'xs_mid': range(dimx_ex)},dims=['ys_mid', 'xs_mid']) 
        ds_ex.attrs['time']=self.tnow
        ds_ex.attrs['grid']="NFLICS msg cutout extended"
        ds_ex.attrs['missing']="nan"
        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds_ex.data_vars}
        #print(self.scratchdir)
        ds_ex.to_netcdf(path=self.scratchdir+"/Convective_struct_extended_"+self.tnow+"_000.nc",\
                 mode='w', encoding=enc, format='NETCDF4')
        # archive to prj
        if self.do_geotiff:
            rasPath = self.plotdir+"/Observed_CTT_"+self.tnow+"_extended.tif"
            rasPath_3857 = self.plotdir+"/Observed_CTT_"+self.tnow+"_extended_3857.tif"
            make_geoTiff([data_interp_ex],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
            os.system('rm '+rasPath)
            #visible channel
            if not np.isnan(self.vis_data).all():
                rasPath= self.plotdir+"/ch1_X_"+self.tnow+"_pc.tif"
                rasPath_3857= self.plotdir+"/ch1_X_"+self.tnow+"_pc_3857.tif"
                data_interp_vis=uinterp.interpolate_data(self.vis_data, inds_ex_3km, weights_ex_3km, new_shape_ex_3km)
                #xx=np.log(self.vis_data[:].astype(float))
                #xx= np.log(data_interp_vis.astype(float))
                xx= data_interp_vis.astype(float)
                xx_shifted = xx -np.nanmin(xx)
                yy= xx_shifted*100/max(np.nanmax(xx_shifted),0.000001)
                make_geoTiff([yy],rasPath,reprojFile=rasPath_3857,extended=True,is_vis=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
                os.system('rm '+rasPath)

            # convective struscure

            rasPath = self.plotdir+"/Observed_ConStruct_"+self.tnow+"_extended.tif"
            rasPath_3857 = self.plotdir+"/Observed_ConStruct_"+self.tnow+"_extended_3857.tif"
            sx = ndimage.sobel(blobmask_interp_ex, axis=0, mode='constant')
            sy = ndimage.sobel(blobmask_interp_ex, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            sob_filter = 2  # original was 2
            sob[sob<=sob_filter] = 0
            sob[sob>sob_filter] = 1
            make_geoTiff([sob],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)
            os.system('rm '+rasPath)

            # past cores
            dt_now=datetime.datetime.strptime(self.tnow,"%Y%m%d%H%M")
            use_core_times=pd.date_range(dt_now,dt_now-pd.Timedelta(minutes=60*6),freq="-15min") 
            past_cores=[]
            past_times=[]
            scratchRoot = '/'.join(self.scratchdir.split('/')[:4])
            #scratchRoot = self.scratchdir
            for core_time in list(use_core_times):
            
                scratchdir_cs=os.path.join(scratchRoot,str(core_time.year),str(core_time.month).zfill(2),\
                           str(core_time.day).zfill(2),str(core_time.hour).zfill(2)+str(core_time.minute).zfill(2))

                
                #scratchfile=self.scratchdir+"/Convective_struct_extended_"+str(core_time).replace("-"," ").replace(":","").replace(" ","")[:-2]+"_000.nc"
                scratchfile=scratchdir_cs+"/Convective_struct_extended_"+str(core_time).replace("-"," ").replace(":","").replace(" ","")[:-2]+"_000.nc"
                #print(scratchfile)
                #print(os.path.exists(scratchfile))
                if os.path.exists(scratchfile):
                    
                    core_ds=xr.open_dataset(scratchfile)   
                    past_cores.append(core_ds["cores"].data)
                    past_times.append(core_ds.attrs["time"])
                    core_ds.close()
                else:  
                    past_cores.append(np.zeros(np.shape(lats_mid_ex))*np.nan)
                    past_times.append(np.nan)   

        
            past_cores=np.array(past_cores)

            past_cores[np.where(past_cores<0)]=1
            past_cores[np.where(np.isnan(past_cores))]=0
            
            # combine into 1 np array
            allPastCores = np.zeros(sob.shape) # ConStruct from before
            #for icore in range
            for icore in range(1,np.shape(past_cores)[0],2)[::-1]:
                ipast_Core = uinterp.interpolate_data(past_cores[icore,...], inds_ex, weights_ex, new_shape_ex)
                
                 #apply sobel filter to hollow out
                sx = ndimage.sobel(ipast_Core, axis=0, mode='constant')
                sy = ndimage.sobel(ipast_Core, axis=1, mode='constant')
                #sx = ndimage.sobel(past_cores[icore,...], axis=0, mode='constant')
                #sy = ndimage.sobel(past_cores[icore,...], axis=1, mode='constant')
                sob = np.hypot(sx, sy)
                sob_filter = 2  # original was 2
                sob[sob<=sob_filter] = 0
                sob[sob>sob_filter] = 1
                 #add to main array  
                allPastCores[sob==1] = (icore+1)/4.  # number of minutes, since this is plotting every half hour
            #shift onto the same grid as the ConStruct grid
            #allPastCores=uinterp.interpolate_data(allPastCores, inds_ex, weights_ex, new_shape_ex)
            
            rasPath = self.plotdir+"/PastCores_"+self.tnow+".tif"
            rasPath_3857 = self.plotdir+"/PastCores_"+self.tnow+"_3857.tif"
            make_geoTiff([allPastCores],rasPath,reprojFile=rasPath_3857,extended=True,v_maj=version_maj['full'],v_min=version_min['full'],v_submin=version_submin['full'],trim=True)    
            os.system('rm '+rasPath)   


            #rasPath = self.plotdir+"/Observed_CTT_ConStruct_"+self.tnow+"_extended.tif"
            #rasPath_3857 = self.plotdir+"/Observed_CTT_ConStruct_"+self.tnow+"_extended_3857.tif"
            #make_geoTiff([data_interp_ex,blobmask_interp_ex],rasPath,reprojFile=rasPath_3857,extended=True)
            #os.system('rm '+rasPath)


            #make_geoTiff([data_all_keep_ex,data_all_m_ex],rasPath,reprojFile=rasPath_3857,extended=True)


def fit_curve_poly(x, y, h):
    # lower order polyfit for nighttime/morning curves to reduce noise
    if (h <= 10) | (h >= 23):
        order = 2
    else:
        order = 6
    z = np.polyfit(x, y, order)

    return np.poly1d(z)


def pcalc_run(h, LSTA_array, table_path):
    """
    :param h: Hour to calculate probability for
    :param LSTA_array: Any LSTA array, numpy array or data array (2D)
    :param table_path: directory path to pre-calculated hourly LSTA probability tables
    :return: Array with probability factors of LSTA-array shape
    """
    tab = pd.read_table(table_path + os.sep + 'BigDomain_LSTAprobability_17-0W_9-20N_' + str(h).zfill(2) + '.csv', delimiter=',')

    f = fit_curve_poly(tab['bins'], tab['factor'], h)
    probability_values = f(LSTA_array)

    # Probability adjustment only valid within reference LSTA ranges. Extremes are set to last valid value.
    p10, p90 = np.percentile(tab['bins'],[10,90])
    probability_values[(LSTA_array>=p90)] = f(p90)
    probability_values[(LSTA_array<=p10)]= f(p10)

    return probability_values

###########################################################
#  function to plot the timeseries of a set of points over the nowcast period
###########################################################

def write_site_cores(csvfile,places,locs,anyblobs):
    with open(csvfile,'w') as ff:
        ff.write('Location,Latitude,Longitude,isCore\n')
        for site in range(len(places)):
            ff.write(','.join([places[site],str(locs[site][0]),str(locs[site][1]),str(int(anyblobs[site]))])+'\n') 

def plt_nflics_ts(forigin,ts,places,locs,plotfile,csvfile,filters,nhrs=6,fixed=False,writeType='w'):

    
        

    # 1 make the csv
    with open(csvfile,writeType) as ff:  
         hrList= [0,1,2,3,4,5,6,8,10,12]            
        #ff.write('Location,Latitude,Longitude,'+','.join(['Filter_T+'+str(ihr).zfill(2)+'hr' for ihr in range(nhrs+1)])+','+','.join(['Prob_T+'+str(ihr).zfill(2)+'hr' for ihr in range(nhrs+1)])+'\n')
         if writeType=='w':
             ff.write('Location,Latitude,Longitude,'+','.join(['Filter_T+'+str(ihr).zfill(2)+'hr' for ihr in hrList])+','+','.join(['Prob_T+'+str(ihr).zfill(2)+'hr' for ihr in hrList])+'\n')

         for site in range(len(places)):
            
            ff.write(','.join([places[site]]+[str(i) for i in locs[site]]+[str(round(i,2)) for i in filters]+[str(i) for i in ts[site]])+'\n')


    # 2 plot it 
    # hours (hard coded to 6)
    t = np.arange(nhrs+1)  
    if not fixed:  # only plot for varibale pixel (ie database, not portal)
        ForcOrigin = forigin
        nlocations = len(places)
        for iax in range(nlocations):
            ax = plt.subplot(nlocations,1,1+iax)
            plt.plot(t, ts[iax])
            plt.setp(ax.get_xticklabels(), fontsize=6)
            plt.setp(ax.get_yticklabels(), fontsize=6)
            plt.text(0.1, 100, places[iax]+' ('+str(locs[iax][1])+'E,'+str(locs[iax][0])+'N)', fontsize=8)
            plt.xlim(0.0, 6.0)
            plt.ylim(0.0, 120)
            if iax==nlocations-1:
                ax.set_xlabel('Leadtime (hours)', fontsize=8)
            if iax==0:
                ax.set_title('Nowcast Origin: '+str(ForcOrigin), fontsize=10)
       	    ax.set_ylabel('Probability (%)', fontsize=8)

      #fig.set_size_inches(6,4)    
	
        figure = plt.gcf()
        figure.set_size_inches(5,8)
        plt.tight_layout(pad=1)
        if type(plotfile)== str:
            plt.savefig(plotfile,dpi=400)




def make_geoTiff(data,rasFile,doReproj = True,origEPSG='4326',newEPSG='3857',reprojFile='test.tif',extended=False,isLST=False,is_vis=False,is_nowcast=False,trim=False,v_maj=3,v_min=1,v_submin=1,subdomain=''):

#   extended = FULL domain over all SSA
#   domain = ''  but set to wa or sa for sub domains. 
#                only gets used if extended=False, as then will need to pick a subdomain
#   CTT, past cores, VIS all over full domain
#   nowcasts, LSTA over subdomain
    versionstr = str(v_maj)+'.'+str(v_min)+'.'+str(v_submin)
    nbands = len(data)
    # SET TRANSFORM
    if extended: # over the full domain 
        dat_type = str(data[0].dtype)
        if is_vis: # visisble rad
           # transform = rasterio.transform.from_origin(-23.0,19.99001,0.026949456,0.026949456)
           # transform = rasterio.transform.from_origin(-20.0,19.99001,0.026949456,0.026949456)
            transform = rasterio.transform.from_origin(-27,27,0.026949456,0.026949456)
        else: # CTT, cores, past cores
            #transform = rasterio.transform.from_origin(-23.0,19.99001,0.04491576,0.04491576)
           # transform = rasterio.transform.from_origin(-20.0,19.99001,0.04491576,0.04491576)
            transform = rasterio.transform.from_origin(-27,27,0.04491576,0.04491576)
        #transform = rasterio.transform.from_origin(-21.6588,20.2517,0.028,0.028)
        #transform = rasterio.transform.from_bounds(-21.6588,4.01137,32.5176,20.2517,data[0].shape[1],data[0].shape[0])
    else: # over subdomain 
        if isLST: # NOT SET UP FOR WA AND SA SUBDOMAINS YET
            dat_type = str(data[0].dtype)
            transform  =rasterio.transform.from_origin(-21.6239,20.259,0.0305,0.0305)
        elif is_nowcast:
            dat_type='float32'
            if subdomain=='wa':
            ####transform = rasterio.transform.from_origin(-20.0,19.9925,0.04491576,0.04491576)#TEST
                transform = rasterio.transform.from_origin(-23.0,19.99001,0.04491576,0.04491576)# WA
            elif subdomain=='sadc':
                transform = rasterio.transform.from_origin(8.0,0.04365,0.04491576,0.04491576)#NOT RIGHT FOR SA YET		
            
        else: #NOT USED???
            transform = rasterio.transform.from_origin(-20.1836,19.908,0.029,0.029)
            dat_type = 'float32'
            #transform = rasterio.transform.from_bounds(-21.6588,19.908,0.029,0.02)
    rasImage = rasterio.open(rasFile,'w',driver='GTiff',
                           height=data[0].shape[0],width=data[0].shape[1],
                           count=nbands,dtype=dat_type,
                           crs = 'EPSG:'+str(origEPSG),
                           transform = transform)
    for ix,Image in enumerate(data):
        rasImage.write(np.flipud(Image[:]),ix+1)
    rasImage.close()
    # add metadata version
    gdaledit= '/users/hymod/stewells/miniconda2/envs/py37/bin/gdal_edit.py'
    #cmd = ['gdal_edit.py',rasFile,'-mo',"VERSION=Version 1.0"]
    #print(versionstr)
    os.system("gdal_edit.py -mo \"xmp_Version_Version="+versionstr+"\" "+rasFile)
    #subprocess.call(cmd,shell=True)
	
    if trim: # only required for extended
#crop parameters       
        upper_left_x = -20
        upper_left_y = 26
        lower_right_x = 55
        lower_right_y = -36.5
        window = (upper_left_x,upper_left_y,lower_right_x,lower_right_y)
        rasFile2 = rasFile[:-4]+'_chop.tif'
        gdal.Translate(rasFile2, rasFile, projWin = window)
        os.system('mv '+rasFile2+' '+rasFile)
        rasFile2 = rasFile
    else:
       rasFile2 = rasFile

	
	
	
    if doReproj:
        ds = gdal.Warp(reprojFile, rasFile2, srcSRS='EPSG:'+str(origEPSG), dstSRS='EPSG:'+str(newEPSG), format='GTiff',creationOptions=["COMPRESS=LZW"])
        ds = None  
    








def get_dummy_ssa_latlon():
    nx=2268
    ny=2080
            #print([nx,ny])
    filename='/mnt/prj/nflics/SSA_data/grads/lat_lon_2268_2080.gra'
    #(filename)	
    if(os.path.isfile(filename)):
        gen_ints = array.array("f")
        gen_ints.fromfile(open(filename, 'rb'), os.path.getsize(filename) // gen_ints.itemsize)
        #print(np.array(gen_ints).shape)
        data_strip=np.array(gen_ints).reshape(2,ny,nx)#-173  #data is rows, columns
    else:
        data_strip=np.zeros([2,ny,nx])*np.nan
#data=data_strip.astype(int)-173
    return data_strip

def rt_mask_nowcast(mask_dir,dt_now,last_day,i_search):
    #i search is the number of minutes after the dt_now
    dt_lt = dt_now + datetime.timedelta(minutes=i_search)
    #print([i_search,dt_now,dt_lt])
    #mask1=xr.open_dataset(mask_dir+str((dt_now.replace(day=1)-datetime.timedelta(days=1)).month)+".nc")["counts"][dt_now.hour,:,:]
    #mask2=xr.open_dataset(mask_dir+str(dt_now.month)+".nc")["counts"][dt_now.hour,:,:]
    #mask3=xr.open_dataset(mask_dir+str((dt_now.replace(day=last_day)+datetime.timedelta(days=1)).month)+".nc")["counts"][dt_now.hour,:,:]
    mask1=xr.open_dataset(mask_dir+str((dt_lt.replace(day=1)-datetime.timedelta(days=1)).month)+".nc")["counts"][dt_lt.hour,:,:]
    mask2=xr.open_dataset(mask_dir+str(dt_lt.month)+".nc")["counts"][dt_lt.hour,:,:]
    mask3=xr.open_dataset(mask_dir+str((dt_lt.replace(day=last_day)+datetime.timedelta(days=1)).month)+".nc")["counts"][dt_lt.hour,:,:]
    mask=np.ones(np.shape(mask1))
    mask[np.where((mask1+mask2+mask3)>20)]=0
    #mask[np.where(mask2>20)]=0
    #mask[np.where(mask3>20)]=0
    return(mask)

def rect_rt_loop(dat_rect,use_rect,datadir,i_search,load_time,filters_real_time,missing_database_vec_t,nadd,lats_mid,do_risk):
    poly_val_freq = -1# DO IN NEED THIS?
    for i_rect,rect in enumerate(use_rect):
        timeStampStart= time.time()		  		    
        loaddir=datadir+rect+"_"+str(nadd)+"/Data_clim_freq_rectrect_"+rect+"_"+str(nadd)+"_search_"+str(i_search)+"_"+\
           "refhours_"+load_time+"00_"+load_time+"00.nc"
        #
		#print([os.path.exists(loaddir),loaddir])
		
        if os.path.exists(loaddir):
            ds=xr.open_dataset(loaddir)
            
            if ds.attrs["ngrids"]>=20:
										
                flt_ind=np.where(ds.flt.data==filters_real_time[int(i_search/60)])[0][0]
                dat_rect.append(ds["freq"].data[flt_ind,:,:]) 
                
                if do_risk:
                    if(i_rect)==0:  
                        poly_val_freq=ds.attrs["poly_val_freq"]
                    else:
                        poly_val_freq=np.maximum(poly_val_freq,ds.attrs["poly_val_freq"])
            else:  
                missing_database_vec_t.append(loaddir.split("/")[-1])
                fill_missing_ds=np.zeros(np.shape(lats_mid))-999
                dat_rect.append(fill_missing_ds)
                if do_risk:
                    if(i_rect)==0:
                        poly_val_freq=-999
                    else:
                        poly_val_freq=poly_val_freq
            ds.close()
        else:
            missing_database_vec_t.append(loaddir.split("/")[-1])
            fill_missing_ds=np.zeros(np.shape(lats_mid))-999
            dat_rect.append(fill_missing_ds)
            if do_risk:							
                if(i_rect)==0:
        	        poly_val_freq=-999
                else:
        	        poly_val_freq=poly_val_freq 
    dat_rect=np.stack(dat_rect,axis=0)
    dat_rect_max = np.nanmax(dat_rect,axis=0)			
    return(dat_rect_max,missing_database_vec_t,poly_val_freq)


def get_portal_outpath(datatype,tnow,viaS = False):
    if viaS:
        return '/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data'
    
    tstring = tnow[:8]
    outRoot = '/mnt/HYDROLOGY_stewells/'
    outDirs = {'Nowcast':os.path.join(outRoot,'geotiff','lawis_nowcast',tstring),
	           'CTT':os.path.join(outRoot,'geotiff','lawis_ctt',tstring),
			   'ConStruct':os.path.join(outRoot,'geotiff','lawis_construct',tstring),
			   'PastCores':os.path.join(outRoot,'geotiff','lawis_past_cores',tstring),
			   'Vis':os.path.join(outRoot,'geotiff','lawis_visible_channel',tstring),
			   'Nowcast_ts':os.path.join(outRoot,'lawis-west-africa','nflics_nowcast',tstring),
			   'Risk':os.path.join(outRoot,'lawis-west-africa','nflics_nowcast',tstring)}
	
    if not os.path.exists(outDirs[datatype]):
        os.makedirs(outDirs[datatype])
    return outDirs[datatype]

def roundDate(dt,nmin=15):
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

