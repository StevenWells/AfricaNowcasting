import gzip
import netCDF4 as nc
import os,sys,subprocess, datetime
import matplotlib.pyplot as plt
import rasterio
import xarray as xr
import multiprocessing
import numpy as np
from osgeo import gdal
import u_interpolate as uinterp
from plot_source_destination_functions import *

# SCW 2024 01 06 (version 2) include creation of additional rainfall accumulation maps
# SCW 2024 02 02 (version 3) include  option to output point values at list of lcoations (givnen in lat-long)

# VARIABLES

make_accumulations = True
# transfer search time(hours) - hours since now to search for files on HSAF ftp
search_hours=3
# keep full msg images
keep_fullmsg = True
#rerun even if already processed
reprocess_data=False
# make point value file
do_point_vals = False


# Path to raw archive (data pulled down from HSAF - probably will be simplified as will delete these on the fly
rawArchive = '/mnt/prj/nflics/NRT_HSAF/precip_h60/full_msg/'
# path to arhcive of precip netcdfs
dataDir = '/mnt/prj/nflics/NRT_HSAF/precip_h60/archive/'
tmpDir = '/home/stewells/AfricaNowcasting/tmp/'
pt_locn_names= ['Chisamba_Chipembi_CECB1', 'Kabwe_Mulungushi Univ_CEKB1', 'Chisamba_Gart farm_CECB2', 'Mufulira_Kafironda_COMU1', 'Kafulafuta_Police yard_COMA1', 'Kitwe_University Cumpas_COKI1', 'Katete_Katete FTC_ESKT1', 'Chama_Chama Ftc_ESCM1', 'Katete_Agric Camp_ESKT1', 'Lusaka_Unza_LULU1', 'chongwe_KKIA_LULR1', 'Luangwa_Kaunga_LULG1', 'Lusaka_Unza Agric_LULU2', 'Chiengi_Chiengi school_LPCE1', 'Milenge_Milenge_LPLU1', 'Nakonde_Mwenzo school_MUNK1', 'Mafinga_Ntendele school_MUMF2', 'Luwingu_school Sec School_NRLW1', 'Chilubi Island_Chilubi school_NRCI1', 'Mpulungu_Mpulungu school_NRML1', 'Nsenga Hill_school_NRSH1', 'Kabompo_Met Yard_NWKO1', 'Mwinilunga_Met office_NWMN2', 'Kalumbila_mine site_NWKB1', 'Maamba_mine office_SOSZ1', 'Gwembe_Muyunbwe_SOGW1', 'Chirundu_Lusitu FTC_SOMA1', 'Monze_kanchomba Ftc_SOPE1', 'Kazungula_Nyawa_SOKZ2', 'Kaoma_Met office_WEKM1', 'Lukulu_Council office_WELK1', 'Mulobezi_Sichili school_WEMZ1', 'Sesheke_Met office_WESS1', 'Kalabo_Met office_WEKB1', 'Mongu_Kataba_WEMG1']
pt_locn_locs =[[-14.928, 28.576], [-14.291, 28.567], [-14.94656, 28.08952], [-12.613, 28.179], [-13.31669, 28.75317], [-12.774, 28.207], [-14.083, 32.061], [-11.24, 33.155], [-14.1046, 31.9201], [-15.391, 28.332], [-15.319, 28.44], [-15.61973, 30.40192], [-15.39463, 28.33722], [-8.653, 29.164], [-12.123, 29.69], [-9.333, 32.755], [-10.263, 33.374], [-10.252, 29.906], [-10.769, 30.282], [-8.773, 31.117], [-9.36441, 31.24967], [-13.596, 24.208], [-11.74, 24.431], [-12.26875, 25.3051], [-17.34, 27.187], [-16.629, 27.772], [-16.1454, 26.7912], [-16.594, 27.494], [-17.19152, 25.89093], [-14.798, 24.804], [-14.342, 23.244], [-16.712, 24.952], [-17.477, 24.301], [-14.989, 22.682], [-15.445, 23.351]]



# full image size from HSAF [rows,column] - could be read from NetCDF
#fSize = [3712,3712]
# cutout [rowtop,rowbottom,colleft,colright] - assumes i=1, j=1 is lower left col,row  - SSA domain
cutout = [2717,638,320,2587]
nx = cutout[3] - cutout[2]+1 #2268
ny = cutout[0]-cutout[1]+1  #2080

# NFLICS variables
plot_area_ex = [-41,-27,27,79]
nflics_base="/mnt/users/hymod/seodey/NFLICS/" 
nx_dakarstrip=164 
ny_dakarstrip=580
blob_dx=0.04491576 #approx 5km over WAfrica for calculation of shape_wave (non-circular Blobs from Coni's code)
do_geotiff=True

class extendedCoreCalc():
    def __init__(self,ds_grid_info_ex,data_all_ex,rt_lats_ex,rt_lons_ex,plot_area_ex,use_times,tnow,do_geotiff,plotdir,tmpdir):
        super(extendedCoreCalc,self).__init__()
        self.ds_grid_info_ex=ds_grid_info_ex
        self.data_all_ex=data_all_ex
        self.use_times = use_times
        self.rt_lons_ex=rt_lons_ex
        self.rt_lats_ex=rt_lats_ex
        self.plot_area_ex = plot_area_ex
        self.tnow = tnow.strftime("%Y%m%d%H%M")
        self.tnowdt = tnow
        self.do_geotiff = do_geotiff
        self.plotdir = plotdir
        self.tmpdir = tmpdir
    def run(self):
    

        grid_lims_rt_ex=[np.where(self.rt_lats_ex[:,0]>self.plot_area_ex[0])[0][0],np.where(self.rt_lons_ex[0,:]>self.plot_area_ex[1])[0][0],
        np.where(self.rt_lats_ex[:,0]<self.plot_area_ex[2])[0][-1],np.where(self.rt_lons_ex[0,:]<self.plot_area_ex[3])[0][-1]]

       
        missing_ex = [np.all(np.isnan(grid)) for grid in self.data_all_ex]
        lats_edge_ex=np.array(self.ds_grid_info_ex['lats_edge'][...])  
        lons_edge_ex=np.array(self.ds_grid_info_ex['lons_edge'][...])
        lats_mid_ex=np.array(self.ds_grid_info_ex['lats_mid'][...])
        lons_mid_ex=np.array(self.ds_grid_info_ex['lons_mid'][...])
        blobs_lons_ex=np.array(self.ds_grid_info_ex['blobs_lons'][...])
        blobs_lats_ex=np.array(self.ds_grid_info_ex['blobs_lats'][...])

        if os.path.exists('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz'):
            print("reading npz weights")
            weightdata = np.load('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz')
            inds_ex = weightdata['inds_ex']
            weights_ex= weightdata['weights_ex']
            new_shape_ex=tuple(weightdata['new_shape_ex'])
                           
        else: # need to make it   
            print("creating weights")
            inds_ex, weights_ex, new_shape_ex=uinterp.interpolation_weights(lons_mid_ex[np.isfinite(lons_mid_ex)], lats_mid_ex[np.isfinite(lats_mid_ex)],blobs_lons_ex, blobs_lats_ex, irregular_1d=True)
            np.savez('/home/stewells/AfricaNowcasting/rt_code/weights_data_ex.npz',inds_ex=inds_ex,weights_ex=weights_ex,new_shape_ex=np.array(new_shape_ex))

       # plt.imshow(self.data_all_ex)
        #plt.show()
        data_interp_ex=uinterp.interpolate_data(self.data_all_ex, inds_ex, weights_ex, new_shape_ex)
       # plt.imshow(data_interp_ex)
       # plt.show()
        if do_point_vals:
            pt_locn_locs_fixed = [[(np.abs(blobs_lats_ex[:]-pt_locn_locs[iloc][0])).argmin(),\
                        (np.abs(blobs_lons_ex[:]-pt_locn_locs[iloc][1])).argmin()] for iloc in range(len(pt_locn_names))]	
            rvals =[]
            rvals = [data_interp_ex[pt_locn_locs_fixed[iloc][0],pt_locn_locs_fixed[iloc][1]] for iloc in range(len(pt_locn_names))]
            
            csvfile= '/mnt/prj/nflics/nflics_nowcasts/'+self.tnowdt.strftime('%Y/%m/%d/%H%M/')+'Site_HSAF_'+self.tnowdt.strftime('%Y%m%d%H%M')+'.csv'
            write_site_cores(csvfile,pt_locn_names,pt_locn_locs,rvals)
        if self.do_geotiff:
            if not os.path.exists(self.plotdir):
                os.mkdir(self.plotdir)
            rasPath = os.path.join(self.tmpdir,"HSAF_precip_h60_"+self.tnow+"_SSA.tif")
            rasPath_3857 = self.plotdir+"/HSAF_precip_h60_SSA_"+self.tnow+"_3857.tif"
            print(rasPath_3857)
            make_geoTiff([data_interp_ex],rasPath,reprojFile=rasPath_3857,trim=True)
            os.system('rm '+rasPath)
			
            if make_accumulations:
                update_accumulations(data_interp_ex,self.tnow,self.plotdir,self.tmpdir,inds_ex, weights_ex, new_shape_ex)
				
				
def write_site_cores(csvfile,places,locs,values):
    with open(csvfile,'w') as ff:
        ff.write('Location,Latitude,Longitude,HSAF_rate(mmh-1)\n')
        for site in range(len(places)):
            ff.write(','.join([places[site],str(locs[site][0]),str(locs[site][1]),str(round(values[site],5))])+'\n') 
            
def update_accumulations(hsafNow,tnow,plotdir,tmpdir,inds_ex, weights_ex, new_shape_ex,maxAcc=72):
    # accFile = npz file holding historical info
    # hsafNow = np array of HSAF image at time now
    # maxAcc = max accumulation period in hours
    # list of 72*4 images (max accumulation) #1,3,6,24,48,72
    # add latest to list and remove first one
	# for for each acc period, take the relevant num of images from the list
	# accumulation = 1/2*P0 + P1 + ... + Pn-1 + 1/2*Pn 
	# eg 1 hr accumulation  at 12:00 =  0.25*[P(12:00)/2 + P(11:45) + P(11:30) + P(11.15) + P(11:00)/2]
	# 0.25 factor is to convert mmh-1 to mm(15mins)-1
    accPeriods = [1,3,6,24,48,72]
    tnowS = tnow
    tnow  = datetime.datetime.strptime(tnow,'%Y%m%d%H%M')
    accArray = []
    datelist = [tnow - n*datetime.timedelta(minutes=15)  for n in range(4*maxAcc+1)]
    filelist = [os.path.join(rawArchive,str(x.year),str(x.month).zfill(2),'h60_'+x.strftime('%Y%m%d_%H%M')+'_fdk.nc.gz') for x in datelist]
    #filelist = filelist[::-1]
    #iacc = np.zeros((2080,2268))
    iacc = np.zeros((3712,3712))
    #initialise total
    accArray = np.copy(iacc)
    grid_lims_ex=ds_grid_info_ex["grid_lims_p"].data
    for ix,ifile in enumerate(filelist):
        #print(ifile)
        try:
            with gzip.open(ifile) as gz:
                with nc.Dataset('dummy',more='r',memory=gz.read()) as ncFile:         
                    iacc =ncFile.variables['rr'][:,:]	
                   
        except:
            print("Missing file "+ifile)
        if ix==0: # half the first value for accumulation
            iacc = iacc/2.0
        if ix in [x*4-1 for x in accPeriods]: # list of indices corresponding to accumulation periods
            acchr = int((ix+1)/4)
            print(acchr)
            accArr_i = np.array(np.round(0.25*np.add(accArray,iacc/2.0),2))
            accArr_i[accArr_i < 1] = 0.0
            accArr_i = accArr_i[(cutout[1]):(cutout[0]+1),(cutout[2]-1):(cutout[3])][:,::-1]
            accArr_i = accArr_i[grid_lims_ex[0]:grid_lims_ex[2],grid_lims_ex[1]:grid_lims_ex[3]]
            accArr_i_const=uinterp.interpolate_data(accArr_i, inds_ex, weights_ex, new_shape_ex)	
            #geotiff_outpath_acc = os.path.join('/home/stewells/AfricaNowcasting/satTest/geotiff/ssa_hsaf_precip_accum',tnowS)
            geotiff_outpath_acc = os.path.join('/mnt/HYDROLOGY_stewells/geotiff/ssa_hsaf_precip_accum',tnowS[:8])
            if not os.path.exists(geotiff_outpath_acc):
                os.makedirs(geotiff_outpath_acc)
            rasPath = os.path.join(tmpdir,"HSAF_precip_acc"+str(acchr)+"h_"+tnowS+"_SSA.tif")
            rasPath_3857 = geotiff_outpath_acc+"/HSAF_precip_acc"+str(acchr)+"h_SSA_"+tnowS+"_3857.tif"

            #rasPath = os.path.join(tmpDir,"HSAF_precip_acc"+str(acchr)+"h_"+tnowS+"_SSA.tif")

            #rasPath_3857 = os.path.join(geotiffDir,tnow.strftime('%Y%m%d'),"rainoverAfrica_SSA_"+tnowS+"_acc"+str(acchr)+"h_3857.tif")

            make_geoTiff([accArr_i_const],rasPath,reprojFile=rasPath_3857,trim=True)
            os.system('rm '+rasPath)


        accArray = np.add(accArray,iacc)


def make_geoTiff(data,rasFile,doReproj = True,origEPSG='4326',newEPSG='3857',reprojFile='test.tif',trim=False):
    nbands = len(data)
    dat_type = str(data[0].dtype)
    transform = rasterio.transform.from_origin(-27,27,0.04491576,0.04491576)
    rasImage = rasterio.open(rasFile,'w',driver='GTiff',
                           height=data[0].shape[0],width=data[0].shape[1],
                           count=nbands,dtype=dat_type,
                           crs = 'EPSG:'+str(origEPSG),
                           transform = transform,
                           
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
    if trim:
        os.system('rm '+rasFile2)


def process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area,nflics_base):
    grid_file=nflics_base+"/geoloc_grids/nxny"+str(nx)+'_'+str(ny)+'_nxnyds'+\
                str(nx_dakarstrip)+str(ny_dakarstrip)+'_blobdx'+str(blob_dx)+\
                '_area'+str(plot_area).replace(" ","").replace(",","_").replace("-","n")[1:-1]+'.nc'
		

    if not os.path.exists(grid_file):
        #geolocations(combine west africa domain ("wa") with dakar strip ("ds"))
        lats_edge,lons_edge=get_geoloc_grids(nx,ny,True,"SSA")   #lats and lons are interpolated to be the edges of each pixel
        lats_mid,lons_mid=get_geoloc_grids(nx,ny,False,"SSA")   #lats and lons are interpolated to be the edges of each pixel


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


# get data from HSAF

# get surrent list of files
#currentData
# get list of files from last n hours to look for
dateNow = datetime.datetime.now()
# nearest 15 minute to now
dateNow = dateNow+ (datetime.datetime.min - dateNow)%(datetime.timedelta(minutes=15)) 
# get list of last n hours
datelist = [dateNow - n*datetime.timedelta(minutes=15)  for n in range(4*search_hours)]
filelist = [datetime.datetime.strftime(x,'ftp://ftphsaf.meteoam.it:/h60/h60_cur_mon_data/h60_%Y%m%d_%H%M_fdk.nc.gz') for x in datelist]
#datelist = [dateNow - 4*datetime.timedelta(minutes=15)]
#filelist = [datetime.datetime.strftime(x,'ftp://ftphsaf.meteoam.it:/h60/h60_cur_mon_data/h60_%Y%m%d_%H%M_fdk.nc.gz') for x in datelist]

##filelist = ['ftp://ftphsaf.meteoam.it:/h60/h60_cur_mon_data/h60_20240201_1000_fdk.nc.gz']

rfiles = []
for ifile,hfile in enumerate(filelist):
    newfileDir =os.path.join(dataDir,str(datelist[ifile].year),str(datelist[ifile].month).zfill(2))
    newfileDir_raw= os.path.join(rawArchive,str(datelist[ifile].year),str(datelist[ifile].month).zfill(2))
    os.makedirs(newfileDir,exist_ok=True)
    os.makedirs(newfileDir_raw,exist_ok=True)

    # check if file has been processed already 
    if not os.path.exists(os.path.join(newfileDir,datetime.datetime.strftime(datelist[ifile],'h60_%Y%m%d_%H%M_fdk_ssa.nc'))) or reprocess_data:
        os.system('wget -P '+newfileDir_raw+' -nc '+hfile)
    # now check again to see if it has been pulled down
        if os.path.exists(os.path.join(newfileDir_raw,datetime.datetime.strftime(datelist[ifile],'h60_%Y%m%d_%H%M_fdk.nc.gz'))):
            rfiles+=[os.path.join(newfileDir_raw,datetime.datetime.strftime(datelist[ifile],'h60_%Y%m%d_%H%M_fdk.nc.gz'))]
    else:
        print(hfile+' already processed')


##rfiles = ['/prj/nflics/NRT_HSAF/precip_h60/full_msg/2024/02/h60_20240201_1000_fdk.nc.gz']
# for each file, process it
for rfile in rfiles:
    print(rfile)
    try:
        with gzip.open(rfile) as gz:
            with nc.Dataset('dummy',more='r',memory=gz.read()) as ncFile:

                fSize = [ncFile.dimensions['y'].size,ncFile.dimensions['x'].size]
                rr_full = ncFile.variables['rr']
            #rr = rr_full[(cutout[1]):(cutout[0]+1),(cutout[2]-1):(cutout[3])][:,::-1]
                rr = rr_full[(cutout[1]):(cutout[0]+1),(cutout[2]-1):(cutout[3])][:,::-1]
                indx_full =ncFile.variables['qind']
                #for name, variable in ncFile.variables['rr']:
            #indx = indx_full[(cutout[1]):(cutout[0]+1),(cutout[2]-1):(cutout[3])][:,::-1]
                indx = indx_full[(cutout[1]):(cutout[0]+1),(cutout[2]-1):(cutout[3])][:,::-1]
            # make a tif
                #geotiff_outpath = '/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data/'
                #geotiff_outpath = '/home/stewells/AfricaNowcasting/satTest/geotiff/ssa_hsaf_precip'        
            #geotiff_outpath = '/data/hmf/projects/LAWIS/WestAfrica_portal/hsaf_test/'
                tnow_day = '_'.join(rfile.split('/')[-1].split('_')[1:3])
                tnow = datetime.datetime.strptime(tnow_day,"%Y%m%d_%H%M")
                tnowStr = tnow.strftime("%Y%m%d")
                #geotiff_outpath = os.path.join('/home/stewells/AfricaNowcasting/satTest/geotiff/ssa_hsaf_precip',tnowStr)
                geotiff_outpath = os.path.join('/mnt/HYDROLOGY_stewells/geotiff/ssa_hsaf_precip',tnowStr)
                #geotiff_outpath = os.path.join('/mnt/HYDROLOGY_stewells/geotiff/ssa_hsaf_precip',tnowStr)
                print("Processing HSAF data for "+datetime.datetime.strftime(tnow,"%Y-%m-%d %H:%M")) 
                use_times=[tnow]
            # set up geoloc grids
        #try:
                ds_grid_info_ex=process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area_ex,nflics_base)
                grid_lims_ex=ds_grid_info_ex["grid_lims_p"].data
                rt_lats=np.array(ds_grid_info_ex['lats_edge'][...])  
                rt_lons=np.array(ds_grid_info_ex['lons_edge'][...])
                rt_lats_ex = np.copy(rt_lats)
                rt_lons_ex = np.copy(rt_lons)
    # get data
                data_all = rr[grid_lims_ex[0]:grid_lims_ex[2],grid_lims_ex[1]:grid_lims_ex[3]]
                data_all_ex = np.copy(data_all)
                pext=extendedCoreCalc(ds_grid_info_ex,data_all_ex,rt_lats_ex,rt_lons_ex,plot_area_ex,use_times,tnow,do_geotiff,geotiff_outpath,tmpDir)
                pext.run()
            
                ds_rr=xr.Dataset()
                #ds_rr['rr']=xr.DataArray(rr_full, coords={'y': range(rr_full.shape[0]) , 'x': range(rr_full.shape[1])},dims=['y', 'x']) 
                ds_rr['rr']=xr.DataArray(rr, coords={'y': range(rr.shape[0]) , 'x': range(rr.shape[1])},dims=['y', 'x']) 
                ds_rr['qind']=xr.DataArray(indx, coords={'y': range(rr.shape[0]) , 'x': range(rr.shape[1])},dims=['y', 'x']) 
                #ds_rr['rr']=xr.DataArray(rr_full[:]) 
                #ds_rr.attrs['time']=self.tnow

            # take rain rate atrtributes from the raw netcdf
                for attrname in ncFile.variables['rr'].ncattrs():
                    ds_rr.rr.attrs[attrname] =ncFile.variables['rr'].getncattr(attrname)
            # copy global attributes            
                for gattr in ncFile.ncattrs():
                    ds_rr.attrs[gattr] = ncFile.getncattr(gattr)

        
                ds_rr.attrs['grid']="HSAF rainfall rate"
                ds_rr.attrs['missing']="nan"
                comp = dict(zlib=True, complevel=5)
                enc = {var: comp for var in ds_rr.data_vars}
            
                ds_rr.to_netcdf(path=os.path.join(dataDir,str(tnow.year),str(tnow.month).zfill(2),datetime.datetime.strftime(tnow,"h60_%Y%m%d_%H%M_fdk_ssa.nc")),\
                     mode='w', encoding=enc, format='NETCDF4')
                ds_rr.close()
    except Exception as e:
        print("Error processing file "+rfile)
        print(e)
    if not keep_fullmsg:
        os.system('rm '+rfile)

       # except Exception as e:
        #    print(e)
        #    print("Failed to process data for "+(datetime.datetime.strftime(tnow,"%Y %m %d %H:%M")))




