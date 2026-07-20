#SRA Created 28/04/2026 to Seperate out the "NFLCIS" flood risk calculations  
#Initially Dakar only, replicating the origional NFLICS flood risk work.     

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



import os,datetime,array
import xarray as xr
import numpy as np
import pandas as pd
import process_realtime_fns as fns   #will need to move this code into sftp_extract directory
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import shapefile as shp
from pyproj import Proj, Transformer,CRS
import matplotlib as mpl
import numpy.ma as ma

nflics_base="/mnt/users/hymod/seodey/NFLICS/"
n_inner=41     #size in grid points of the inner square used for selecting from the historical database
nadd=20         #number of additional points each side to add for getting historical sample
n_search=6
t_search=60
search_freq=60
plot_area=[8,-20,20,0]		        #ll_lat,ll_lon,urlat,urlon in degrees
nx=1640                                 #Specify grid size. origional resolution data (cells not calculated)
ny=580
nx_dakarstrip=164 
ny_dakarstrip=580
blob_dx=0.04491576 #approx 5km over WAfrica for calculation of shape_wave (non-circular Blobs from Coni's code)
plot_area_sub =  {'wa':[-2,-23,20,32],'sadc':[-38,8,0,55]}	# now going further south
#geoloc_sub_file = {}
#geoloc_sub_file['wa'] = fns.process_grid_info(nx,ny,-1,-1,blob_dx,plot_area_sub['wa'],nflics_base)	

#PROCESS THE LOCAL commune grid 
local_grid_file="dakar_grid_v4.nc"
ds_grid_info=fns.process_grid_info(nx,ny,nx_dakarstrip,ny_dakarstrip,blob_dx,plot_area,nflics_base)  
grid_lims_p=ds_grid_info["grid_lims_p"].data
lats_mid=ds_grid_info["lats_mid"].data
lons_mid=ds_grid_info["lons_mid"].data
lats_edge=ds_grid_info["lats_edge"].data
lons_edge=ds_grid_info["lons_edge"].data

def process_local_grid(local_grid_file,ds_grid_info):
    if not os.path.exists(local_grid_file):
        #function to process the dakar admin grid
        #def process_dakar_grid(nflics_base):
        #-------------------------------------------------------------------------------
        # Commune shapefiles
        #------------------------------------------------------------------------------
        sf=shp.Reader("/mnt/prj/nflics/shapefiles/commune_shapefiles/Communes_limits_Dkr.shp")
        fields=np.array([field[0] for field in sf.fields][1:])
        communes=[np.array(record)[fields=='CCRCA'][0] for record in sf.records()]
        names_dict=dict(zip(np.unique(communes),range(0,len(np.unique(communes)))))
        areas=[np.array(record)[fields=='Shape_Area'][0] for record in sf.records()]
        departement=[np.array(record)[fields=='DEPT'][0] for record in sf.records()]
        grid_poly=np.ones(np.shape(ds_grid_info["lats_mid"].data)).astype("int")-1000
        use_grid=zip(ds_grid_info["lons_mid"].data.flatten(),ds_grid_info["lats_mid"].data.flatten())
        #projection information
        transformer = Transformer.from_crs('epsg:32628', 'epsg:4326', always_xy=True)
        #-------------------------------------------------------------------------------
        # Derived variables
        #-------------------------------------------------------------------------------
        #Get the grid of jobs to create
        grid_lims_p=ds_grid_info["grid_lims_p"].data
        lats_mid=ds_grid_info["lats_mid"].data
        lons_mid=ds_grid_info["lons_mid"].data
        lats_edge=ds_grid_info["lats_edge"].data
        lons_edge=ds_grid_info["lons_edge"].data
        #create a lat-lon grid covering the required (Dakar) area    
        lons_dakar_grid=np.arange(-17.55,-17.1,0.0025)
        lats_dakar_grid=np.arange(14.6,14.9,0.0025)
        dakar_grid=np.meshgrid(lons_dakar_grid,lats_dakar_grid)
        dakar_communes=np.zeros(np.shape(dakar_grid[0]))*np.nan #grid relating dakar_grid to commune points 
        dakar_msg_pt=np.zeros(np.shape(dakar_grid[0]))*np.nan   #grid relating dakar_grid to msg points 
        dakar_msg_ll_lat=np.zeros(np.shape(dakar_grid[0]))*np.nan  #lower left MSG lat coordinate
        dakar_msg_ll_lon=np.zeros(np.shape(dakar_grid[0]))*np.nan  #lower left MSG lon coordinate
        #setup the basemap
        fig,ax=plt.subplots(figsize=(10, 10))
        m = Basemap(projection='merc', ax=ax, lat_0=0.,lon_0=0., resolution='l',
        llcrnrlon=-17.6,llcrnrlat=14.5,urcrnrlon=-17.1,urcrnrlat=15.0)
        X, Y = m(ds_grid_info["lons_edge"].data,ds_grid_info["lats_edge"].data)
        pt_vec=[]
        pt_vec_ll=[]
        pt_vec_ul=[]
        pt_vec_ur=[]
        pt_vec_lr=[]
        #process the msg points on the fine resolution dakar_grid
        dakar_points=list(zip(dakar_grid[0].flatten(),dakar_grid[1].flatten()))
        i=0
        #loop over the MSG points and put MSG points onto Dakar grid
        for ix in range(105,121,1):
            for iy in range(235,246,1):
                pt_vec.append((ix,iy)) 
                pt_vec_ll.append((lons_edge[iy,ix],lats_edge[iy,ix])) 
                pt_vec_ul.append((lons_edge[iy+1,ix],lats_edge[iy+1,ix])) 
                pt_vec_ur.append((lons_edge[iy+1,ix+1],lats_edge[iy+1,ix+1])) 
                pt_vec_lr.append((lons_edge[iy,ix+1],lats_edge[iy,ix+1])) 
                lons_vec=[lons_edge[iy,ix],lons_edge[iy+1,ix],lons_edge[iy+1,ix+1],lons_edge[iy,ix+1],lons_edge[iy,ix]]
                lats_vec=[lats_edge[iy,ix],lats_edge[iy+1,ix],lats_edge[iy+1,ix+1],lats_edge[iy,ix+1],lats_edge[iy,ix]]
                path=mpltPath.Path(list(zip(lons_vec,lats_vec)))
                inside = path.contains_points(dakar_points).reshape(np.shape(dakar_grid[0]))
                if inside.any():
                    #print("Index:",i,"point(x,y):", ix,iy, "covered by Dakar grid")
                    dakar_msg_pt[inside==True]=i #give the grid a number
                    dakar_msg_ll_lat[inside==True]=lats_edge[iy,ix] #give the grid a number
                    dakar_msg_ll_lon[inside==True]=lons_edge[iy,ix] #give the grid a number          
                i=i+1
                
        Xt, Yt = m(dakar_grid[0],dakar_grid[1])  
        #process the dakar communes on the fine resolution dakar_grid
        for i,name in enumerate(communes): 
            loc=np.where(np.array(communes)==name)[0][0]
            print("processing "+communes[loc])
            poly=sf.shape(loc).points
            xproj,yproj=zip(*poly)
            x,y=transformer.transform(xproj,yproj)
            xm,ym=m(x,y)
            path=mpltPath.Path(list(zip(x,y)))
            inside = path.contains_points(dakar_points).reshape(np.shape(dakar_grid[0]))    
            dakar_communes[inside==True]=i #give the grid a number
            #patch=mpl.patches.Polygon(list(zip(xm,ym)),fill=False,facecolor="white",edgecolor='black')
            #ax.add_patch(patch)
        
        Xt, Yt = m(dakar_grid[0],dakar_grid[1])
        
        #get the msg points for each commune
        commune_msg_pt=[]
        commune_msg_ll=[]
        commune_msg_ind=[]
        for i,name in enumerate(communes):    
            commune_select=np.where(dakar_communes==i)
            pt,count=np.unique(dakar_msg_pt[commune_select],return_counts=True)
            print("MSG name,pt,count:",name,pt,count)
            perc=count*100/sum(count)
            print((np.round(perc,0)))
            pt=pt[perc>5]
            commune_msg_pt.append([pt_vec[int(p)] for p in pt])
            commune_msg_ll.append([pt_vec_ll[int(p)] for p in pt])
            commune_msg_ind.append([int(p) for p in pt])
        
        #save the dakar dataset
        ds=xr.Dataset() #create dataset to save to netcdf for future use
        ds['dakar_lons']=xr.DataArray(dakar_grid[0], coords={'ys_mid': range(len(lats_dakar_grid)) , 'xs_mid': range(len(lons_dakar_grid))},dims=['ys_mid', 'xs_mid'])
        ds['dakar_lats']=xr.DataArray(dakar_grid[1], coords={'ys_mid': range(len(lats_dakar_grid)) , 'xs_mid': range(len(lons_dakar_grid))},dims=['ys_mid', 'xs_mid'])
        ds['dakar_msg_pt']=xr.DataArray(dakar_msg_pt, coords={'ys_mid': range(len(lats_dakar_grid)) , 'xs_mid': range(len(lons_dakar_grid))},dims=['ys_mid', 'xs_mid'])
        ds['dakar_communes']=xr.DataArray(dakar_communes, coords={'ys_mid': range(len(lats_dakar_grid)) , 'xs_mid': range(len(lons_dakar_grid))},dims=['ys_mid', 'xs_mid'])
        ds['commune_msg_ind']=xr.DataArray([str(ind)[1:-1].replace(" ","") for ind in commune_msg_ind], coords={'communes': communes},dims=['communes'])        
        ds['commune_msg_pt']=xr.DataArray([str(pt)[2:-2].replace(" ","").replace("),(","_") for pt in commune_msg_pt], coords={'communes': communes},dims=['communes'])
        ds['commune_msg_ll']=xr.DataArray([str(pt).replace("np.float64","").replace(")), ((","_").replace("), (",",")[3:-3] for pt in commune_msg_ll], coords={'communes': communes},dims=['communes'])
        ds['commune_ind']=xr.DataArray(range(len(communes)), coords={'communes': communes},dims=['communes'])
        
        for i,name in enumerate(ds['commune_ind'].data):    
            llarr=np.array(commune_msg_ll[i]).transpose()
            ptarr=np.array(commune_msg_pt[i]).transpose()
            indarr=np.array(commune_msg_ind[i]).transpose()
            commune_arr=np.vstack((llarr,ptarr,indarr))
            ds[str(name).zfill(2)+"_"+"msg"]=xr.DataArray(commune_arr,coords={'info':["lon","lat","X","Y","ind"],'point':range(len(commune_msg_ind[i]))},dims=['info','point'])
            
        comp = dict(zlib=True, complevel=5)
        enc = {var: comp for var in ds.data_vars}
        
        #ds.to_netcdf(path="/mnt/prj/nflics/geoloc_grids/dakar_grid_v4.nc",mode='w', encoding=enc, format='NETCDF4')
        ds.to_netcdf(path=local_grid_file,mode='w', encoding=enc, format='NETCDF4')
    else:
        ds=xr.open_dataset("dakar_grid_v4.nc")
        
   #get dictionary in form {COMMUNE1: (lat, lon), COMMUNE2:[(lat1,lat2), (lon1, lon2)],...}
    msg_lons=[]
    msg_lats=[]
    for i in range(len(ds["commune_msg_ll"])):
        commune_ll=ds["commune_msg_ll"][i]
        msg_lons.append([float(s.split(",")[0]) for s in commune_ll.data.tolist().split("_")])
        msg_lats.append([float(s.split(",")[1]) for s in commune_ll.data.tolist().split("_")])
        commune_locs=dict(list(zip(ds.coords["communes"].data.tolist(),list(zip(msg_lats,msg_lons)))))
        
    return(ds, commune_locs)
   

ds_dakar, commune_locs=process_local_grid(local_grid_file,ds_grid_info)

 

#get the MSG gridpoint from (lat, lon). 1D version
def get_grid_from_ll(lats_ref, lons_ref, locs):
    pt_locs=[]
    for loc in locs.keys():
       pt_locs.append([(np.abs(lats_ref-locs[loc][0])).argmin(),\
                 (np.abs(lons_ref-locs[loc][1])).argmin()]) 
    return(pt_locs)
    
#get the MSG gridpoint from (lat, lon). 2D version 
def get_grid_from_ll_2d(lats_ref, lons_ref, locs):
    if isinstance(locs,dict):
        pt_locs={}
        for loc in locs.keys():
            if len(locs[loc])>0:
                pt_loc_sub=[]
                for sublat,sublon in zip(locs[loc][0],locs[loc][1]):
                    dist_sq = (lons_ref - sublon)**2 + (lats_ref - sublat)**2
                    min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                    pt_loc_sub.append(min_idx)
                pt_locs[loc]=list(zip(*pt_loc_sub))
            else:
                dist_sq = (lons_ref - locs[loc][1])**2 + (lats_ref - locs[loc][0])**2
                min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                pt_locs[loc]=min_idx   
    else:  #list of locations. Each element in list is a []
        pt_locs=[]
        for lat, lon in locs:
            dist_sq = (lons_ref - lon)**2 + (lats_ref - lat)**2
            min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
            pt_locs.append(min_idx)
    return(pt_locs) 
    
def tofit_shift(x,fits,shift):
    if shift>=x:
        y=100
    else:
        a=fits[0]
        b=fits[1]
        c=fits[2]
        y = a/(b+(x-shift))*np.exp(-c*(x-shift))
    return(y)


def get_portal_outpath(datatype,tnow,lawisDirs,viaS = False):
    if lawisDirs["lawisOffline"]:  #SRA 04/26 added offline option. Send everything here offline!
        tstring = tnow[:8]
        outRoot = lawisDirs['offlineDir']
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
        
    if viaS:
        return lawisDirs["viaSDir"]
        
    tstring = tnow[:8]
    outRoot = lawisDirs["mntDir"]
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



###########################################################
#  function to plot the risk maps -moved from process_realtime_fns.py SRA Jul 2026
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
    #pc = m.pcolormesh(X,Y,toplt_poly_dakar_m,cmap='Greys',vmin=295,vmax=317,lw=2)
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

"""
#Testing options for stand-alone run
tnow="202509090800"   #hard code for testing. Will be arguments
lawisDirs={"viaSDir":'/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/SANS_transfer/data',
               "mntDir":'/mnt/HYDROLOGY_stewells/',
               "offlineDir":'/mnt/users/hymod/seodey/NFLICS/nflics_current/'}
lawisDirs["lawisOffline"]=True
options={"do_extended_core_calcs":False,
            "do_lst_adjustments" :False,
            "do_shared_plots":True,
            "do_point_timeseries" : True,
            "do_geotiff" : True, #Needs to be true to output the timeseris
            "output_site_cores":True,
            # OPTIMISATION OPTIONS
            "opt_geotiff" : True,
            "opt_geotiff_float32":True,
            "opt_geotiff_ndpls" : 2,
            "extract_riskpt": True,
             "nflics_output_version_portal": 2 #
}

#run_flood_risk(tnow,lawisDirs,options,plotbase)
 """   
def run_flood_risk(tnow,lawisDirs,options,plotbase):
    dt_now=datetime.datetime.strptime(tnow,"%Y%m%d%H%M")    # get the required dates
    #fixed file paths
    rt_code_input="/mnt/prj/nflics/RT_code_v2_input/"
    #plotbase="/mnt/users/hymod/stewells/NFLICS/nflics_nowcasts/"
    plotdir=os.path.join(plotbase,tnow[:4],tnow[4:6],tnow[6:8],tnow[8:12])
    #read in preseent input information
    hazard_mapping=pd.read_csv(rt_code_input+"/Hazard_mapping.csv").to_dict(orient="list")
    vul_mapping=pd.read_csv(rt_code_input+"/Vulnerability.csv").to_dict(orient="list")
    ant_cond=pd.read_csv(rt_code_input+"/Antecedent_conditions.csv",index_col=0).transpose().to_dict(orient="list")
    ds_rg_hist=xr.open_dataset("/mnt/prj/nflics/RT_code_v2_input/dakar_rain_hazard_v2.nc")
    
    #polygon information for plotting
    grid_poly_ds=xr.open_dataset(nflics_base+'/shape_files/wca_admbnda_adm1_ocha/wca_admbnda_adm1_ocha'+str(plot_area_sub["wa"]).replace(" ","").replace(",","_").replace("-","n")[1:-1]+'.nc')
    
    #time calculations
    day_in_season=(dt_now-pd.Timedelta(minutes=int(ant_cond["Day_start"][0][:2])*60+int(ant_cond["Day_start"][0][2:]))-datetime.datetime(dt_now.year,5,31,0,0)) 
    
    ###############################
    #todays cores -> calculated in the NFLICS nowcast updated in process_realtime_fns.py code
    #cores SO FAR TODAY. 
    #load in file dependent on time where a new NFLICS day startes (defined in ant_cond["Day_start"])
    
    if int(tnow[-4:])<int(ant_cond["Day_start"][0]): 
        yesterday=dt_now-pd.Timedelta(hours=24)
        daily_dir=os.path.join("/",*plotdir.split("/")[:-3],str(yesterday.month).zfill(2),str(yesterday.day).zfill(2))
    else:
        daily_dir=os.path.join("/",*plotdir.split("/")[:-1])
    os.makedirs(daily_dir,exist_ok=True)
    daily_file=os.path.join(daily_dir,"Day_cores_"+"".join(daily_dir.split("/")[-3:])+".csv")
    daily_cores=pd.read_csv(daily_file)             #read in daily cores file
    #format
    daily_cores_arr=np.array(daily_cores)[:,2:] #remove the CCRCA and Geolocation index columns
    daily_cores_arr[daily_cores_arr<0]=np.nan
    daily_cores_sites=np.array(daily_cores["Geolocation index"])
    today_cores=np.nansum(daily_cores_arr,axis=1)#total number of cores so far "today"
    
    if dt_now.month<6:                   #cutoff for statistics. Dependant on Month
        print("Fix for months outside of JJAS")
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==6)].data[0]
    elif dt_now.month>9:
        print("Fix for months outside of JJAS")
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==9)].data[0]
    else:
        core_cutoff=ds_rg_hist["core_sample_cutoffs"][np.where(ds_rg_hist.coords["months"]==dt_now.month)].data[0]
    
    today_cores[np.where(today_cores>=core_cutoff)]=core_cutoff-1
    
    ##########################################
    # Anticedent cores
    season_file_ante=os.path.join("/",*plotdir.split("/")[:-3],"Season_cores_ante_"+daily_dir.split("/")[-3]+".csv")#rain amount associated with 
    season_cores_anti=pd.read_csv(season_file_ante)
    anti_cores=np.array(season_cores_anti)[:,2:(day_in_season.days+1)] #remove CCRCA and Geolocation columns, then read from 1 to day_in_season-1
    anti_cores[np.where(anti_cores==-999)]=np.nan
    
    #anticedent conditions & water amounts
    m,n=[int(ant_w) for ant_w in ant_cond["Weighting"][0].split("_")]
    anti_vec=np.array(1/np.power(np.arange(day_in_season.days)+1,m/n)[::-1][:-1]) #upto and including YESTERDADY
    anti_core_rain=[]           #needs to be calculated
    anti_dry=[int(ant_cond["Dry"][0])]*len(today_cores) #fixed per site
    anti_wet=[int(ant_cond["Wet"][0])]*len(today_cores) #fixed per site
    
    #the probability values (all need to be calculatd)
    prob_core_rain=[]   #probability of rain | core (anti from core)
    prob_dry=[]         #probability of rain | core  (anti dry)
    prob_wet=[]         #probability of rain | core  (anti wet)
    
    if dt_now.month<6:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==6)[0][0]].decode() 
    elif dt_now.month>9:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==9)[0][0]].decode() 
    else:
        month_assoc=ds_rg_hist["months_assoc"].data[np.where(ds_rg_hist.coords["months"]==dt_now.month)[0][0]].decode() 
        
    mean_vals=ds_rg_hist["mean_vec_"+month_assoc]
    fits_all=ds_rg_hist["fits_all_"+month_assoc]
    
    ##########################################
    #Nowcast probabilites for each commune P_NFLICS(core)
    ncast_prob_all=[]
    ncast_dir="/mnt/users/hymod/seodey/NFLICS/nflics_current/lawis-west-africa/nflics_nowcast/20250909/Nowcast_risk/"
    for lead in [str(x).zfill(3) for x in np.arange(60,420,60)]:
        print(lead)
        ncast_prob_all.append(pd.read_csv(ncast_dir+"nflics_nowcast_pt_202509090800_"+lead+".csv")[lead].tolist())
        
    ncast_prob_all=np.array(ncast_prob_all)    
    prob_core_nflics=np.amax(ncast_prob_all,axis=0)#P_NFLICS(core)
    
    ##########################################
    #hazard probability and risk matrix mapping
    for i_site,region in enumerate(hazard_mapping['wca_admbnda_adm1_ocha shapefile ']):   
        anti_core_rain.append(np.nansum(anti_cores[i_site,:]*anti_vec))                               #anticenent rain estimate from cores
        prob_core_rain.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_core_rain[-1]))  #probability of rain | core (anti from core)
        prob_dry.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_dry[i_site]))          #probability of rain | core  (anti dry)
        prob_wet.append(tofit_shift(60,fits_all[today_cores[i_site]].data,anti_wet[i_site]))          #probability of rain | core  (anti wet)
        
    prob_core_rain=np.array(prob_core_rain)     #P^core(rain|core)
    prob_wet=np.array(prob_wet)                 #P^wet(rain|core)
    prob_dry=np.array(prob_dry)                 #P^dry(rain|core)
    
    #Arrays to contain the flood hazard row in risk matrix (bottom to top)
    risk_row_core=np.zeros(np.shape(prob_core_nflics)).astype(int)
    risk_row_wet=np.zeros(np.shape(prob_core_nflics)).astype(int)
    risk_row_dry=np.zeros(np.shape(prob_core_nflics)).astype(int)
    
    #hazard levels define the thresholds between rows in %
    haz_l=np.array(hazard_mapping['Low likelihood thresh'])
    haz_m=np.array(hazard_mapping['Medium likelihood thresh'])
    haz_h=np.array(hazard_mapping['High likelihood thresh'])
    print("Using hazard thresholds: low",np.unique(haz_l), ", med", np.unique(haz_m), ", high", np.unique(haz_h), "[%]")
    
    #multiplicatin of probabilites (not %) so comparison is: (prob_core_nflics/100)*(prob_core_rain/100) vs haz_m/100
    #                                                     ->  prob_core_nflics*prob_core_rain/100 vs haz_m
    
    risk_row_core[np.where((prob_core_nflics*prob_core_rain/100.>=haz_l) & (prob_core_nflics*prob_core_rain/100.<haz_m))]=1
    risk_row_core[np.where((prob_core_nflics*prob_core_rain/100.>=haz_m) & (prob_core_nflics*prob_core_rain/100.<haz_h))]=2
    risk_row_core[np.where(prob_core_nflics*prob_core_rain/100.>=haz_h)]=3
    
    risk_row_wet[np.where((prob_core_nflics*prob_wet/100.>=haz_l) & (prob_core_nflics*prob_wet/100.<haz_m))]=1
    risk_row_wet[np.where((prob_core_nflics*prob_wet/100.>=haz_m) & (prob_core_nflics*prob_wet/100.<haz_h))]=2
    risk_row_wet[np.where(prob_core_nflics*prob_wet/100.>=haz_h)]=3
    
    risk_row_dry[np.where((prob_core_nflics*prob_dry/100.>=haz_l) & (prob_core_nflics*prob_dry/100.<haz_m))]=1
    risk_row_dry[np.where((prob_core_nflics*prob_dry/100.>=haz_m) & (prob_core_nflics*prob_dry/100.<haz_h))]=2
    risk_row_dry[np.where(prob_core_nflics*prob_dry/100.>=haz_h)]=3
    
    ##########################################
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
    
    #put all sites in for portal output
    for v_geo,vul_c,vul_s in zip(vul_geo,vul_col,vul_sort):
        vul_col_all[np.where(haz_geo==v_geo)]=vul_c
        vul_pop[np.where(haz_geo==v_geo)]=vul_s.round(0)
        
    ##########################################
    #Risk nowcast output
    toout={'Commune shapefile CCRCA':hazard_mapping['Commune shapefile CCRCA'],\
            'Geolocation index':hazard_mapping['Geolocation index'],\
            'Prob(core)':prob_core_nflics,\
            'Antecedent past-cores':np.array(anti_core_rain).round(0),'Antecedent wet':anti_wet,'Antecedent dry':anti_dry,\
            'Prob(rain|past-cores)':prob_core_rain.round(0),'Prob(rain|wet)':prob_wet.round(0),'Prob(rain|dry)':prob_dry.round(0),\
            'Risk row past-cores':risk_row_core,'Risk row wet':risk_row_wet,'Risk row dry':risk_row_dry,\
             'Risk col':vul_col_all, 'Population at risk':vul_pop}
             
    #output information
    outpath=plotdir+"/Risk_nowcast_v3_1_0_"+tnow+"_000.csv"  #DO NOT CHANGE - HARD CODED IN GUI
    outpath2 = get_portal_outpath('Risk',tnow,lawisDirs)+"/Risk_nowcast_v"+str(options["nflics_output_version_portal"])+"_"+tnow+"_000.csv"
    
    pd.DataFrame(toout).to_csv(outpath,index=False)
    if options["do_geotiff"]:
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
        if options["do_shared_plots"]:
            plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,vul_col_all,outpath_vul,"vul",\
            "Surface water flooding 2009 (population impacted)", 'Impact \n (population)',use_ylab)
    else:
        pass
    
    risk_matrix=np.array(([0,0,0,0],[0,0,1,1],[1,1,2,2],[1,2,2,3],[np.nan,np.nan,np.nan,np.nan])).transpose()  #row then column
    outpath_prob="/mnt/prj/nflics/"  #DO NOT CHANGE - HARD CODED IN GUIvul_col_all)
    vul_col_all[vul_col_all<0]=-1
    
    toplt_risk_core=[risk_matrix[r,c] for r,c in zip(risk_row_core, vul_col_all)]
    print(toplt_risk_core)
    toplt_risk_wet=[risk_matrix[r,c] for r,c in zip(risk_row_wet, vul_col_all)]
    toplt_risk_dry=[risk_matrix[r,c] for r,c in zip(risk_row_dry, vul_col_all)]
    
    plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_core,plotdir+"/Risk_estimated_v3_1_0_"+tnow+"_000.png",\
                "risk_ante","Flood Risk \n given estimated surface conditions",'Flood Risk')
    plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_wet,plotdir+"/Risk_wet_v3_1_0_"+tnow+"_000.png",\
                "risk_wet","Flood Risk \n given wet surface ("+ant_cond["Wet"][0]+"mm)",'Flood Risk')
    plot_slice_risk(ds_dakar,ds_grid_info,grid_poly_ds,toplt_risk_dry,plotdir+"/Risk_dry_v3_1_0_"+tnow+"_000.png",\
               "risk_dry","Flood Risk \n given dry surface ("+ant_cond["Dry"][0]+"mm)",'Flood Risk') 
               
     