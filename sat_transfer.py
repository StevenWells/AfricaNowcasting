#!/users/hymod/stewells/NFLICS/python_env/Miniconda2/envs/py27/bin/python
# ncas_transfer.py
#
# Update the NCAS mirror directory and move any new files into the correct location on the database
#
# Author: stewells
#
# Date: 18/02/2020
#
# Usage: python ncas_transfer_wa.py <runtype> <db_version>
#
# <runtype> is either 'realtime' or 'historical' (no quotes)
# <db_version> is the database version (current options 1 or 2). If omitted then 2 will be assumed
#
# 01/04/2020 SRA edited to include production of NFLICS nowcasts (v1)
# 12/05/2020 SRA tydied up and edited to run on python 3.7 
# 27/05/2020 SRA edited to include mirror of ukceh nowcasts to UKCEH sftp site (v1.4)
# 08/07/2020 SRA edited to run on wllf005
# 08/07/2020 SRA edited to include option to be a "shadow" run with 15minutes delay in upload to sftp
#26/02/2021 SRA/SCW edits for 2021 changes to be made
# 12/08/2021 SCW edited to incorporate LST adjustment factors
# 13/09/2021 SCW Added versioning to keep separate outputs with and without LST adjustment. File names of outputs amended to include version numbers
#             Version a and 2 both being run as GUI needs version 1 at the moment. Cron job runs every other minute for each.
# 13/09/2021 SCW Added option to output a tif of the nowcasts (for Chris)
# 10/01/2022 SCW Added geotiff outputs for use by LAWIS portal
# 12/02/2022 SCW Added temporary fix to realtime functions core cutoff outside of the JJAS months, to allow cores to be calculated then, but assuming no nowcasts being made
#                TODO: fix this properly, will need adjusting when all months start being available.
# 16/03/2022 SCW Added additional save of visible channel image to rt_save location (if present)
# 11/05/2023 SCW Allow main input domain to be whole of West Africa
# 11/04/2024 SCW Update historical database extraction to be a linear combination of three months (last month, this month and next month). Include masking of probabilities
# 31/05/2025 SCW Bugfix on mask file selection. Updating geotiff outputs to be direct to Lancaster
import os,glob,shutil,sys, re

import netCDF4 as nc  
import process_realtime_fns as fns   #will need to move this code into sftp_extract directory
import time
import datetime

##################################################
#Set up paths (more below inside the main_code_loop and get_dat functions!)
##################################################


##################################################
#Get datype of run (historical or real-time)
##################################################
runtype = sys.argv[1]
if runtype.lower() not in ['historical','realtime']:
    print('Run type incorrectly defined. Must either be \"historical\" or \"realtime\"')
else:
    runtype = runtype.lower()



if runtype=='realtime':
   # mirror_path = '/scratch/NFLICS/sftp_extract/current'   # path to NCAS mirror 
    mirror_path = '/mnt/scratch/cmt'
    #mirror_path = '/users/hymod/stewells/NFLICS/SSA/sample_live/'
else:
    mirror_path = '/mnt/scratch/cmt'
    #mirror_path = '/scratch/NFLICS/sftp_extract/current'
      #mirror_path = '/prj/nflics/real_time_data/2024/01/08/' # path to where historical NCAS raw data to process is held

shadow_run=False
##################################################
#Error handelling using the signal package
##################################################
import signal

# run_offline: True saves data to /users/hymod/stewells/NFLICS/NFLICS_scw rather than operation folders
run_offline = False

alarm_length_ftp=60*5  #give the ftp 1 minute to work
alarm_length_code=60*10 #give the code 10 minutes per .nc file (normal operational run - 1 nc file per 15 mins)

#def handler(signum, frame):
 #   raise Exception("End of time")




##################################################
#Get database version
##################################################
db_version = sys.argv[2] if len(sys.argv)>2 else 3  # latest

##################################################
#Mirror data from NCAS sftp site
##################################################
# list the files currently in the mirror

def get_data(mirror_path,shadow_run,db_version):
    ##################################################
    #Set up paths 
    ##################################################
    print("getting the data")
    ######### testing options #######################
    user='swells'
    rt_type="UKCEH_backup" #ncas cutout

    last_set_files = glob.glob(os.path.join(mirror_path,'*IR*.nc'))
    if rt_type=="UKCEH":
        # run the lftp script to rsync the mirror directory with the NCAS server directory        
        os.system("lftp sftp://"+user+"@sci.ncas.ac.uk -e 'mirror -c --newer-than=now-3hours --exclude-glob .* --verbose /data/ "+mirror_path+";quit;'")  
             
    # get files now present
        current_files = sorted(glob.glob(os.path.join(mirror_path,'*IR*.nc')))
        
        #current_files = sorted(glob.glob(os.path.join(mirror_path,'*IR*.nc')))
        # list files that are no present that werent before
        if shadow_run==False:
           
            new_files = list(sorted(set(current_files).difference(last_set_files)))
            
			# extra check, if empty, then just check t see
            if len(new_files)==0:		
                #file_stamp =datetime.datetime.strptime(time.ctime(os.path.getmtime(current_files[-1])),'%a %b %d %H:%M:%S %Y')
                file_stamp2 = datetime.datetime.strptime(time.ctime(os.stat(current_files[-1]).st_ctime),'%a %b %d %H:%M:%S %Y')
                recent_thresh = datetime.datetime.now() - datetime.timedelta(seconds=180)
                if file_stamp2 > recent_thresh:
                    new_files = [current_files[-1]]
                   # print("Corrected current _files")
                   # print(new_files)
        else:
            print("this is a shadow run")
            new_files_temp = list(sorted(set(current_files).difference(last_set_files)))
            if len(new_files_temp)>2:
                new_files=new_files_temp[:-1]
            elif len(new_files_temp)==1:
                new_files = list([sorted(current_files)[-2]])
            else:
                new_files=new_files_temp
    elif rt_type=="UKCEH_backup": # CHris' EUMDAT processed data
        # check in Chris folder for any files made in last 20 minutes
        new_files = []
        cronFreq=20
        total_files=glob.glob(os.path.join(mirror_path,'IR_108_BT_*nc'))
        

        for f in total_files:
            modTimesinceEpoc = os.path.getmtime(f)
            modificationTime = datetime.datetime.fromtimestamp(time.mktime(time.localtime(modTimesinceEpoc)))
            if modificationTime > datetime.datetime.today()-datetime.timedelta(minutes=cronFreq):
#               only include if not already processed

                new_files.append(f)


    else:
        new_files = list(sorted(last_set_files))
    print ("new files to process:",new_files)
    return(new_files)
     




##################################################
#Loop over new files and create NFLICS nowcasts
###############################################

def main_code_loop(use_file,mirror_path,shadow_run,db_version,run_offline):
    import os,glob,shutil,sys
    import netCDF4 as nc
    print("into the main code loop")
    #########UKCEH file path options#############
    #datadir="/prj/nflics/historical_database/msg9_cell_shape_wave_rect_20040601to20190930_realtime/msg9_cell_shape_wave_rect_" #historical database
    #datadir="/prj/nflics/historical_database/date_split_WA_v2/msg9_cell_shape_wave_rect_20040601to20190930_WA_v2/msg9_cell_shape_wave_rect_"
    datadir="/mnt/prj/nflics/historical_database/date_split_WA_v2_realtime/msg9_cell_shape_wave_rect_20040601to20190930_WA_v2/msg9_cell_shape_wave_rect_"
    #datadir ="/prj/NC_Int_CCAd/3C/seodey/data/historical_database/msg9_cell_shape_wave_rect_2004001to20190930_WA_v2
	###datadir="/prj/nflics/historical_database/msg9_cell_shape_wave_rect_20040601to20190930_realtime/msg9_cell_shape_wave_rect_" #historical database
    
    if run_offline:
        plotbase="/mnt/users/hymod/stewells/NFLICS/NFLICS_scw/nflics_nowcasts/"
        daily_base="/mnt/users/hymod/stewells/NFLICS/NFLICS_scw/daily_summary_plots_test"
        scratchbase="/mnt/users/hymod/stewells/NFLICS/NFLICS_scw/nflics_current/"    #plots go here
        #rt_save="/users/hymod/stewells/NFLICS/NFLICS_scw/real_time_data/" 
        rt_save="/mnt/prj/nflics/real_time_data/"  
    else:
        scratchbase="/mnt/scratch/NFLICS/nflics_current/"
    #      #plots go here
        plotbase="/mnt/prj/nflics/nflics_nowcasts/"      #plots go here
    #
        daily_base="/mnt/prj/nflics/daily_summary_plots"
        rt_save="/mnt/prj/nflics/real_time_data/"        #archive of real time data from ncas
        lst_path="/mnt/prj/swift/SEVIRI_LST/data_anom_wrt_historic_clim_withmask"
        rt_code_input="/mnt/prj/nflics/RT_code_v2_input/"
    remote_path="/anacimcehrw/w"
    months={"05":"May","06":"June","07":"July","08":"August","09":"September"} #not sure this is actually used???

    ######### options for where the code is #######################
    nflics_base="/mnt/users/hymod/seodey/NFLICS/"    #the overall base directory
    code_base="/mnt/users/hymod/seodey/rt_data/"     #where the code is

    #feed="ncas"
    #feed = 'historical'
    feed = 'eumdat'
    ukceh_mirror=False
    make_gif=False
    overwrite = True
    ######### shadow run options #######################
    #run shadow run can be set to run 15-minutes behind the main run for contingency against machine problems
	
    #settings only used when shadow_run==True
    if run_offline:
        save_plotbase="/mnt/users/hymod/stewells/NFLICS/NFLICS_scw/nflics_nowcasts_test/"      #plots go here
        save_scratchbase="/mnt/scratch/stewells/nflics_current/"  #plots go here
		
        save_rt_save="/mnt/users/hymod/stewells/NFLICS/NFLICS_scw/real_time_data/"        #archive of real time data from ncas
    else:
        save_plotbase="/mnt/prj/nflics/nflics_nowcasts/"      #plots go here
        save_scratchbase="/mnt/scratch/NFLICS/nflics_current/"
        save_rt_save="/mnt/prj/nflics/real_time_data/"        #archive of real time data from ncas
    try:
        test = use_file
        #test=nc.Dataset(use_file) 
        use_file=use_file.split("/")[-1] #remove directory (want file name only)
        print("processing file:", use_file)
        if feed=='historical':
            use_year=use_file[:4]
            use_month=use_file[4:6]
            use_day=use_file[6:8]
            use_time=use_file[8:12] 
             
        else: 
            use_year=use_file[10:14]
            use_month=use_file[14:16]
            use_day=use_file[16:18]
            use_time=use_file[19:23]
        tnow=use_year+use_month+use_day+use_time
        
        ###############################################
        #1. copy the rt data to somewhere safe
        #print("Move observed image to database")
        savedir=os.path.join(rt_save,use_year,use_month,use_day)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        else:
            pass
        sfile =use_file[:-3]+'_eumdat.nc' if  feed=='eumdat' else use_file
        #print(sfile)
# if file already in saved folder assume processed and exit function for this file
        
 
        if not os.path.exists(os.path.join(savedir,sfile)):
            shutil.copy2(os.path.join(mirror_path,use_file),os.path.join(savedir,sfile))
        else:
            if overwrite:
                print("Overwriting previously processed file")
            else:
                print("Already processed")
                return

        vis_file = 'VIS_006_rad_'+use_year+use_month+use_day+'_'+use_time+'.nc'
        sfile = vis_file[:-3]+'_eumdat.nc' if  feed=='eumdat' else vis_file
        #print(vis_file)
        if os.path.exists(os.path.join(mirror_path,vis_file)):# visible channel exists, so copy it too
            if not os.path.exists(os.path.join(savedir,sfile)):
                shutil.copy2(os.path.join(mirror_path,vis_file),os.path.join(savedir,sfile))            

        



        ###############################################
        #2. make the nowcast directories
        #print("Process image to make nowcast")
        #create plot directory
        plotdir=os.path.join(plotbase,use_year,use_month,use_day,use_time)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
            try:
                os.system('chmod g+rwx '+plotdir)
            except:
                pass
            try:
                os.system('chmod g+rwx '+os.path.join(plotbase,use_year,use_month,use_day))
            except:
                pass
            try:
                os.system('chmod g+rwx '+os.path.join(plotbase,use_year,use_month))
            except:
                pass
            try:
                os.system('chmod g+rwx '+os.path.join(plotbase,use_year))
            except:
                pass                                              
        else:
            pass
        scratchdir=os.path.join(scratchbase,use_year,use_month,use_day,use_time)
        if not os.path.exists(scratchdir):
            os.makedirs(scratchdir)
        else:
            pass

        ###############################################
        #3. calculate nowcasts
		
		
		
        fns.process_realtime_v3(tnow,datadir,mirror_path,plotdir,scratchbase,lst_path,nflics_base,rt_code_input,feed,db_version)
            
        if ukceh_mirror==True:
            remotedir=os.path.join(remote_path,"nflics_nowcasts",use_year,use_month,use_day,use_time)
            remotedir_daily_summary=os.path.join(remote_path,"daily_summary_plots")
            daily_summary_dir=daily_base
            if shadow_run==False:
                os.system("lftp sftp://anacimcehrw@wlsftp.nwl.ac.uk -e 'mkdir -p "+remotedir+";mirror -cR --verbose "+plotdir+" "+remotedir+";quit;'")   
                os.system("lftp sftp://anacimcehrw@wlsftp.nwl.ac.uk -e 'mirror -cR --newer-than=now-24hours --verbose "+daily_summary_dir+" "+remotedir_daily_summary+";quit;'")   
            else:
                save_plotdir=os.path.join(save_plotbase,use_year,use_month,use_day,use_time)
                if not os.path.exists(save_plotdir):
                    os.system("lftp sftp://anacimcehrw@wlsftp.nwl.ac.uk -e 'mkdir -p "+remotedir+";mirror -cR --verbose "+plotdir+" "+remotedir+";quit;'")   
                    os.system("lftp sftp://anacimcehrw@wlsftp.nwl.ac.uk -e 'mirror -cR --newer-than=now-1hour --verbose "+daily_summary_dir+" "+remotedir_daily_summary+";quit;'")   
                #copy everything accross to main-run directories
                    os.system("cp -r "+os.path.join(plotbase,use_year,use_month,use_day,use_time)+" "+os.path.join(save_plotbase,use_year,use_month,use_day))
                    os.system("cp -r "+os.path.join(scratchbase,use_year,use_month,use_day,use_time)+" "+os.path.join(save_scratchbase,use_year,use_month,use_day,use_time))
                    os.system("cp -r "+os.path.join(rt_save,use_year,use_month,use_day)+"/*"+use_time+".nc "+os.path.join(save_rt_save,use_year,use_month,use_day))
        else:
            pass
        if make_gif and int(db_version)==2:
            if len(glob.glob(os.path.join(plotbase,use_year,use_month,use_day,use_time,'Nowcast_v2*.png')))>0:
                os.system('convert -delay 60 -loop 0 '+'/'.join([plotbase,use_year,use_month,use_day,use_time])+'/Nowcast_v2*.png '+'/'.join([plotbase,use_year,use_month,use_day,use_time])+'/Nowcast_v2_'+''.join([use_year,use_month,use_day,use_time])+'_000_360.gif')
            else:
                print("No Nowcasts for gif")
    except OSError as err:
        print("OPENING ERROR!",err," file ",use_file," could not be opened")


#######
#run the loops
#######

#signal.signal(signal.SIGALRM, handler)
#signal.alarm(60)
 
#get_data(mirror_path,shadow_run,db_version)
#try:
#    get_data(user,mirror_path)
#except Exception as e:
#    print(e)
#    print("ftp loop took too long")
    #signal.alarm(0)
#sys.exit(0)
#test=signal.alarm(600)
#print(test)

#try:
#    main_code_loop(new_files)
#except Exception:
#    print("Loop took too long!")
#    signal.alarm(0)


def handler(signum, frame):
    raise Exception("End of time")

if runtype=='realtime':
    signal.signal(signal.SIGALRM, handler)
    a1=signal.alarm(alarm_length_ftp)


timeStampStart= time.time()
 
# get the files
if runtype=='realtime':
    new_files = []
    try:
        new_files=get_data(mirror_path,shadow_run,db_version)
        #print(new_files)
        signal.alarm(0)
    except Exception as e:
        print(e)
        print("FTP loop took too long!")
        signal.alarm(0)
    a2=signal.alarm(alarm_length_code)
else:
    #ts = ['01','06','07','08','09','10','11']
    #ts = [str(x).zfill(2) for x in range(2,7)]
    ts = ['21']
    #ts = ['29']
    #hrs = [str(x).zfill(2) for x in range(12)]
    hrs = ['14']
    mins = ['30']
    #mins = ['00']
    new_files  = []
    for t in ts:
        for h in hrs:
            for m in mins:
                #new_files = new_files+[mirror_path+"/202207"+t+h+m+'.gra']
            	new_files=new_files+[mirror_path+"/IR_108_BT_202403"+t+"_"+h+m+'.nc']
    
    new_files= glob.glob("/scratch/cmt/IR_108_BT_20240509_0945.nc")
    #new_files = glob.glob(mirror_path+"/202208141200.gra")
    new_files = sorted(new_files)

#print(new_files)

for new_file in new_files:
    if runtype=='realtime':
        main_code_loop(new_file,mirror_path,shadow_run,db_version,run_offline)
        a2=signal.alarm(a2)
        #except Exception as e:
        #    print(e)
        #    print("Nowcast code took too long")       
        #    signal.alarm(0)
    else:
        #try:
        #print(new_file)
        main_code_loop(new_file,mirror_path,shadow_run,db_version,run_offline)
        #except Exception as e:
        #    print(e)
        #    print("Error in processing of "+new_file)

print(''.join(["time: ",str((time.time()-timeStampStart))]))                                                                                               


