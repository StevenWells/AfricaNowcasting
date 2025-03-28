## Africa Nowcasting real-time scripts

This Readme describes the scripts available in the repository and their usage. Note that some scripts require additional data/files that are not contained within the repository. The real-time workflows are controlled by systemd timers and services. Details of the workflow configuration are described elsewhere; this is merely a catalogue of the scripts in use as part of the real-time system. 

### Cloud top temperature, Visible radiation, cores and nowcasts

The processing reads in CTT data (either in NetCDF format as delivered by EUMDAT or raw .gra files) and generates corresponding GeoTIFFs for the CTT, convective cores (current and recent) and nowcast probabilities. 

Script: **sat_transfer.py** <br />
Usage: sat_transfer.py [-h] [--startDate STARTDATE]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [--endDate ENDDATE]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
                       [--toPortal TOPORTAL]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [--feed {eumdat,historical}]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
                       [--fStruct {direct,YMD,YM}]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
                       {realtime,historical}<br />
`runmode` (required): realtime (pick up any recently avaiable files) or historical (process a range of dates from pre-existing data)<br />
`--startDate`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDhhmm<br />
`--endDate`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDDhhmm<br />
`--toPortal`: Boolean (True or False) flag indicating if output is to be sent to the portal or saved locally. The local path is defined within the script. Default is `True`.<br />
`--feed`: Source of feed for historical processing. Not required for realtime running. EUMDAT .nc files (eumdat) or raw .gra files (historical). Default is `eumdat`.<br />
`--fStruct`: Folder structure for historical data. Root directory controlled by variable mirror_path, but data may sit in subfolders. Options are `direct` (data in mirror_path), `YMD` (data in mirror_path/YYYY/MM/DD folder) or `YM` (data in mirror_path/YYYY/MM folder). Not required for realtime running. Default is `direct`.<br />
**Real-time running**: The script is called via the shell script **run_satproc.sh**, which in turn is called by the systemd timer afnow_nflics.timer.


### Lightning flashes

Lightning flash counts are supplied to the nowcasting portal in csv format, with each row in the file indicating the location of a lightning flash and a flag indicating the 15-minute window ($T_0, T_{-15}, T_{-30}, T_{-45}, T_{-60}$) in which the flash occured. The input data is a single .gra array containing the MTG flash count over the Africa domain. 

Script: **portal_lightning_pt_convert.py** <br />
usage: portal_lightning_pt_convert.py [-h] [--mode {realtime,historical}]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--startDate STARTDATE]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                                      [--endDate ENDDATE]<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                                      [--toPortal TOPORTAL]<br />
For realtime mode (default behavior if no arguments are provided) the most recent files added to the data directory are processed (CSV files created and sent to the Lancaster SAN) and archived. For historical mode, any files that exist between `startDate` and `endDate` inclusive will be processed (if data are available). The date and time should be provided in the form YYYYMMDDhhmm. <br />
`Portal`: Boolean value specifying whether the outputs are sent to the Nowcasting portal (`True`, default) or to a local path (`False`) specified in the script by the variables `testDir` (output GeoTIFFs) and `testArchiveDir` (archive of the raw .gra files). <br />
<br /><br />
**Real-time running**: The script is called via the shell script **run_lightning_alt.sh**, which in turn is called by the systemd timer afnow_lightning.timer.

### Soil Moisture Anomaly

Soil moisture anomaly is calculated externally (cmt) in .gra format, and used as input to this process which convert to GeoTIFF for display on the Nowcasting portal. There are two files, morning and afternoon (based on the satellite passes) which are updated throughout the day as more information/passes become available. 

Script: **portal_soilmoisture_convert.py**<br />

Usage:  portal_soilmoisture_convert.py [-h] [--mode {realtime,historical}]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--startDate STARTDATE]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--endDate ENDDATE]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--outDir OUTDIR]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--domain {WA,SSA}]<br /><br />
`mode`: Pick up most recent files (`realtime`) or process older files based upon a date range (`historical`)<br />
`--startDate`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDhhmm<br />
`--endDate`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDDhhmm<br />
`--outDir`: Explicity define a path to output the data to. If not included, defaults to the Lancaster SAN for display on the portal. Note that the files will be contained within the a subdirectory of `outDir` based on the year and month of the file. <br />
`--domain`: Process data covering either Sub-Saharan Africa (`SSA`) or West Africa (`WA`). Default is `SSA`.<br />
**Real-time running**: The script is called via the shell script **run_sm_convert.sh**, which in turn is called by the systemd timer afnow_sm.timer.

### Rain Over Africa Precipitation accumulations

Rain Over Africa rainfall rate is based on a machine learning approach applied to Meteosat Second Generation imagery. It is generated externally from these scripts. This process calculates accumulations for 1,3,6,12,24,48 and 72h periods based on the rainfall rate images and generates geoTIFF maps to display on the Africa Nowcasting portal.

Script:

**portal_roa_accums.py**<br />

Usage: portal_roa_accums.py [-h] [--mode {realtime,historical}]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--startDate STARTDATE]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--endDate ENDDATE]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[--dataDir DATADIR]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--tmpDir TMPDIR]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--geotiffDir GEOTIFFDIR]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--reprocess REPROCESS]<br />
`mode`: Pick up most recent files (`realtime`) or process older files based upon a date range (`historical`)<br />
`--startDate`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDhhmm<br />
`--endDate`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDDhhmm<br />
`--geotiffDir`: Explicity define a path to output the data to. If not included, defaults to the Lancaster SAN for display on the portal. Note that the files will be contained within the a subdirectory of `geotiffDir` based on the year, month and day of the file. <br />
`--tmpDir`: Explicity define a path to hold intemeidate temporary files. They are removed after each geoTIFF is generated. If not included, defaults to the local drive where the processing is taking place (~/AfricaNowcasting/tmp/). <br />
`--dataDir`: Explicity define a path to the rain over Africa archived precipitation rate data. If not included, defaults to the  SWIFT project space. Note that the files will be contained within the a subdirectory of `geotiffDir` based on the year, month and day of the file. <br />
**Real-time running**: The script is called via the shell script **run_roa_accums.sh**, which in turn is called by the systemd timer afnow_roacc.timer.

### HSAF Precipitation rate and accumulations

Rainfall rate produced in near-real time by the Hydrology Satellite Applications Facility (HSAF) is converted into GeoTIFF format for display on the Nowcasting portal. An accumulated rainfall layer is also created over a range of periods from 1 hour to three days.

Script: **portal_hsafprecip_convert.py**<br />

Usage: portal_hsafprecip_convert.py [-h] [--mode {realtime,historical}]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[--startDate STARTDATE]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--endDate ENDDATE] <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--outDir OUTDIR]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--makeAccumulations MAKEACCUMULATIONS]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--reprocess REPROCESS]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--getPoints GETPOINTS]<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--keepMSG KEEPMSG]<br />
`mode`: Pick up most recent files (`realtime`) or process older files based upon a date range (`historical`)<br />
`--startDate`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDhhmm<br />
`--endDate`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDDhhmm<br />
`--outDir`: Explicity define a path to output the data to. If not included, defaults to the Lancaster SAN for display on the portal. Note that the files will be contained within the a subdirectory of `outDir` based on the year and month of the file. <br />
`--makeAccumulations`: Boolean argument. In addition to precipitation rate for that timestamp, generate the accumulated rainfall for 1hr, 3hr, 6hr, 12hr, 24hr and 72hr prior to that timestamp. Default is `True`. <br />
`--reprocess`: Boolean argument. If `True`, reprocess the file even if it has already been processed. Default is `False`. Useful for infilling historical data where missing data is intermittent.<br />
`--getPoints`: Boolean argument. Extract values of rainfall rate from image at set of (hard-coded) locations and save to .csv file. Default is `False`. <br />
`--keepMSG`: Boolean argument. Retain the full domain image as downloaded from the HSAF sftp and archive. Default is `True`. <br /> <br />
**Real-time running**: The script is called by the script **run_hsaf_precip_convert.sh** , which is controlled by the systemd timer afnow_lmf.timer.

### Land Modification Factor

This product provides a quantitative estimate of how the likelihood of intense convective rainfall is modified by the land surface. It is derived from a combination of daytime Land Surface Temperature (LST) anomalies, observed from Meteosat Second Generation (MSG), and historical data encapsulating the statistical relationships between convective cores and LST anomalies (LSTA). The raw .gra files are converted into GeoTIFF for display on the Africa Noecasting portal.

Script: **lmf_convert.sh** 

Usage: ./lmf_convert.sh [-m, --mode] [-d, --outdir] [-s, --startDir] [-e, --enddir]<br /><br />
`[-m, --mode]`: Pick up most recent files (`realtime`) or process older files based upon a date range (`historical`)<br />
`[-s, --startDate]`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDm<br />
`[-e, --endDate]`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDD<br />
`[-d, --outDir]`: Explicity define a path to output the data to. If not included, defaults to the Lancaster SAN for display on the portal. Note that the files will be contained within the a subdirectory of `outDir` based on the year and month of the file. <br />
<br /> 
**Real-time running**: The script is called directly by the systemd timer afnow_lmf.timer.


### LSTA conversion to GeoTIFF

Convert the daily mean LSTA NetCDF file into GeoTIFF for display on the Africa Noecasting portal.

Script: **lst_convert.sh** 

Usage: ./lst_convert.sh [-m, --mode] [-d, --outdir] [-s, --startDir] [-e, --enddir] [-p, --reprocess]<br /><br />
`[-m, --mode]`: Pick up most recent files (`realtime`) or process older files based upon a date range (`historical`)<br />
`[-s, --startDate]`: Start date for historical processing (only necessary if running in historical mode ). Format is  YYYYMMDDm<br />
`[-e, --endDate]`: Start date for historical processing (only necessary if running in historical mode ). Format is YYYYMMDD<br />
`[-d, --outDir]`: Explicity define a path to output the data to. If not included, defaults to the Lancaster SAN for display on the portal. Note that the files will be contained within the a subdirectory of `outDir` based on the year and month of the file. <br />
`[-p, --reprocess]`: Process the file even if the output GeoTIFF already exists. Default is `false`. <br />
<br /> 
**Real-time running**: The script is called directly by the systemd timer afnow_lsta.timer.