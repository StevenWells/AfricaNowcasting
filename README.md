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
<br />
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

