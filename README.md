## Africa Nowcasting real-time scripts

This Readme describes the scripts available in the repository and their usage. Note that some scripts require additional data/files that are not contained within the repository. The real-time workflows are controlled by systemd timers and services. Details of the workflow configuration are described elsewhere; this is merely a catalogue of the scripts in use as part of the real-time system. 

### Cloud top temperature, Visible radiation, cores and nowcasts

The processing reads in CTT data (either in NetCDF format as delivered by EUMDAT or raw .gra files) and generates corresponding GeoTIFFs for the CTT, convective cores (current and recent) and nowcast probabilities. 

Script: **sat_transfer.py** <br />
Usage: sat_transfer.py [-h] [--startDate STARTDATE] [--endDate ENDDATE]  <br />
                       [--toPortal TOPORTAL] [--feed {eumdat,historical}]<br />
                       [--fStruct {direct,YMD,YM}]<br />
                       {realtime,historical}<br />
runmode (required): realtime (pick up any recently avaiable files) or historical (process a range of dates from pre-existing data)<br />
--startDate YYYYMMDDhhmm: Start date for historical processing (only necessary if running in historical mode )<br />
--endDate YYYYMMDDhhmm: Start date for historical processing (only necessary if running in historical mode )<br />
--toPortal TOPORTAL: Boolean (True or False) flag indicating if output is to be sent to the portal or saved locally. The local path is defined within the script. Default is `True`.<br />
--feed {eumdat,historical}: Source of feed for historical processing. Not required for realtime running. EUMDAT .nc files (eumdat) or raw .gra files (historical). Default is `eumdat`.<br />
--fStruct {direct,YMD,YM}: Folder structure for historical data. Root directory controlled by variable mirror_path, but data may sit in subfolders. Options are `direct` (data in mirror_path), `YMD` (data in mirror_path/YYYY/MM/DD folder) or `YM` (data in mirror_path/YYYY/MM folder). Not required for realtime running. Default is `direct`.<br />



### Lightning flashes

Lightning flash counts are supplied to the nowcasting portal in csv format, with each row in the file indicating the location of a lightning flash and a flag indicating the 15-minute window ($T_0, T_{-15}, T_{-30}, T_{-45}, T_{-60}$) in which the flash occured. The input data is a single .gra array containing the MTG flash count over the Africa domain. 

Script: **portal_lightning_pt_convert.py** <br />
Usage: python portal_lightning_pt_convert.py `<run mode>` `<<start datetime>>` `<<end datetime>>` `<<toPortal>>`<br />
`<run mode>`: 'realtime' or 'historical'. For realtime mode (default behavior if no arguments are provided) the most recent files added to the data directory are processed (CSV files created and sent to the Lancaster SAN) and archived. For historical mode, any files that exist between `<<start datetime>>` and `<<end datetime>>` inclusive will be processed (if data are available). The date and time should be provided in the form YYYYMMDDhhmm. <br />
`<<toPortal>>`: Binary value specifying whether the outputs are sent to the Nowcasting portal (1, default) or to a local path (0) specified in the script by the variables `testDir` (output GeoTIFFs) and `testArchiveDir` (archive of the raw .gra files). Note that to define this argument explicitly, `<run mode>` must also be specified (and `<<start datetime>>`, `<<end datetime>>` if `<run mode>`== `historical`).<br />
<br /><br />
**Real-time running**: The script is called via the shell script **run_lightning_alt.sh**, which in turn is called by the systemd timer afnow_lightning.timer.

