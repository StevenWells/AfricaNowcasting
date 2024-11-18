## Africa Nowcasting real-time scripts

This Readme describes the scripts available in the repository and their usage. Note that some scripts require additional data/files that are not contained within the repository. The real-time workflows are controlled by systemd timers and services. Details of the workflow configuration are described elsewhere; this is merely a catalogue of the scripts in use as part of the real-time system. 


### Lightning flashes

Lightning flash counts are supplied to the nowcasting portal in csv format, with each row in the file indicating the location of a lightning flash and a flag indicating the 15-minute window ($T_0, T_{-15}, T_{-30}, T_{-45}, T_{-60}$) in which the flash occured. The input data is a single .gra array containing the MTG flash count over the Africa domain. 

Script: **portal_lightning_pt_convert.py** <br />
Usage: python portal_lightning_pt_convert.py `<run mode>` `<<start datetime>>` `<<end datetime>>`<br />
`<run mode>`: 'realtime' or 'historical'. For realtime mode (default behavior if no arguments are provided) the most recent files added to the data directory are processed (CSV files created and sent to the Lancaster SAN) and archived. For historical mode, any files that exist between `<<start datetime>>` and `<<end datetime>>` inclusive will be processed (if data are available). The date and time should be provided in the form YYYYMMDDhhmm. <br />
**Real-time running**: The script is called via the shell script **run_lightning_alt.sh**, which in turn is called by the systemd timer afnow_lightning.timer.

