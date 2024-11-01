#!/bin/bash

# Created: 18 October 2024, Steven Wells (UKCEH)
# Purpose: Syncronise the Convective cores being saved to Wallingford Linux scratch area with an archive on 
#          JASMIN
# Notes on usage:
#           Submit as 
#           nohup ./push_cores_jasmin.sh > push_cores_jasmin.log 2>&1 &
#           
#            User is prompted for JASMIN passkey.

#           THIS SCRIPT WILL BE RUN ON USING NOHUP
#           SO WILL CONTINUE TO RUN EVEN WHEN CONNECTION TO LOCAL MACHINE IS CLOSED
#           THE INFINITE LOOP IS IN HERE
#           CHECK IF RUNNING WITH
#           ps -ef | grep portal
#           END PROCESS WITH
#           kill <PID>

# Procedure:
#       Rsync data from SCRATCHDIR in Wallingford with JASMIN REMOTEDIR
#           



# location of the cores outputted from satdev processing
SCRATCHDIR="/mnt/scratch/NFLICS/nflics_current"
#REMOTEDIR="/gws/nopw/j04/cehhmf/hmf/NFLICS/rt_cores/outputs/real_time_data/"
REMOTEDIR="/gws/nopw/j04/swift/rt_cores/"

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa_jasmin

# rsync usage
#rsync options source destination
# options: -v : verbose details of ongoing operations
#          -a : archive mode
#          -u : update - skip files that are still new in the destination directory
#          -n : dry run - dont acutally make any changes, just list what is done
#     --stats : summary of outputs

while true
do 
date > /home/stewells/AfricaNowcasting/logs/afnow_push_cores_jasmin.log
rsync -avu  --files-from=<(find "$SCRATCHDIR" -type f -mmin -5 | sed "s|^$SCRATCHDIR/||") "$SCRATCHDIR/" "swells@xfer-vm-01.jasmin.ac.uk:$REMOTEDIR/" >> /home/stewells/AfricaNowcasting/logs/afnow_push_cores_jasmin.log
sleep 120
done


