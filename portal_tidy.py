# Remove from the Portal any data in the set provided that is older than a certain period
# The raw data for creating the tiffs is stored elsewhere so can be recreated

# Caveats

# Allowances need to be made for keeping data within certain "event" periods. This might be solved by the 
# "lower bound" options, although not watertight 
# The HSAF accumulations are to be remvoed, except for the midnight daily accumulation


import os, sys, glob
import datetime


# SAN root
sanRoot = '/mnt/HYDROLOGY_stewells/geotiff/'
cutoff = 101 # days


tnow = datetime.datetime.now()
tminus_cutoff = tnow  -datetime.timedelta(days=cutoff)
tminus_lowerbound = tminus_cutoff - datetime.timedelta(days=40)

# products to be remvoed
# these names should be the folder names in sanRoot
ProdsToCull = ['ssa_hsaf_precip','ssa_hsaf_precip_accum','lawis_nowcasts',
               'ssa_africarain_precip','ssa_africarain_precip_accum','lawis_visible_channel']

for iprod in ProdsToCull:
    print(iprod)
    ppath = os.path.join(sanRoot,iprod)
    alldirs = sorted([x for x in  glob.glob(ppath+'/*') if (datetime.datetime.strptime(x.split('/')[-1],"%Y%m%d") < tminus_cutoff) and (datetime.datetime.strptime(x.split('/')[-1],"%Y%m%d") > tminus_lowerbound) ])
    if iprod == 'ssa_hsaf_precip_accum':
        for deldir in alldirs:
            for acc in [1,3,6,48,72]:
                all_files=glob.glob(os.path.join(deldir,'HSAF_precip_acc'+str(acc)+'h*'))
                for ifile in all_files:
                    print('rm '+ifile)
                    os.system("rm "+ifile)
        print("need to keep specific files")
    else:
        for idir in alldirs:
            print('rm -r '+idir)
            os.system('rm -r '+idir)

