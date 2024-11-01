# Move the accumulations from SAN to backup on S drive that are more than 1 month old
import os, sys, glob
import datetime




inDir = '/mnt/HYDROLOGY_stewells/geotiff/ssa_hsaf_precip_accum'
outDir= '/mnt/data/hmf/projects/LAWIS/WestAfrica_portal/portal_archive/geotiff/ssa_hsaf_precip_accum'


tnow = datetime.datetime.now()
tminus_31d = tnow - datetime.timedelta(days=31)
tminus_40d = tnow - datetime.timedelta(days=40)


alldirs = sorted([x for x in  glob.glob(inDir+'/*') if (datetime.datetime.strptime(x.split('/')[-1],"%Y%m%d") < tminus_31d) and (datetime.datetime.strptime(x.split('/')[-1],"%Y%m%d") > tminus_40d) ])

for idir in alldirs:
    print(idir)
    for acc in [1,3,6,48]:
        #print(acc)
        all_files=glob.glob(os.path.join(idir,'HSAF_precip_acc'+str(acc)+'h*'))
        for f in all_files:
            subdir = f.split('/')[-2]
            newdir = os.path.join(outDir,subdir)
            os.makedirs(newdir,exist_ok=True)
            newfile = os.path.join(outDir,subdir,f.split('/')[-1])
            os.system('mv '+f+' '+newfile)


 

