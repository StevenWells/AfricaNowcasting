#-------------------------------------------------------------------------------
# Name:       plot_source_destination_functions.py
# Purpose:    FUnctions from program plot_source_destination.py 
#
# Author:      seodey, CEH Wallingford
#
# Created:     27/03/2019
#
# taken as subset from test_nflics_functions.py to speed up load in of functions
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans
import pandas as pd
import os,pickle
import numpy.ma as ma


def MSGpix2geo(column,row,domain):
    import numpy as np

    if domain=="WAfrica":
        lr_row=2000
        lr_col=824
    elif domain=="SSA":
        lr_row=638
        lr_col=320
    elif domain=="DakarStrip":
        lr_row=2000
        lr_col=2464
    else:
        print("ERROR! Please select a configured domain to use!")
        return

    column=column+lr_col  #824 WAfrica, 2464 Dakar strip
    row=row+lr_row   #2000 WAfrica, 2000 Dakar strip
    #define parameters needed
    ITwoToSixteen = 2**16
    RTwoToSixteen = float(ITwoToSixteen)
    RRTwoToSixteen = 1.0/RTwoToSixteen
    PI = 3.14159265359

    #These values taken from EUMETSAT document CGMS03 "LRIT/HRIT Global Specification".
    SAT_HEIGHT = 42164.0                   # Satellite distance from centre of Earth (km).
    R_EQ       = 6378.169                  # Radius of Earth at equator (km).
    R_POL      = 6356.5838                 # Radius of Earth at poles (km).
    SUB_LON    = 0.0                       # Longitude of Sub-Satellite Point (degrees).
    Q2 = (R_EQ*R_EQ)/(R_POL*R_POL)         # = 1.006803 (-).
    RQ2= 1.0/Q2                            # = 0.993243  (-).
    D2 = SAT_HEIGHT*SAT_HEIGHT - R_EQ*R_EQ # = 1737121856.0 (km^2).
    RAT1 = 1.0 - (R_POL*R_POL)/(R_EQ*R_EQ) # = 0.00675701 (-).

    #These are the values applicable to the full disc.
    #Negative FAC values indicate South-to-North and East-to-West 
    #scanning directions.  i.e. pixel (1,1) is South-Easternmost point.
    #See "LRIT/HRIT Mission Specific Implementation" (EUM/MSG/SPE/057)
    #for derivation of CFAC, LFAC, COFF, LOFF.
    CFAC_FD = -781651420 # Coefficient of image spread in E-W direction (/ra.
    LFAC_FD = -781651420 # Coefficient of image spread in N-S direction (/rad).
    COFF_FD = 1856       # Column offset of centre pixel on full disc.
    LOFF_FD = 1856       # Line offset of centre pixel on full disc.


    # Use either user-defined or default full-disc offsets of 
    #centre pixel and image spread parameters.
    ccoff = COFF_FD
    lloff = LOFF_FD
    ccfac = CFAC_FD
    llfac = LFAC_FD

    c = column
    l = row

    # Calculate viewing angle of the satellite by use of the equation 
    # on page 28, Ref [1].
    x = float(RTwoToSixteen * (c - ccoff))/ccfac
    y = float(RTwoToSixteen * (l - lloff))/llfac

    #  Now calculate the inverse projection using equations on page 25, Ref. [1]  
    #
    #  First check for visibility, whether the pixel is located on the earth 
    #  surface or in space. 
    #  To do this calculate the argument to sqrt of "sd", which is named "sa". 
    #  If it is negative then the pixel will be located in space, otherwise all 
    #  is fine and the pixel is located on the Earth surface.

    sa = (SAT_HEIGHT * np.cos(x) * np.cos(y) )**2 - (np.cos(y)*np.cos(y) + Q2 * np.sin(y)*np.sin(y)) * D2
    if ( sa <= 0.0 ):
	    latitude  = -999.999
	    longitude = -999.999
    else:
    #Now calculate the rest of the formulas using eq. on page 25 Ref [1].
	    sd = np.sqrt(sa)
	    sn = (SAT_HEIGHT * np.cos(x) * np.cos(y) - sd) /( np.cos(y)*np.cos(y) + Q2 * np.sin(y)*np.sin(y) ) 

	    s1 = SAT_HEIGHT - sn * np.cos(x) * np.cos(y)
	    s2 = sn * np.sin(x) * np.cos(y)
	    s3 = -sn * np.sin(y)

	    sxy = np.sqrt( s1*s1 + s2*s2 )
    #Using the previous calculations now the inverse projection can be
    # calculated, which means calculating the lat./lon. from the pixel
    #row and column by equations on page 25, Ref [1].
	    loni = np.arctan(s2/s1 + SUB_LON)
	    lati = np.arctan((Q2*s3)/sxy)

    #Convert from radians into degrees.
	    latitude  = lati*180./PI
	    longitude = loni*180./PI
    return latitude,longitude

#-------------------------------------------------------------------------------
# Wraper function for MSGpix2geo(column,row):
#-------------------------------------------------------------------------------
def get_geoloc_grids(nx,ny,interp,domain):

    if interp==True:
        print("Calculating interpolated grid")
        lats=np.ndarray(shape=(ny+2,nx+2)) #shape is no. rows by no. colmns ligning up with the data
        lons=np.ndarray(shape=(ny+2,nx+2))
        for i in range(0,nx+2):  	       #loop over columns
            for j in range(0,ny+2):  #loop over rows
                lat,lon=MSGpix2geo(i-1,j-1,domain)  #(i,j) is column, row
                lats[j,i]=lat   #lats is row, column
                lons[j,i]=lon   #lons is row, column
        lats=lats[:,::-1]
        lons=lons[:,::-1]

        #calculate the pixel edge points
        lats=lats[0:-1,:]+0.5*np.diff(lats,axis=0)
        lats=lats[:,0:-1]+0.5*np.diff(lats,axis=1)
        lons=lons[0:-1,:]+0.5*np.diff(lons,axis=0)
        lons=lons[:,0:-1]+0.5*np.diff(lons,axis=1)
    else:
        print("Calculating non-interpolated grid")
        lats=np.ndarray(shape=(ny,nx)) #shape is no. rows by no. colmns ligning up with the data
        lons=np.ndarray(shape=(ny,nx))
        for i in range(0,nx):  	       #loop over columns
            for j in range(0,ny):  #loop over rows
                lat,lon=MSGpix2geo(i,j,domain)  #(i,j) is column, row
                lats[j,i]=lat  #lats is row, column
                lons[j,i]=lon  #lons is row, column
        lats=lats[:,::-1]
        lons=lons[:,::-1]

    return lats, lons



def get_rect_pt(lons_mid,lats_mid,rect_ll): 
        rect_pt=[np.where((lons_mid<=rect_ll[1]) & (lats_mid<=rect_ll[0]))[0][-1],        
                 np.where((lons_mid<=rect_ll[1]) & (lats_mid<=rect_ll[0]))[1][-1],
                 np.where((lons_mid>=rect_ll[3]) & (lats_mid>=rect_ll[2]))[0][0],        
                 np.where((lons_mid>=rect_ll[3]) & (lats_mid>=rect_ll[2]))[1][0]]
        #print rect_ll, rect_pt
        return rect_pt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def onclick(event):    
    global coords
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('Selected point x = %d, y = %d'%(
        ix, iy))
    coords.append((ix, iy))



def get_area_grid(lats,lons,plot_lims):
        #function to get a grid of grid areas
        plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
	           np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

        minlat=plot_lims_p[0]
        maxlat=plot_lims_p[2]
        minlon=plot_lims_p[1]
        maxlon=plot_lims_p[3]

        m = Basemap(projection='cea',lat_0=0.,lon_0=0., resolution='h',
                    llcrnrlon=plot_lims[1],llcrnrlat=plot_lims[0],
                    urcrnrlon=plot_lims[3],urcrnrlat=plot_lims[2])
        X, Y = m(lons,lats)

        diffX=np.diff(X,axis=1) #this is the difference allong longitudes (dx)
        diffY=np.diff(Y,axis=0) #this is the difference allong latitudes (dY)
     
        approxA=diffX[:-1,:]*diffY[:,:-1]/1000000  #ignore diferences in other direction for simplicity
                                         #    (3 orders of magnitude smaller over 1 grid saquaer)
        return(approxA)

def get_dfc(lats,lons,plot_lims):
#function to get a grid of distances from coast in km
        plot_lims_p=[np.where(lats[:,0]>plot_lims[0])[0][0],np.where(lons[0,:]>plot_lims[1])[0][0],
                  np.where(lats[:,0]<plot_lims[2])[0][-1],np.where(lons[0,:]<plot_lims[3])[0][-1]]

        minlat=plot_lims_p[0]
        maxlat=plot_lims_p[2]
        minlon=plot_lims_p[1]
        maxlon=plot_lims_p[3]

        m = Basemap(projection='cea',lat_0=0.,lon_0=0., resolution='h',
                   llcrnrlon=plot_lims[1],llcrnrlat=plot_lims[0],
                    urcrnrlon=plot_lims[3],urcrnrlat=plot_lims[2])
        X, Y = m(lons,lats)
        masked_dat=maskoceans(lons,lats,Y,inlands=False)
        loc=ma.notmasked_edges(masked_dat,1)  #x indices for the coast line
        ind_arr=np.indices(np.shape(Y))[1]
        dfc_pts=(ind_arr.transpose() - loc[0][1]).transpose()  #distance from coast in grid poionts
        dfc_m=(X.transpose() - X[loc[0]]).transpose()          #distance from coast in m
        dfc_km=dfc_m/1000
        return(dfc_pts,dfc_km)



