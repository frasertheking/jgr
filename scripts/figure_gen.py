#!/usr/bin/env python
# -*- coding: utf-8 -*-
%matplotlib inline

"""@author: Fraser King
   @date: 2019

   @desc:
   General plotting library for displaying
   CloudSat Reflectivities, Cloud Mask, EXMWF Temperatures,
   Snowfall rates, as well as scatter plots of averages and
   other various figures.
"""

##################################################
########## IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
import math
from enum import Enum
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches

##################################################
########## ENUM AND VAR SETUP

# PlotType used for tracking CloudSat products to display
class PlotType(Enum):
    REFL = 1
    MASK = 2
    TEMP = 3
    SNOW = 4
    POWR = 5
    SFCS = 6

flatten = lambda l: [item for sublist in l for item in sublist]
plt.rcParams.update({'font.size': 22})

##################################################
########## CLOUDSAT PLOTTING

def genCloudSatFigure(filepath, data_field, title, label, plot_type, cmap):
    """Ingest CloudSat overpasses (HDF) and plot
    parameters of interest like snowfall rates as
    observed by the CPR.
    """

    # Load and unpack HDF
    f = HDF(filepath) 
    vs = f.vstart() 
    Latitude = vs.attach('Latitude')
    Longitude = vs.attach('Longitude')
    Time = vs.attach('Profile_time')
    UTC = vs.attach('UTC_start')
    
    if plot_type == PlotType.SFCS:
        snowfall_rate_sfc = vs.attach('snowfall_rate_sfc')
        c = snowfall_rate_sfc[:]
        c = flatten(c)
        snowfall_rate_sfc.detach()
    
    if plot_type == PlotType.TEMP:
        EC_Height = vs.attach('EC_height')
        b = EC_Height[:]
        b = flatten(b)
        EC_Height.detach()

    a = Time[:]
    a = flatten(a)
    d = Longitude[:]
    d = flatten(d)
    utc_start = UTC[0][0]
    
    Latitude.detach() # "close" the vdata
    Longitude.detach() # "close" the vdata
    Time.detach() # "close" the vdata
    UTC.detach() # "close" the vdata
    vs.end() # terminate the vdata interface
    f.close() 

    #---------- Read HDF Files ----------#

    cpr_2b_geoprof = SD(filepath, SDC.READ)
    offset = 28000 # Position along granule
    span = 1000 # Plot length

    if plot_type != PlotType.SFCS:
        if plot_type == PlotType.REFL or plot_type == PlotType.MASK or plot_type == PlotType.SNOW:
            cpr_2b_geoprof_height = cpr_2b_geoprof.select('Height')
            cpr_2b_geoprof_height_data = cpr_2b_geoprof_height.get()

        cpr_2b_geoprof_radar_reflectivity = cpr_2b_geoprof.select(data_field)
        cpr_2b_geoprof_radar_reflectivity_data = cpr_2b_geoprof_radar_reflectivity.get()

        if plot_type == PlotType.TEMP or plot_type == PlotType.SNOW:
            cpr_2b_geoprof_radar_reflectivity_data[cpr_2b_geoprof_radar_reflectivity_data < 0] = math.nan
        elif plot_type == PlotType.POWR:
            cpr_2b_geoprof_radar_reflectivity_data[cpr_2b_geoprof_radar_reflectivity_data < 0] = math.nan

        fillvalue = 15360
        missing = -8888

        img = np.zeros((span,125))

        if plot_type == PlotType.REFL:
            img.fill(-30)

        factor = 1
        if plot_type == PlotType.REFL:
            factor = 0.01

        for i in np.arange(span):
            for j in np.arange(125):
                if plot_type == PlotType.TEMP:
                    k = int( 125 * (b[j] + 5000) / 35000 )
                else:
                    k = int( 125 * (cpr_2b_geoprof_height_data[i+offset,j] + 5000) / 35000 )

                if cpr_2b_geoprof_radar_reflectivity_data[i+offset,j] > -3000 and \
                    cpr_2b_geoprof_radar_reflectivity_data[i+offset,j] < 2100:
                    img[i,k] = cpr_2b_geoprof_radar_reflectivity_data[i+offset,j] * factor

        # Begin plotting granule
        fig = plt.figure(figsize=(18, 6))
        ax = plt.subplot(111)
        im = ax.imshow(img.T, interpolation='bilinear', cmap=cmap, origin='lower', extent=[0,200,-10,60])

        plt.title(title)
        plt.ylabel('Height (km)')
        plt.xlabel('Time')
        plt.ylim(0,20)
        pylab.yticks([0,5,10,15,20],[0,5,10,15,20])
        position_tick_labels = [str(round(a[offset]+utc_start, 3)), str(round(a[offset+200]+utc_start, 3)), str(round(a[offset+400]+utc_start, 3)), str(round(a[offset+600]+utc_start, 3)), str(round(a[offset+800]+utc_start, 3)), str(round(a[offset+1000]+utc_start, 3))]
        pylab.xticks([0,40,80,120,160, 200], position_tick_labels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)

        if plot_type == PlotType.MASK:
            plt.colorbar(im, cax=cax, boundaries=[-10,0,10,20,30,40], ticks=[-10,0,10,20,30,40], label=label)
        elif plot_type == PlotType.TEMP:
            plt.colorbar(im, cax=cax, label=label, boundaries=[200, 220, 240, 260, 280], ticks=[200, 220, 240, 260, 280])
        else:
            plt.colorbar(im, cax=cax, label=label)

        plt.savefig("cloudsat_radar_reflectivity.png")
        plt.show()
    else:
        fig = plt.figure(figsize=(15.5, 2.4))
        c = c[offset:offset+span]
        index = np.arange(len(c))
        plt.plot(index, c, color="#6699ff")
        plt.fill(index, c, color="#6699ff")
        plt.grid()
        plt.title("CloudSat Surface Snowfall")
        plt.xlabel("Time")
        plt.ylabel("Rate (mm / hr)")
        plt.ylim(0,3)
        plt.xlim(0,span)
        
        position_tick_labels = [str(round(a[offset]+utc_start, 3)), str(round(a[offset+200]+utc_start, 3)), str(round(a[offset+400]+utc_start, 3)), str(round(a[offset+600]+utc_start, 3)), str(round(a[offset+800]+utc_start, 3)), str(round(a[offset+1000]+utc_start, 3))]
        pylab.xticks([0,200,400,600,800,1000], position_tick_labels)
        
        plt.show()

# Run plot for various CloudSat plotting
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_2B-GEOPROF_GRANULE_P_R05_E02_F00.hdf', 'Radar_Reflectivity', 'CloudSat Reflectivity Profile', 'dBZ', PlotType.REFL, cm.jet)
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_2B-GEOPROF_GRANULE_P_R05_E02_F00.hdf', 'CPR_Cloud_mask', 'CloudSat Cloud Mask', 'conf', PlotType.MASK, cm.bwr)
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_ECMWF-AUX_GRANULE_P_R05_E02_F00.hdf', 'Temperature', 'ECMWF Temperatures', 'K', PlotType.TEMP, cm.tab20b)
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_2C-SNOW-PROFILE_GRANULE_P_R04_E02.hdf', 'snowfall_rate', 'CloudSat Snowfall Rates', 'mm / hr', PlotType.SNOW, cm.jet)
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_2C-SNOW-PROFILE_GRANULE_P_R04_E02.hdf', 'snowfall_rate', 'CloudSat Surface Snowfall Rate', 'mm / hr', PlotType.SNOW, cm.jet)
genCloudSatFigure('../h5/figure1/2007281095505_07690_CS_2C-SNOW-PROFILE_GRANULE_P_R04_E02.hdf', 'snowfall_rate_sfc', 'CloudSat Surface Snowfall Rate', 'mm / hr', PlotType.SFCS, cm.jet)

##################################################
########## CANADA BASEMAP PLOT

# Show the station locations used in this study
m = Basemap(width=3000000,height=2500000, \
                resolution='c',projection='stere',\
                lat_ts=10,lat_0=73,lon_0=-80)

fig = plt.figure(figsize=(13, 10))
plt.rcParams.update({'font.size': 18})
m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10, dashes=[5, 5], linewidth=0.25)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10, dashes=[5, 5], linewidth=0.25)
m.drawcoastlines(linewidth=0.25)
m.drawcountries()
m.drawlsmask(land_color='#d0d0d0',ocean_color='#f0f0f0',)

lon = [-85.93, -94.97, -105.14, -68.54]
lat = [79.99, 74.72, 69.11, 63.75]
labs = ["Eureka (WEU)", "Resolute Bay (YRB)", "Cambridge Bay (YCB)", "Iqaluit (XFB)"]
x,y = m(lon, lat)
m.plot(x, y, 'or', markersize=15)

for i in range(4):
    plt.text(x[i]+50000,y[i]+50000, labs[i], weight='bold')

plt.show()

##################################################
########## CLIMATOLOGICAL MEAN SWE PLOTS

# Station data is generated in swe_main and plotted here
x = np.arange(31)
station_names = ['Eureka', 'Resolute Bay', 'Cambridge Bay', 'Iqaluit']
eureka = [-1, -1, 4.99, 8.10, 7.23, 9.09, -1, -1]
eureka_e = [-1, -1, 1.84, 3.08, 3.48, 3.76, -1, -1]
resolute = [-1, -1, 6.26, 9.34, 11.95, 7.3, -1, -1]
resolute_e = [-1, -1, 3.49, 3.51, 5.52, 3.19, -1, -1]
cambridge = [-1, -1, 4.08, 11.04, 8.72, 10.83, -1, -1]
cambridge_e = [-1, -1, 1.81, 4.11, 5.07, 5.30, -1, -1]
iqaluit = [-1, -1, 8.25, 19.36, 21.5, 22.79, -1, -1]
iqaluit_e = [-1, -1, 5.05, 8.00, 10.66, 9.39, -1, -1]
markers = ['.', '.', 'o', 'o', 'o', 'o', '.', '.']
colors = ['white', 'white', '#d35e60', '#84ba5b', '#4dc4c4', '#f9b000', 'white', 'white']
name_arr = ['', '', 'CloudSat', 'Blended-4', 'ASRV1', 'ASRV2', '', '']

eccc_vals = np.array([-1, -1, 3.53, -1, -1, -1, -1, -1,
        -1, -1, 6.25, -1, -1, -1, -1, -1,  
        -1, -1, 5.52, -1, -1, -1, -1, -1,
        -1, -1, 12.2 -1, -1, -1, -1, -1])

# Combine to master
all_vals = eureka+resolute+cambridge+iqaluit
all_e = eureka_e+resolute_e+cambridge_e+iqaluit_e

# DIsplay plot
fig, ax = plt.subplots(figsize=(14, 10))
plt.grid(axis='y')
ax.set_xlim((0,31))
ax.set_ylim((0,31))
ax.set_title("Snow Accumulation Comparison")
ax.set_ylabel('Mean Annual Accumulation (mm SWE / month)')
ax.set_xlabel('Station')
_ = ax.set_xticklabels(station_names)
_ = ax.xaxis.set_ticks(np.arange(4, 32, 8))

for i in x:
    if i < 9:
        plt.errorbar(i, all_vals[i], yerr=all_e[i], fmt=markers[i%8], color=colors[i%8], markersize=20, label=name_arr[i%8])
    else:
        plt.errorbar(i, all_vals[i], yerr=all_e[i], fmt=markers[i%8], color=colors[i%8], markersize=20)

plt.scatter(x, eccc_vals, color="black", marker="x", s=600, label="ECCC", zorder=1000)
ax.legend(loc='upper left')


