#!/usr/bin/env python
# -*- coding: utf-8 -*-
%matplotlib inline

"""@author: Fraser King
   @date: 2019

   @desc:
   General plotting library for visualizing
   geographical comparison between CloudSat, ECCC and
   the gridded SWE products. 
"""

##################################################
########## IMPORTS

import sys,os,warnings,math,glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import pandas as pd
import matplotlib.patches as mpatches
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import figure
from scipy.stats.kde import gaussian_kde
from numpy import linspace

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

##################################################
########## PLOTTING MAIN

def plot_netcdf(filepath, name, var, title, units, colorscale, vmin, vmax, steps, show_zonal_averages, show_neg_pos, plt_min, plt_max, i):
    """We take in a variety of plotting parameters
    and display these on a Basemap image of the northern
    hemisphere. This function can be tweaked to display
    many different types of data.
    """

    # Open netCDF file to plot
    fh = Dataset(filepath, mode='r')
    lons = fh.variables['lon'][:]
    lats = fh.variables['lat'][:]
    sacc = fh.variables[var][:]    
    sacc = sacc[i] # Extract a certain timestep
    
    lon = lons.mean()
    lat = lats.mean()
    
    # Setup basemap
    m = Basemap(width=7500000,height=7500000, \
                resolution='c',projection='stere',\
                lat_ts=90,lat_0=90,lon_0=0)
    
    lon, lat = np.meshgrid(lons, lats)
    
    ## Fix for plotting offset
    lon = lon-0.5
    lat = lat-0.5
    
    xi, yi = m(lon, lat)
    fig = plt.figure(figsize=(13, 13))
    plt.rcParams.update({'font.size': 18})
    cs = m.pcolor(xi,yi,np.squeeze(sacc),cmap=plt.cm.get_cmap(colorscale, steps), vmin=vmin, vmax=vmax)
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10, dashes=[5, 5], linewidth=0.25)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10, dashes=[5, 5], linewidth=0.25)
    m.drawcoastlines(linewidth=2.0)
    m.drawcountries()
    m.drawlsmask(land_color='#d0d0d0',ocean_color='#f0f0f0',)
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(units)
    plt.title(title)
    plt.show()
    
    # Display the zonal average values of the input param
    if show_zonal_averages:
        fig2, ax = plt.subplots(figsize=(14,8))
        plt.grid()
        ax.set_title(title)
        ax.set_ylabel(units)
        ax.set_xlabel('Latitude')
        lat_pos = np.arange(60,90,1)
        lat_bin = [np.nan for x in range(30)]
        for row in sacc:
            for i, lat in enumerate(row):
                if not(math.isnan(np.mean(lat))):
                    lat_bin[i] = np.nanmean(lat)
        ax.set_xlim((60, 82))
        ax.set_ylim((plt_min, plt_max))
        plt.plot(lat_pos, lat_bin, linewidth=3, color="#d35e60")
        
        if show_neg_pos:
            lat_top = [0 for x in range(30)]
            lat_bottom = [plt_min for x in range(30)]
            ax.fill_between(lat_pos, lat_top, lat_bottom, color="#7293cb", alpha=0.1)
            plt.plot(lat_pos, lat_top, color="#7293cb", linewidth=1)
    
    # Close the netCDF file
    fh.close()


for i in range(108):
    plot_netcdf('grids/timeseries/cloudsat_1x1_frland.nc', 'sacc', 'CloudSat Monthly Sacc Time Series', 'Snow Accumulation (mm SWE)', 'Blues', 0, 50, 6, False, False, -16, 16, i)


