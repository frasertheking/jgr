#!/usr/bin/env python
# -*- coding: utf-8 -*-
%matplotlib inline

"""@author: Fraser King
   @date: 2019

   @desc:
   General extractor used to compare CloudSat data with 
   station and reanalysis estimates of SWE. Statistics and 
   general figure plotting included throughout.
"""

##################################################
########## IMPORTS

import sys, os, datetime, json, math, glob, csv
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import calendar
import copy
import matplotlib.animation as animation
from netCDF4 import Dataset
from datetime import timedelta
from datetime import datetime
from matplotlib.animation import FFMpegWriter
from collections import deque
from class_def import * 

##################################################
########## CONFIGURATION

### SELECT A STATION:
desc = ["eureka", 0.722390, 159, 188, ((80.5, -85.5), (79.5, -85.5), (79.5, -86.5), (80.5, -86.5)), 40, 60, 38, 188]
#desc = ["resolute", 0.541174, 149, 170, ((75.5, -94.5), (74.5, -94.5), (74.5, -95.5), (75.5, -95.5)), 40, 60, 28, 170]
#desc = ["cambridgebay", 0.766129, 138, 149, ((69.5, -104.5), (68.5, -104.5), (68.5, -105.5), (69.5, -105.5)), 40, 60, 18, 148]
#desc = ["iqaluit", 0.766672, 127, 222, ((64.5, -68.5), (63.5, -68.5), (63.5, -69.5), (64.5, -69.5)), 40, 120, 6, 222]
base_station = BaseStation(desc[0], desc[1], desc[2], desc[3], desc[4], desc[5], desc[6], desc[7], desc[8])

### DATA PATHS:
path = '/Volumes/Samsung_T5/Data/csa/CloudSat_decade/decade/all/r05/' + base_station.station
station_name = base_station.station.capitalize()
save_csv_path = "sacc_" + base_station.station + ".csv"
clim_image_path_1 = "../images/clim_" + base_station.station + "_1.gif"
clim_image_path_2 = "../images/clim_" + base_station.station + "_2.gif"
clim_image_path_3 = "../images/clim_" + base_station.station + "_3.gif"
time_image_path_1 = "../images/time_" + base_station.station + "_1.gif"
time_image_path_2 = "../images/time_" + base_station.station + "_2.gif"
time_image_path_3 = "../images/time_" + base_station.station + "_3.gif"
precip_path = "/Volumes/Samsung_T5/Data/csa/CloudSat_decade/decade/" + base_station.station + "_eccc_precip.csv"
temp_path = "/Volumes/Samsung_T5/Data/csa/CloudSat_decade/decade/" + base_station.station + "_eccc_temp.csv"
snow_path = "/Volumes/Samsung_T5/Data/csa/CloudSat_decade/decade/old/" + base_station.station + "_eccc_snow.csv"
rain_path = "/Volumes/Samsung_T5/Data/csa/CloudSat_decade/decade/" + base_station.station + "_eccc_rain.csv"
blended_path = "/Volumes/Samsung_T5/Data/csa/SWE/blended4"
asr_path = "../grids/climatology/asrv2_05x05.nc"
asrv1_path = "../grids/climatology/asrv1_05x05.nc"
asr_path_monthly = "../grids/timeseries/asr_05x05.nc"
blended_std_path = "/Volumes/Samsung_T5/Data/csa/SWE/blended4/std"

##################################################
########## GLOBALS

periods = []
pre_2011 = []
post_2011 = []
climatology = []
colors = ["#0c2be5", "#ed008c", "#d0191b", "#f06730", "#f08622", "#e9eb28", "#b4e742", "#5fc650", "#761ca2", "#b23593", "#ba54ff", "#ed293b"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
year_names = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
days_in_period = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

##################################################
########## FLAGS

use_snow = False
generate_anim = False
temp_thresh = 5 # Degrees C

##################################################
########## INITILIZATION

for i in range(0, 12):
    periods.append(Period(month_names[i], "Month", days_in_period[i]))
    pre_2011.append(Period(month_names[i], "Month", days_in_period[i]))
    post_2011.append(Period(month_names[i], "Month", days_in_period[i]))

for i in range(0, 108):
    climatology.append(Period((str((i % 12) + 1) + str(year_names[math.floor(i/12)])), "Month", days_in_period[i % 12]))

##################################################
########## HELPER FUNCTIONS

def drawAnimatedLineGraph(fig, ax, x, y, c, marker, title, xlab, ylab, grid_flag, xmin, xmax, ymin, ymax, font_size, width, height, fname, fill_lower, fill_upper):
    """Used in the generation of animated line graphs.
    Takes in a bunch of custom plotting params.
    """
    
    line, = ax.plot(x, y[0], color=c, marker=marker, linewidth=3)
    plt.title(title)
    plt.xlabel(xlab, fontsize=font_size)
    plt.ylabel(ylab, fontsize=font_size)
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(num, x, y, line):
        line.set_data(x, y[num])
        line.axes.axis([xmin, xmax, ymin, ymax])
        if not(fill_lower is None): 
            ax.collections.clear()
            ax.fill_between(x, fill_lower[num], fill_upper[num], facecolor=c, alpha = 0.2)
        return line,

    ani = animation.FuncAnimation(fig, update, len(y), fargs=[x, y, line], interval=30, blit=True, repeat=False)
    ani.save(fname)
    
def drawAnimatedBarLineGraph(fig, ax, x, y, bar_width, c, title, xlab, ylab, grid_flag, xmin, xmax, ymin, ymax, font_size, width, height, fname):
    """Used in the generation of animated bar graphs.
    Takes in a bunch of custom plotting params.
    """
    
    barcollection = ax.bar(x, y[0], bar_width, color=c)
    plt.title(title)
    plt.xlabel(xlab, fontsize=font_size)
    plt.ylabel(ylab, fontsize=font_size)

    def update(num):
        for i,b in enumerate(barcollection):
            b.set_height(y[num][i])
        return barcollection

    ani = animation.FuncAnimation(fig, update, len(y), interval=30, blit=True, repeat=False)
    ani.save(fname)
    
def drawAnimatedScatter(fig, ax, x, y, c, title, xlab, ylab, grid_flag, xmin, xmax, ymin, ymax, font_size, width, height, fname):
    """Used in the generation of animated scatterplot.
    Takes in a bunch of custom plotting params.
    """
    
    line, = ax.plot(x, y[0], color=c, marker="o", ls="")
    plt.title(title)
    plt.xlabel(xlab, fontsize=font_size)
    plt.ylabel(ylab, fontsize=font_size)
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(num, x, y, line):
        line.set_data(x, y[num])
        return line,

    ani = animation.FuncAnimation(fig, update, len(y), fargs=[x, y, line], interval=23, blit=True, repeat=False)
    ani.save(fname)

def getYStepArray(y_end):
    """ Custom linspace (interp) function used for gap
    filling in animated figure generation.
    """
    
    y_step = [[] for x in range(len(y_end))]
    steps=47
    for i,value in enumerate(y_end):
        if math.isnan(value):
            y_step[i] = [math.nan for x in range(48)]
        elif value == 0:
            y_step[i] = [0 for x in range(48)]
        else:
            y_step[i] = np.linspace(0, value, 48) 

    return [[y_step[j][i] for j in range(len(y_step))] for i in range(len(y_step[0]))]

def get_gridbox():
    """Getter for station bounds.
    """
    
    return base_station.bounds

def is_point_in_grid(point, rect):
    """Basic check for whether point lies within gridbox.
    """
    
    if ((point[0] < rect[3][0] and point[0] > rect[2][0]) and \
        (point[1] > rect[3][1] and point[1] < rect[0][1])):
        return True
    return False

def rmse(predictions, targets):
    """Getter for model RMSE.
    """
    
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

def count_leaps(year_start, year_end):
    leap_years = 0
    for i in range (year_start, year_end):
        if calendar.isleap(i):
            leap_years += 1
    return leap_years

def find_month_by_name(name, group):
    """Retrieval of month from period object 
    (object equivalence overwriter basically)
    """
    
    count = 0
    for month in group:
        if month.name == name:
            return count
        count += 1
        
def snow_frac(temp):
    """Linear interp of rain snow fraction basic on 
    work done by Ross Brown, 2003.
    """
    
    if temp <= 0:
        return 1
    else:
        return 1 - (temp / temp_thresh)
    
##################################################
########## MAIN CLOUDSAT

custom_grid = get_gridbox()

def cloudsat_parse(startBoundary, endBoundary):
    """Parsing script to extract CloudSat data form the
    raw granuals (saved in csv) to the proper period object
    based on the grid size for the selected station (bounds).
    """
    
    print("Running snowfall parse...")
    for filename in os.listdir(path):
        if filename.endswith(".csv"): 
            cloudsat_path = os.path.join(path, filename)
            cloudsat_data = pd.read_csv(cloudsat_path)
            max_snowfall_rate = cloudsat_data['Surface_Snowfall_Rate'].max()
            current_month = -1
            current_year = -1
            pre_2011_event = -1
            post_2011_event = -1
            monthname = -1

            # limit selection to period of interest (overlapping with ECCC)
            if (cloudsat_data.iloc[0]['Year'] <= startBoundary) or (cloudsat_data.iloc[0]['Year'] >= endBoundary):
                continue     

            for index, row in cloudsat_data.iterrows():
                if not(is_point_in_grid((row['Cloudsat_Lat'], row['Cloudsat_Lon']), custom_grid)):
                    continue

                day = (datetime(row['Year'], 1, 1) + timedelta(row['Day_of_Year'] - 1)).day
                week = (datetime(row['Year'], 1, 1) + timedelta(row['Day_of_Year'] - 1)).isocalendar()[1] # week
                month = (datetime(row['Year'], 1, 1) + timedelta(row['Day_of_Year'] - 1)).month
                year = (datetime(row['Year'], 1, 1) + timedelta(row['Day_of_Year'] - 1)).year
                time = row['UTC_Time'] / 3600
                
                current_month = month
                current_year = year
                color = colors[month-1]
                period_array = periods[month-1]
                            
                # Time series comparison
                if period_array.event_count < len(period_array.snowfall_events):
                    period_array.snowfall_events[period_array.event_count].append(row['Surface_Snowfall_Rate'])
                else:
                    period_array.snowfall_events.append([row['Surface_Snowfall_Rate']])

                # Climatologcial comparison
                monthname = str(month) + str(year)
                clim_month = climatology[find_month_by_name(monthname, climatology)]
                
                if clim_month.event_count < len(clim_month.snowfall_events):
                    clim_month.snowfall_events[clim_month.event_count].append(row['Surface_Snowfall_Rate'])
                else:
                    clim_month.snowfall_events.append([row['Surface_Snowfall_Rate']])    

            if current_month > 0:
                periods[current_month - 1].event_count += 1
            
            if monthname != -1:
                climatology[find_month_by_name(monthname, climatology)].event_count += 1
                
            continue
        else:
            continue

    print("CloudSat Parse Complete!")
    

### Run cloudsat data extraction and grouping
cloudsat_parse(2006, 2016)

### Analysis of CloudSat soundings 
all_medians = []
count_all = False

def analyze_soundings(multiplier, period_group):
    """Look through each individual radar pulse recorded
    by CloudSat during an overpass within the defined grid box
    and extract some statistics.
    """
    
    # Cleanup of soundings with negative snowfall rates
    for period in period_group:
        count = 0
        for event in period.snowfall_events:
            adjusted_soundings = []
            for sounding in event:
                if sounding >= 0:
                    adjusted_soundings.append(sounding) 
            period.snowfall_events[count] = adjusted_soundings
            count += 1

    # Calculate accumulation values for period_groups
    for i,period in enumerate(period_group):
        h = 0
        means = []
        for event in period.snowfall_events:
            if len(event) > 0:
                means.append(np.median(event))
                if count_all:
                    all_medians.append(np.median(event))

        if len(means) > 0:
            period.mean_snowfall_rate = np.mean(means)
            period.sd = np.std(means, ddof=1) 
            period.se = period.sd / math.sqrt(len(means))
            h = period.se * sp.stats.t._ppf((1+0.95)/2., len(means)-1)
        else:
            period.mean_snowfall_rate = math.nan
            period.sd = math.nan
            period.se = math.nan
            h = math.nan

        period.CI_upper = period.mean_snowfall_rate + h
        period.CI_lower = period.mean_snowfall_rate - h
        period.accumulation = (period.mean_snowfall_rate*24*period.days_in_period * multiplier) * base_station.lc_frac # year multiplier
        period.accumulation_upper = (period.CI_upper*24*period.days_in_period * multiplier) * base_station.lc_frac
        period.accumulation_lower = (period.CI_lower*24*period.days_in_period * multiplier) * base_station.lc_frac

analyze_soundings(1, periods)
count_all = True
analyze_soundings(1, climatology)
    

### Run cloudsat data extraction and grouping
cloudsat_parse(2006, 2016)

### Analysis of CloudSat soundings 
all_medians = []
count_all = False

def analyze_soundings(multiplier, period_group):
    """Look through each individual radar pulse recorded
    by CloudSat during an overpass within the defined grid box
    and extract some statistics.
    """
    
    # Cleanup of soundings with negative snowfall rates
    for period in period_group:
        count = 0
        for event in period.snowfall_events:
            adjusted_soundings = []
            for sounding in event:
                if sounding >= 0:
                    adjusted_soundings.append(sounding) 
            period.snowfall_events[count] = adjusted_soundings
            count += 1

    # Calculate accumulation values for period_groups
    for i,period in enumerate(period_group):
        h = 0
        means = []
        for event in period.snowfall_events:
            if len(event) > 0:
                means.append(np.median(event))
                if count_all:
                    all_medians.append(np.median(event))

        if len(means) > 0:
            period.mean_snowfall_rate = np.mean(means)
            period.sd = np.std(means, ddof=1) 
            period.se = period.sd / math.sqrt(len(means))
            h = period.se * sp.stats.t._ppf((1+0.95)/2., len(means)-1)
        else:
            period.mean_snowfall_rate = math.nan
            period.sd = math.nan
            period.se = math.nan
            h = math.nan

        period.CI_upper = period.mean_snowfall_rate + h
        period.CI_lower = period.mean_snowfall_rate - h
        period.accumulation = (period.mean_snowfall_rate*24*period.days_in_period * multiplier) * base_station.lc_frac # year multiplier
        period.accumulation_upper = (period.CI_upper*24*period.days_in_period * multiplier) * base_station.lc_frac
        period.accumulation_lower = (period.CI_lower*24*period.days_in_period * multiplier) * base_station.lc_frac

analyze_soundings(1, periods)
count_all = True
analyze_soundings(1, climatology)
    


##################################################
########## PLOTTING MAIN

def plot_results(blended_periods, blended_accumulations, asrv1_sacc, asr_sacc, objects, eccc_data, multiplier, ylim, title):
    """General plotting function to display climatological and time series
    comparison between CloudSat, ECCC and the various gridded products used
    in this analysis.
    """
        
    plt.rcParams.update({'font.size': 22})
    cloudsat_accumulations_per_month = []
    bottom_region = []
    top_region = []
    bottom_region_blended = []
    top_region_blended = []
    
    # Read through our period data
    for period in blended_periods:
        cloudsat_accumulations_per_month.append(period.accumulation)
        bottom_region.append(period.accumulation_lower)
        top_region.append(period.accumulation_upper)
        
    for blended_object in objects: # month
        means = blended_object.yearly_accumulations
        blended_object.sd = np.std(means, ddof=1) 
        blended_object.se = blended_object.sd / math.sqrt(len(means))
        h = blended_object.se * sp.stats.t._ppf((1+0.95)/2., len(means)-1)
        
        blended_object.CI_upper = blended_object.total_accumulation*multiplier + h
        blended_object.CI_lower = blended_object.total_accumulation*multiplier - h

        top_region_blended.append(blended_object.CI_upper)
        bottom_region_blended.append(blended_object.CI_lower)
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] # was overwritten
    month_names = np.roll(month_names, 3)
    cloudsat_accumulations_per_month = np.roll(cloudsat_accumulations_per_month, 3)
    blended_accumulations = np.roll(blended_accumulations, 3)
    eccc_data = np.roll(eccc_data, 3)
    top_region = np.roll(top_region, 3)
    bottom_region = np.roll(bottom_region, 3)
    asr_sacc = np.roll(asr_sacc, 3)
    asrv1_sacc = np.roll(asrv1_sacc, 3)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.grid()
    ax.set_title(title)
    ax.set_ylabel('Accumulation (mm SWE)')
    ax.set_xlabel('Month')
    ax.fill_between(month_names, bottom_region, top_region, color="#d35e60", alpha=0.2)
    ax.plot(month_names, cloudsat_accumulations_per_month, marker='o', color="#d35e60", linewidth=3)
    ax.plot(month_names, blended_accumulations, marker='o', color="#84ba5b", linewidth=3)
    ax.plot(month_names, asr_sacc, marker='o', color="#f9b000", linewidth=3)
    ax.plot(month_names, asrv1_sacc, marker='o', color="#4dc4c4", linewidth=3)
    ax.plot(month_names, eccc_data, marker='o', color="black", linewidth=4)
    
    ### Winter Means
    eccc_winter = np.mean(eccc_data[2:3])
    cs_winter = np.mean(cloudsat_accumulations_per_month[2:3])
    b4_winter = np.mean(blended_accumulations[2:3])
    
    ### Summer Means
    eccc_summer = np.mean(eccc_data[8:10])
    cs_summer = np.mean(cloudsat_accumulations_per_month[8:10])
    b4_summer = np.mean(blended_accumulations[8:10])
    
    print("WINTER MEAN INFORMATION:")
    print("ECCC", eccc_winter)
    print("CloudSat", cs_winter)
    print("Blended-4", b4_winter)
    
    print("SUMMER MEAN INFORMATION:")
    print("ECCC", eccc_summer)
    print("CloudSat", cs_summer)
    print("Blended-4", b4_summer)
    print()
    
    ################################################### SPONGE
    ### GET ANNUAL AVERAGES AND CONFIDENCE INTERVALS
    average = np.mean(cloudsat_accumulations_per_month)
    sd = np.std(cloudsat_accumulations_per_month)
    se = sd / math.sqrt(len(cloudsat_accumulations_per_month))
    me = se * sp.stats.t.ppf((1+0.95)/2., len(cloudsat_accumulations_per_month)-1)
    
    average_b4 = np.mean(blended_accumulations)
    sd_b4 = np.std(blended_accumulations)
    se_b4 = sd_b4 / math.sqrt(len(blended_accumulations))
    me_b4 = se_b4 * sp.stats.t.ppf((1+0.95)/2., len(blended_accumulations)-1)
    
    ### ASRV2 SE Calcs
    average_asrv2 = np.mean(asr_sacc)
    sd_asrv2 = np.std(asr_sacc)
    se_asrv2 = sd_asrv2 / math.sqrt(len(asr_sacc))
    me_asrv2 = se_asrv2 * sp.stats.t.ppf((1+0.95)/2., len(asr_sacc)-1)
    
    ### ASRV1 SE Calcs
    average_asrv1 = np.mean(asrv1_sacc)
    sd_asrv1 = np.std(asrv1_sacc)
    se_asrv1 = sd_asrv1 / math.sqrt(len(asrv1_sacc))
    me_asrv1 = se_asrv1 * sp.stats.t.ppf((1+0.95)/2., len(asrv1_sacc)-1)
    
    print("CSAT:", average, me)
    print("B4:", average_b4, me_b4)
    print("ECCC", np.mean(eccc_data))
    print("ASRV1", np.mean(asrv1_sacc), me_asrv1)
    print("ASRV2", np.mean(asr_sacc), me_asrv2)
    
    eccc_avg = mlines.Line2D([0, 108], [np.mean(eccc_data), np.mean(eccc_data)], color='black', linestyle='dashed')
    blended_avg = mlines.Line2D([0, 108], [np.mean(blended_accumulations), np.mean(blended_accumulations)], color='#84ba5b', linestyle='dashed')
    asr_avg = mlines.Line2D([0, 108], [np.mean(asr_sacc), np.mean(asr_sacc)], color='#f9b000', linestyle='dashed')
    asrv1_avg = mlines.Line2D([0, 108], [np.mean(asrv1_sacc), np.mean(asrv1_sacc)], color='#4dc4c4', linestyle='dashed')
    cloudsat_avg = mlines.Line2D([0, 108], [average, average], color='#d35e60', linestyle='dashed')
    ax.add_line(eccc_avg)
    ax.add_line(blended_avg)
    ax.add_line(cloudsat_avg)
    ax.add_line(asr_avg)
    ax.add_line(asrv1_avg)
    ax.set_ylim([0, ylim])
    ax.set_xlim([0, 11])
    
    print("CLIM Correlation " + station_name)
    corr_df = pd.DataFrame({'CloudSat': cloudsat_accumulations_per_month, 'Blended': blended_accumulations, 'ASRV2': asr_sacc, 'ASRV1': asrv1_sacc, 'ECCC': eccc_data})
    print(corr_df.corr())
    
    rmse_val_blended = rmse(np.array(eccc_data), np.array(blended_accumulations))
    rmse_val_cloudsat = rmse(np.array(eccc_data), np.array(cloudsat_accumulations_per_month))
    rmse_val_cb = rmse(np.array(blended_accumulations), np.array(cloudsat_accumulations_per_month))
    rmse_asrv2 = rmse(np.array(asr_sacc), np.array(cloudsat_accumulations_per_month))
    print("\nRMSE Blended-4: " + str(rmse_val_blended) + " (mm SWE)")
    print("RMSE Cloudsat : " + str(rmse_val_cloudsat) + " (mm SWE)")
    print("RMSE Cloudsat vs. Blended : " + str(rmse_val_cb) + " (mm SWE)")
    print("RMSE Cloudsat vs. ASR : " + str(rmse_asrv2) + " (mm SWE)")
    
    if generate_anim:
        print("Creating animations")
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.rcParams.update({'font.size': 22})
        plt.grid()
        ax.legend(handles=[red_patch, green_patch, blue_patch])
        drawAnimatedLineGraph(fig, ax, month_names, getYStepArray(cloudsat_accumulations_per_month), '#d35e60', 'o', station_name + " Monthly SWE Climatology (1 degree)", 'Month', 'Accumulation (mm SWE)', True, 0, 11, 0, base_station.ymax_clim, 22, 14, 10, clim_image_path_1, getYStepArray(bottom_region), getYStepArray(top_region))
        drawAnimatedLineGraph(fig, ax, month_names, getYStepArray(eccc_data), '#7293cb', 'o', station_name +" Monthly SWE Climatology (1 degree)", 'Month', 'Accumulation (mm SWE)', True, 0, 11, 0, base_station.ymax_clim, 22, 14, 10, clim_image_path_2, None, None)
        drawAnimatedLineGraph(fig, ax, month_names, getYStepArray(blended_accumulations), '#84ba5b', 'o', station_name + " Monthly SWE Climatology (1 degree)", 'Month', 'Accumulation (mm SWE)', True, 0, 11, 0, base_station.ymax_clim, 22, 14, 10, clim_image_path_3, None, None)

plot_results(blended4_periods, blended4_accumulations_per_month, asrv1_monthly_sacc, asr_monthly_sacc, blended_objects, eccc_accumulations_per_month, 9, base_station.ymax_clim, station_name)


##################################################
########## INTERANNUAL VARIABILITY

datenames = []
cloudsat_values = []
monthly_overpass_count = []

for period in climatology:
    datenames.append(period.name)
    cloudsat_values.append(period.accumulation)
    monthly_overpass_count.append(period.event_count)

##################################################
########## ECCC MAIN

df = pd.read_csv(precip_path, index_col = False)
df_rain = pd.read_csv(rain_path, index_col = False)
df_temps = pd.read_csv(temp_path, index_col = False)

if (use_snow):
    df = pd.read_csv(snow_path, index_col = False)

snowfall_amounts = [[] for x in range(108)]
eccc_accumulations_per_month = np.zeros(108)
eccc_missing_days_per_month = np.zeros(108)

def get_eccc_accumulation(startBoundary, endBoundary):
    """Extract station weather params and organize them
    for comparison with CloudSat and the gridded products.
    Extracting interannual variability here.
    """
    
    starting_loc = df.columns.get_loc("DESC")+1
    ending_loc = len(df.columns)
    count = 1
    
    for row in df.itertuples(index=False, name='Pandas'):
        for i in range(starting_loc, ending_loc):
            month = row[3]
            year = row[2]
            
            if year <= startBoundary or year >= endBoundary:
                continue
            
            temperature = df_temps[df_temps.columns.get_values()[i]][count-1]
            rain = df_rain[df_rain.columns.get_values()[i]][count-1]
            
            if use_snow:
                temperature = 0
                
            if (row[i] >= 0):
                if use_snow or (not(use_snow) and temperature <= temp_thresh): 
                    if rain < 0:
                        if temperature <= -999:
                            eccc_missing_days_per_month[row[0]-12] += 1
                            snowfall_amounts[row[0]-12].append(np.nan)
                        else:
                            snowfall_amounts[row[0]-12].append(row[i] * snow_frac(temperature))   
                    else:
                        snowfall_amounts[row[0]-12].append(row[i] - rain)  
                elif not(use_snow):                
                    snowfall_amounts[row[0]-12].append(0)
            elif row[i] < 0:
                if (i-5) < days_in_period[month-1]: # Don't count days that don't exist in a month as missing
                    eccc_missing_days_per_month[row[0]-12] += 1
                    snowfall_amounts[row[0]-12].append(np.nan)
        count += 1
                
get_eccc_accumulation(2006, 2016)

# Gap filling
for i,amount in enumerate(snowfall_amounts):
    accum = np.nanmean(amount)*days_in_period[i%12]
    if math.isnan(accum):
        eccc_accumulations_per_month[i] = 0
    else:
        eccc_accumulations_per_month[i] = accum

##################################################
########## ASR MAIN

print("Gettings ASR netCDF contents..")
asr_monthly_sacc_discrete = []
f = Dataset(asr_path_monthly)

# Extract netCDF variables
day = f.variables['time'][:]
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
swe = f.variables['sacc'][:]

for i in range (0, day.shape[0]):
    swe_matrix = swe[i]
    asr_monthly_sacc_discrete.append(np.mean([swe_matrix[base_station.station_x][base_station.station_y],
                                    swe_matrix[base_station.station_x][base_station.station_y + 1],
                                    swe_matrix[base_station.station_x + 1][base_station.station_y],
                                    swe_matrix[base_station.station_x + 1][base_station.station_y + 1]]))
    

##################################################
########## BLENDED-4 MAIN
    
blended_accumulations_per_month = np.zeros(108)
month_pos = 0
for blended_object in blended_climatology:  
    accumulation_diffs = np.asarray([j-i for i, j in zip(blended_object.accumulations[:-1], blended_object.accumulations[1:])])
    accumulation_diffs = accumulation_diffs[accumulation_diffs >= 0]
    blended_accumulations_per_month[month_pos] = np.sum(accumulation_diffs)
    month_pos += 1   

##################################################
### FIX FOR BAD CLOUDSAT MONTHS:
### Sep to Dec 2009
### Jan 2011
### May 2011 to April 2012
### We mask periods where CloudSat has no data due to battery failures 

bad_indices = [32, 33, 34, 35, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
for i in range(108):    
    if i in bad_indices:
        cloudsat_values[i] = np.nan 
        eccc_accumulations_per_month[i] = np.nan 
        blended_accumulations_per_month[i] = np.nan 
        asr_monthly_sacc_discrete[i] = np.nan

##################################################
########## ANNUAL MEANS
annual_timeseries_csat = [[] for x in range(9)]
annual_timeseries_eccc = [[] for x in range(9)]
annual_timeseries_blen = [[] for x in range(9)]
annual_mean_timeseries_csat = []
annual_mean_timeseries_eccc = []
annual_mean_timeseries_blen = []
for i in range(108):
    annual_timeseries_csat[math.floor(i/12)].append(cloudsat_values[i])
    annual_timeseries_eccc[math.floor(i/12)].append(eccc_accumulations_per_month[i])
    annual_timeseries_blen[math.floor(i/12)].append(blended_accumulations_per_month[i])

for i in range(9):
    annual_mean_timeseries_csat.append(np.nanmean(annual_timeseries_csat[i])*12)
    annual_mean_timeseries_eccc.append(np.nanmean(annual_timeseries_eccc[i])*12)
    annual_mean_timeseries_blen.append(np.nanmean(annual_timeseries_blen[i])*12)

##################################################
########## INTERANNUAL VARIABILITY PLOTS
    
bottom_region = []
top_region = []

for period in climatology:
    if period.accumulation_lower < 0:
        bottom_region.append(0)
    else:
        bottom_region.append(period.accumulation_lower)
    top_region.append(period.accumulation_upper)

fig4, ax = plt.subplots(figsize=(14, 10))
plt.grid()
ax.set_title(base_station.station)
ax.set_ylabel('Accumulation (mm SWE)')
ax.set_xlabel('Date')
ax.fill_between(datenames, bottom_region, top_region, color="#d35e60", alpha=0.2)
ax.plot(datenames, blended_accumulations_per_month, color="#84ba5b", linewidth=3)
#ax.plot(datenames, asr_monthly_sacc_discrete, color="#f9b000", linewidth=2)
ax.plot(datenames, cloudsat_values, color="#d35e60", linewidth=3)
ax.plot(datenames, eccc_accumulations_per_month, color="black", linewidth=4)
#ax.plot(datenames, grid_values, color="black", linewidth=1)
_ = ax.set_xticklabels(year_names)
_ = ax.xaxis.set_ticks(np.arange(0, 108, 12))
ax.set_ylim([0, 50])
ax.set_xlim([0, 107])

### MEAN LINES
eccc_avg = mlines.Line2D([0, 108], [np.nanmean(eccc_accumulations_per_month), np.nanmean(eccc_accumulations_per_month)], color='#7293cb', linestyle='dashed', alpha=0.5)
blended_avg = mlines.Line2D([0, 108], [np.nanmean(blended_accumulations_per_month), np.nanmean(blended_accumulations_per_month)], color='#84ba5b', linestyle='dashed', alpha=0.5)
cloudsat_avg = mlines.Line2D([0, 108], [np.nanmean(cloudsat_values), np.nanmean(cloudsat_values)], color='#d35e60', linestyle='dashed', alpha=0.5)
asr_avg = mlines.Line2D([0, 108], [np.nanmean(asr_monthly_sacc_discrete), np.nanmean(asr_monthly_sacc_discrete)], color='#f9b000', linestyle='dashed', alpha=0.5)
ax.add_line(eccc_avg)
ax.add_line(blended_avg)
ax.add_line(cloudsat_avg)
ax.add_line(asr_avg)

##################################################
########## SCATTER PLOTS

cbay_csat = copy.deepcopy(cloudsat_values)
cbay_eccc = copy.deepcopy(eccc_accumulations_per_month)
cbay_blended = copy.deepcopy(blended_accumulations_per_month)

ylim = 100
fig, ax = plt.subplots(figsize=(10, 10))
plt.grid()
line = mlines.Line2D([0, ylim], [0, ylim], color='black')
ax.add_line(line)
ax.set_title("ECCC vs. CloudSat")
ax.set_ylabel('CloudSat Accumulation (mm SWE)')
ax.set_xlabel('ECCC Accumulation (mm SWE)')
ax.plot(cloudsat_values, blended_accumulations_per_month, color='red', marker='o', linewidth=0, markersize=10)
ax.plot(cbay_csat, cbay_blended, color='red', marker='o', linewidth=0, markersize=10)
ax.plot(eccc_accumulations_per_month, blended_accumulations_per_month, color='red', marker='o', linewidth=0, markersize=10)
ax.plot(eccc_accumulations_per_month, cloudsat_values, color='red', marker='o', linewidth=0, markersize=10)
ax.set_xlim((0, ylim))
ax.set_ylim((0, ylim))
ax.set_xscale("symlog")
ax.set_yscale("symlog")


##################################################
########## STATION OVERPASS SUMMARY STAT PLOTS

fig, ax = plt.subplots(figsize=(10, 10))
plt.grid()
ax.set_title("Station Overpass Counts")
ax.set_xlabel('Year')
ax.set_ylabel('Monthly Overpass Count')
_ = ax.set_xticklabels(year_names)
_ = ax.xaxis.set_ticks(np.arange(0, 108, 12))
ax.set_ylim([0, 40])
ax.set_xlim([0, 107])

# CBAY count is calculated from a previous station run (you can do this for any station pair)
# Not usually hard coded but allows us to plot two separate stations against one another in one run
cbay_count = [6, 6, 6, 5, 5, 8, 8, 8, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 5, 4, 6, 6, 6, 6, 6, 5, 6, 8, 7, 7, 8, 4, 0, 0, 0, 0, 3, 6, 6, 6, 7, 6, 8, 6, 7, 6, 6, 6, 0, 3, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 7, 6, 6, 7, 4, 6, 3, 7, 7, 5, 8, 6, 8, 7, 5, 7, 7, 7, 6, 5, 7, 7, 8, 5, 5, 7, 6, 6, 6, 6, 6, 5, 6, 6, 7, 8, 8, 8, 6, 7, 6, 6]
ax.bar(np.arange(0, 108), cbay_count, 1, color='blue')
ax.bar(np.arange(0, 108), monthly_overpass_count, 1, bottom=cbay_count, color="red")
red_patch = mpatches.Patch(color='blue', label='Cambridge Bay')
blue_patch = mpatches.Patch(color='red', label='Eureka')
ax.legend(handles=[blue_patch, red_patch])
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


##################################################
########## SUMMARY STATISTICS OUTPUT

print("Summary statistics output:")

print("\nCorrelation:")
corr_df = pd.DataFrame({'CloudSat': cloudsat_values, 'Blended': blended_accumulations_per_month, 'ECCC': eccc_accumulations_per_month})
print(corr_df.corr())

rmse_val_blended = rmse(np.array(eccc_accumulations_per_month), np.array(blended_accumulations_per_month))
rmse_val_cloudsat = rmse(np.array(eccc_accumulations_per_month), np.array(cloudsat_values))
rmse_val_cb = rmse(np.array(blended_accumulations_per_month), np.array(cloudsat_values))
print("\nRMSE Blended-4: " + str(rmse_val_blended) + " (mm SWE)")
print("RMSE Cloudsat : " + str(rmse_val_cloudsat) + " (mm SWE)")
print("RMSE Cloudsat vs. Blended : " + str(rmse_val_cb) + " (mm SWE)")

print("\n\nMissing data & overpass counts:")
print("\nMissing ECCC days per month:", eccc_missing_days_per_month, np.sum(eccc_missing_days_per_month))
print("\nOverpass counts per month:", monthly_overpass_count, np.sum(monthly_overpass_count))

print("\n\nAverages:")
print("\nECCC:", np.nanmean(eccc_accumulations_per_month))
print("\nCloudSat:", np.nanmean(cloudsat_values))
print("\nBlended:", np.nanmean(blended_accumulations_per_month))


