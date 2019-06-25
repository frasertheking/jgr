#!/usr/bin/env python
# -*- coding: utf-8 -*-
%matplotlib inline

"""@author: Fraser King
   @date: 2019

   @desc:
   Script to organize CloudSat soundings by 
   cell into a master grid for easy spatio-
   derived access later on in our analysis.
"""

##################################################
########## IMPORTS

import glob
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from netCDF4 import Dataset
import os
import sys
import math

##################################################
########## GLOBALS

lat_range = (60.25, 82.25)
lon_range = (-179.75, 180.25)

lats = np.arange(lat_range[0], lat_range[1], 0.5)
lons = np.arange(lon_range[0], lon_range[1], 0.5)
months_from_jan_2007 = list(range(108))
cols = ['lat', 'lon', 'month', 'ovrps_count', 'sacc', 'me', 'uncert', 'qual', 'conf']
master_df = pd.DataFrame(columns=cols)
days_in_period = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

##################################################
########## MAIN RUNLOOP

# Loop through all locations in NH CloudSat grid
for c, lat in enumerate(lats):
    for lon in lons:
        master_df = pd.DataFrame(columns=cols)
        print("Working on: (" + str(lat) + ", " + str(lon) + ") : " + str(c) + " / " + str(31900))
        filepath = '../master/half_degree_cells/' + str(lat) + '_' + str(lon) + '_cell.csv'

        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            months = [[] for i in range(108)]
            for index, row in df.iterrows():
                if row['year'] == 2016 or row['year'] == 2006 or row['rate'] <= -999:
                    continue
                months[(int(row['month']) + (12*(int(row['year'])-2007)))-1].append((row['utc_start'], row['rate'], row['rate_uncert'], row['confidence'], row['quality']))

            # Group by month
            for i, month in enumerate(months):
                month.sort(key=lambda tup: tup[0])
                prev_start = -1
                overpass_median_rates = []
                overpass_median_uncertainties = []
                overpass_sounding_rates = []
                overpass_uncertainty_rates = []
                overpass_confidences = []
                overpass_qualities = []
                
                for sounding in month:
                    if prev_start == -1:
                        prev_start = sounding[0]
                    
                    overpass_confidences.append(sounding[3])
                    overpass_qualities.append(sounding[4])
                    
                    if prev_start == sounding[0]:
                        overpass_sounding_rates.append(sounding[1])
                        overpass_uncertainty_rates.append(sounding[2])
                    else:
                        overpass_median_rates.append(np.median(overpass_sounding_rates))
                        overpass_median_uncertainties.append(np.median(overpass_uncertainty_rates))
                        overpass_sounding_rates = []
                        overpass_uncertainty_rates = []
                        overpass_sounding_rates.append(sounding[1])
                        overpass_uncertainty_rates.append(sounding[2])
                        prev_start = sounding[0]
            
                if len(overpass_sounding_rates) > 0:
                    overpass_median_rates.append(np.median(overpass_sounding_rates))
                if len(overpass_uncertainty_rates) > 0:
                    overpass_median_uncertainties.append(np.median(overpass_uncertainty_rates))
            
                # Setup for missing vals
                monthly_accumulation = -9999
                monthly_uncertainty = -9999
                monthly_h = -9999
                monthly_confidence = -9999
                monthly_quality = -9999
                hours_in_month = days_in_period[(i % 12)] * 24
                
                if len(overpass_median_rates) > 1:
                    sd = np.std(overpass_median_rates, ddof=1) 
                    se = sd / math.sqrt(len(overpass_median_rates))
                    monthly_h = se * sp.stats.t._ppf((1+0.95)/2., len(overpass_median_rates)-1) * hours_in_month
                    monthly_accumulation = np.mean(overpass_median_rates) * hours_in_month
                    monthly_uncertainty = np.mean(overpass_median_uncertainties) * hours_in_month
                elif len(overpass_median_rates) == 1:
                    monthly_accumulation = np.mean(overpass_median_rates) * hours_in_month
                    monthly_uncertainty = np.mean(overpass_median_uncertainties) * hours_in_month
                
                if len(overpass_confidences) > 0:
                    monthly_confidence = int(sp.stats.mode(overpass_confidences)[0][0])
                
                if len(overpass_qualities) > 0:
                    monthly_quality = int(sp.stats.mode(overpass_qualities)[0][0])
           
                # Save to a DF
                master_df = master_df.append({'lat':lat,
                                              'lon':lon,
                                              'month':i,
                                              'ovrps_count':len(overpass_median_rates),
                                              'sacc':monthly_accumulation,
                                              'me':monthly_h,
                                              'uncert':monthly_uncertainty,
                                              'qual':monthly_quality,
                                              'conf':monthly_confidence}, ignore_index=True)
        else: # Missing
            for i in range(108): 
                master_df = master_df.append({'lat':lat, 
                                          'lon':lon,
                                          'month':i,
                                          'ovrps_count':0,
                                          'sacc':-9999,
                                          'me':-9999,
                                          'uncert':-9999,
                                          'qual':-9999,
                                          'conf':-9999}, ignore_index=True)

        # Save DF to CSV for later use
        master_df.to_csv('../master/half_degree_stats/' + str(lat) + '_' + str(lon) + '_stats.csv', index=None, header=['lat', 'lon', 'month', 'ovrps_count', 'sacc', 'me', 'uncert', 'qual', 'conf'])



