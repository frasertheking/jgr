#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@author: Fraser King
   @date: 2019

   @desc:
   A collection of primarily data processes scripts
   for organizing and extracting CloudSat data onto
   a grid. 
"""

##################################################
########## IMPORTS

import glob
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import sys, os, datetime
import math
import calendar
from netCDF4 import Dataset
from datetime import timedelta
from datetime import datetime
from class_def import *

##################################################
########## FLAGS (SET WHAT YOU WANT TO RUN)

run_combiner = False
run_extractor = False
run_extractor_one_degree = False
generate_netCDF = False
generate_netCDF_one_degree = False
gen_blended_accums = True
run_climatology_extractor = False
gen_delta_swe = False

##################################################
########## MAIN EXTRACTION

if run_combiner:
    """This cobines all individual CloudSat grid cells
    over all years into a master grid cell. Organized this way
    it is easy for us to analyze the data later on in swe_main.py
    """
    
    # Range of cells to concat
    lat_range = (60.25, 90.25)
    lon_range = (-179.75, 180.25)

    lats = np.arange(lat_range[0], lat_range[1], 0.5)
    lons = np.arange(lon_range[0], lon_range[1], 0.5)

    print("Sorting through lats/lons...")
    count = 1
    for lat in lats:
        for lon in lons:

            # Find files across years
            files = []
            for filename in glob.iglob('../csa/CloudSat/Gridding/data/**/' + str(lat) + '_' + str(lon) + '_cloudsat_cell.csv', recursive=True):
                files.append(filename)
            frame = pd.DataFrame()
            
            # Combine files into single file
            if len(files) > 1:
                list_ = []
                for file_ in files:
                    df = pd.read_csv(file_, index_col=None, header=None)
                    list_.append(df)
                frame = pd.concat(list_)
            elif len(files) == 1:
                frame = pd.read_csv(files[0], index_col=None, header=None)

            # Save combined output to combined master cell csv
            if not(frame.empty):
                frame.to_csv('../csa/CloudSat/Gridding/master/' + str(lat) + '_' + str(lon) + '_cell.csv', index=None, header=["lat", "lon", "rate", "rate_uncert", "utc_start", "profile_time", "day", "month", "year"])
            count += 1


if run_extractor:    
    """Extract CloudSat snowfall accumulation estimates at each 
    of the master cells generated in run_combiner
    """
    
    # Range of cells to concat
    lat_range = (60.25, 82.25)
    lon_range = (-179.75, 180.25)

    lats = np.arange(lat_range[0], lat_range[1], 0.5)
    lons = np.arange(lon_range[0], lon_range[1], 0.5)
    months_from_jan_2007 = list(range(108))
    cols = ['lat', 'lon', 'month', 'ovrps_count', 'sacc', 'me', 'uncert', 'qual', 'conf']
    master_df = pd.DataFrame(columns=cols)
    days_in_period = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    print("Sorting through lats/lons...")
    count = 1 # Used for printing progress
    for lat in lats:
        for lon in lons:
            lat=63.75
            lon=-68.75
            print("Working on: (" + str(lat) + ", " + str(lon) + ") : " + str(count) + " / " + str(31900))
            filepath = '/Users/fraserking/Desktop/' + str(lat) + '_' + str(lon) + '_cell.csv'
            count += 1
            
            if os.path.isfile(filepath):
                df = pd.read_csv(filepath)
                months = [[] for i in range(108)]
                for index, row in df.iterrows():
                    if row['year'] == 2016 or row['year'] == 2006 or row['rate'] <= -999:
                        continue
                    months[(int(row['month']) + (12*(int(row['year'])-2007)))-1].append((row['utc_start'], row['rate'], row['rate_uncert'], row['confidence'], row['quality']))

                month_count = 0
                for month in months:
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

                    # Missing vals
                    monthly_accumulation = -9999
                    monthly_uncertainty = -9999
                    monthly_h = -9999
                    monthly_confidence = -9999
                    monthly_quality = -9999
                    hours_in_month = days_in_period[(month_count % 12)] * 24

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

                    master_df = master_df.append({'lat':lat,
                                                  'lon':lon,
                                                  'month':month_count,
                                                  'ovrps_count':len(overpass_median_rates),
                                                  'sacc':monthly_accumulation,
                                                  'me':monthly_h,
                                                  'uncert':monthly_uncertainty,
                                                  'qual':monthly_quality,
                                                  'conf':monthly_confidence}, ignore_index=True)
                    month_count += 1
            else: # missing
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

    master_df.to_csv('/Users/fraserking/Desktop/cell_master.csv', index=None, header=['lat', 'lon', 'month', 'ovrps_count', 'sacc', 'me', 'uncert', 'qual', 'conf'])


##################################################
########## MAIN NETCDF GENERATION

if generate_netCDF:    
    """Package the newly generated CloudSat grid
    into netCDF format for further analysis
    """
    
    print("Creating .nc file...")
    
    # Opening file
    root_grp = Dataset('cloudsat_grid_cdf_v2.nc', 'w', format='NETCDF3_CLASSIC')
    root_grp.description = 'Monthly Average Accumulation (CloudSat)'
    
    # Setup ranges
    lat_range = (60.25, 90.25)
    lon_range = (-179.75, 180.25)
    time_range = (0, 108)

    lats = np.arange(lat_range[0], lat_range[1], 0.5)
    lons = np.arange(lon_range[0], lon_range[1], 0.5)
    times = np.arange(time_range[0], time_range[1], 1)
    
    # dimensions
    root_grp.createDimension('time', len(times))
    root_grp.createDimension('lat', len(lats))
    root_grp.createDimension('lon', len(lons))

    # variables
    time = root_grp.createVariable('time', 'f8', ('time'))
    lat = root_grp.createVariable('lat', 'f4', ('lat'))
    lon = root_grp.createVariable('lon', 'f4', ('lon'))
    sacc = root_grp.createVariable('sacc', 'f8', ('time', 'lat', 'lon'))
    qual = root_grp.createVariable('qual', 'f8', ('time', 'lat', 'lon'))
    uncert = root_grp.createVariable('uncert', 'f8', ('time', 'lat', 'lon'))
    conf = root_grp.createVariable('conf', 'f8', ('time', 'lat', 'lon'))
    counter = root_grp.createVariable('count', 'f8', ('time', 'lat', 'lon'))

    time.setncatts({'units': u"months since January 2007"})
    lat.setncatts({'long_name': u"latitude", 'units': u"degrees north"})
    lon.setncatts({'long_name': u"longitude", 'units': u"degrees east"})
    sacc.setncatts({'long_name': u"snow accumulation", 'units': u"mm"})
    qual.setncatts({'long_name': u"data quality", 'units': u"quality units"})
    uncert.setncatts({'long_name': u"snow accumulation uncertainity", 'units': u"mm"})
    conf.setncatts({'long_name': u"snow accumulation confidence", 'units': u"confidence units"})
    counter.setncatts({'long_name': u"number of overpasses"})

    # data
    lat[:] = lats
    lon[:] = lons
    time[:] = times

    for i in range(108):
        print()
        print("Month", i)
        lat_count = 0
        for lat in lats:
            print("Lat:", lat)
            lon_count = 0
            for lon in lons:
                filepath = 'data/grid/half_degree/' + str(lat) + '_' + str(lon) + '_stats.csv'
                if os.path.isfile(filepath):
                    df = pd.read_csv(filepath)
                    sacc[i,lat_count,lon_count] = df.sacc[i]
                    qual[i,lat_count,lon_count] = df.qual[i]
                    uncert[i,lat_count,lon_count] = df.uncert[i]
                    conf[i,lat_count,lon_count] = df.conf[i]
                    counter[i,lat_count,lon_count] = df.ovrps_count[i]
                else: #missing
                    sacc[i,lat_count,lon_count] = -9999
                    qual[i,lat_count,lon_count] = -9999
                    uncert[i,lat_count,lon_count] = -9999
                    conf[i,lat_count,lon_count] = -9999
                    counter[i,lat_count,lon_count] = -9999
                lon_count += 1
            lat_count += 1
        
    # File closing
    root_grp.close()
    print("File created!")


##################################################
########## BLENDED-4 ACCUMULATION CALCULATIONS

if gen_blended_accums:
    """Ingests daily SWE on ground values
    provided in the Blended-4 dataset and produces
    accumulated SWE estimates.
    """
    
    # Local helper functions
    def count_leaps(year_start, year_end):
        leap_years = 0
        for i in range (year_start, year_end):
            if calendar.isleap(i):
                leap_years += 1
        return leap_years

    def find_month_by_name(name, group):
        count = 0
        for month in group:
            if month.name == name:
                return count
            count += 1

    # Local variables
    days_in_period = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    blended_climatology = []

    # Initialization
    for i in range(0, 12):
        blended_climatology.append(BlendedObject(month_names[i], days_in_period[i]))

    base_date = datetime.strptime('1981-1-1', '%Y-%m-%d')
    swe_values = []

    # Begin loading .nc files
    for filename in os.listdir("data/SWE/blended4"):
        if filename.endswith(".nc"): 
            file = os.path.join("/Users/fraserking/desktop/temp", filename)
            print("Loading netCDF contents for ..." + filename)

            f = Dataset(file)

            # Extract netCDF variables
            day = f.variables['time'][:]
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            swe = f.variables['snw'][:]
            year_value = int(filename[-7:][:4])

            print("Extracting SWE values for superstations...")
            
            for i in range (0, day.shape[0]):
                swe_matrix = swe[i]
                day_value = int(day[i]) + count_leaps(1981, year_value)
                month_value = (base_date+timedelta(days=day_value)).month
                
                # Add to climatology
                clim_month = blended_climatology[month_value-1]
                clim_month.accumulations.append(swe_matrix.data)
                
    # Calculate Accums from SWE
    daily_accums = [[] for i in range(12)]
    first = True
    prev_grid = np.empty(shape=(180,720))
    for i, month in enumerate(blended_climatology):
        print("Calculating Daily Accums in Month:", i)
        for j,grid in enumerate(month.accumulations):
            if first or (j % days_in_period[i] == 0):
                prev_grid = grid
                first = False
                continue
            
            accum = np.subtract(grid, prev_grid)
            accum[accum < 0] = 0
            daily_accums[i].append(accum)
            prev_grid = grid
            
            
    monthly_accums = [[] for i in range(12)]
    for j, month in enumerate(daily_accums):
        print("Calculating Monthly Accums in Month:", j)
        monthly_accum = np.empty(shape=(180,720))
        first = True
        for grid in month:
            if first:
                monthly_accum = grid
                first = False
                continue
            monthly_accum = np.add(monthly_accum, grid)
        monthly_accums[j] = monthly_accum
                

##################################################
########## GENERATE NETCDF FROM BLENDED ACCUMS

if gen_blended_accums: 
    """Takes the accumulated swe values previously generated
    and saves them into netcdf format for further analysis.
    """

    # Opening file
    root_grp = Dataset('rossbrown_climatology.nc', 'w', format='NETCDF3_CLASSIC')
    root_grp.description = 'Ross Brown Accumulations'
    
    # Setup ranges
    lat_range = (0.25, 90.25)
    lon_range = (-179.75, 180.25)
    time_range = (0, 1)

    lats = np.arange(lat_range[0], lat_range[1], 0.5)
    lons = np.arange(lon_range[0], lon_range[1], 0.5)
    times = np.arange(time_range[0], time_range[1], 1)
    
    # dimensions
    root_grp.createDimension('time', len(times))
    root_grp.createDimension('lat', len(lats))
    root_grp.createDimension('lon', len(lons))

    # variables
    time = root_grp.createVariable('time', 'f8', ('time'))
    lat = root_grp.createVariable('lat', 'f4', ('lat'))
    lon = root_grp.createVariable('lon', 'f4', ('lon'))
    sacc = root_grp.createVariable('sacc', 'f8', ('time', 'lat', 'lon')) # sacc = snow accumulation

    time.setncatts({'units': u"Months"})
    lat.setncatts({'long_name': u"latitude", 'units': u"degrees north"})
    lon.setncatts({'long_name': u"longitude", 'units': u"degrees east"})
    sacc.setncatts({'long_name': u"snow accumulation", 'units': u"mm"})

    # data
    lat[:] = lats
    lon[:] = lons
    time[:] = times

    for i in range(1):
        print()
        print("Month", i)
        for j in range(180):
            for k in range(720):
                sacc[i,j,k] = ((monthly_accums[j][k] * 1000) / 9)
        
    # closing
    root_grp.close()
    print("File created!")

