#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@author: Fraser King
   @date: 2019

   @desc:
   Code to read the raw HDF values from CloudSat
   around each station and extract what is required
   into individual csv files. Used in initial extraction
   and calibration of CloudSat data around stations.
"""

##################################################
########## IMPORTS

import sys,os,csv,glob
import numpy as np
import pandas as pd
import h5py
import math

##################################################
########## ARGS (INPUT) - inputpath, filename, outputpath

if len(sys.argv) < 3:
    print ("ERROR: Please include an inpath, filename & outpath in this program call as arguments")
    sys.exit()

##################################################
########## HELPER FUNCTIONS

def haversine_np(lon1, lat1, lon2, lat2):
    """Using the haversine method, determines the distance
       between CloudSat and and ground at a point"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
    
def get_box_points(lat, lon, offset):
    """Return a grid box around a point
    with a specifized size."""
    lat_offset  = (offset / 6371) * (180 / math.pi)
    lon_offset = (offset / 6371) * (180 / math.pi) / math.cos(lat * math.pi/180)
    top_right = (lat + lat_offset, lon + lon_offset)
    bottom_right = (lat - lat_offset, lon + lon_offset)
    top_left = (lat + lat_offset, lon - lon_offset)
    bottom_left = (lat - lat_offset, lon - lon_offset)
    return (top_right, bottom_right, bottom_left, top_left)

##################################################
########## GLOBALS

# Configuration variables (update as necessary)
grid_size = 100
use_box = False
station_name = "cambridge" 
station_lat = 69.10805556
station_lon = -105.13833333

# Get box info
if use_box:
    print("Grid box: ", get_box_points(station_lat, station_lon, grid_size))

# Open the file and read contents
inpath = sys.argv[1]
filename = sys.argv[2]
outpath = sys.argv[3]

with h5py.File(inpath,'r') as hdf:
    year = str(filename)[0:4]
    day_of_year = str(filename)[4:7]

    # Read in HDF data
    ls = list(hdf.keys()) 
    base_layer = list(hdf.items()) 

    # Open CloudSat Layer
    group1 = hdf.get('2B-CLDCLASS')
    group1_items = list(group1.items())

    # Geolocation Fields
    geolocation_fields = group1.get('Geolocation Fields')
    geolocation_fields_items = list(geolocation_fields.items())

    longitude = geolocation_fields.get('Longitude')
    longitude = np.array(longitude)
    longitude = longitude.astype(np.float64)
    
    height = geolocation_fields.get('Height')
    height = np.array(height)
    height = height.astype(np.float64)

    latitude = geolocation_fields.get('Latitude')
    latitude = np.array(latitude)
    latitude = latitude.astype(np.float64)

    utc_time = geolocation_fields.get('UTC_start')
    utc_time = np.array(utc_time)

    profile_time = geolocation_fields.get('Profile_time')
    profile_time = np.array(profile_time) 

    # Data Fields
    data_fields = group1.get('Data Fields')
    data_fields_items = list(data_fields.items())
    
    cloud_scenario = data_fields.get('cloud_scenario')
    cloud_scenario = np.array(cloud_scenario)
    cloud_scenario = cloud_scenario.astype(np.float64)

    data_status = data_fields.get('Data_status')
    data_status = np.array(data_status)
    data_status = data_status.astype(np.float64)

    cloud_scenario = list(cloud_scenario)
    latitude = list(latitude)
    longitude = list(longitude)
    height = list(height)
    profile_time = [i for sub in profile_time for i in sub]
    utc_time = [i for sub in utc_time for i in sub]
    new_list = [x+utc_time for x in profile_time]
    utc_time = [i for sub in new_list for i in sub]

    df_station = pd.DataFrame({"Year" : year,
                              "Day_of_Year" : day_of_year,
                              "Cloudsat_Lat" : latitude,
                              "Cloudsat_Lon" : longitude,
                              "UTC_Time" : utc_time,
                              "Height" : height,
                              "Data_status" : data_status,
                              "cloud_scenario" : cloud_scenario})
                              
    rect = get_box_points(station_lat, station_lon, grid_size)

    df_station['Ground_Lat'] = station_lat
    df_station['Ground_Lon'] = station_lon

    df_station5 = []

    if not(use_box):
        df_station['Distance'] = haversine_np(df_station['Cloudsat_Lon'], df_station['Cloudsat_Lat'], df_station['Ground_Lon'], df_station['Ground_Lat'])
        df_station5 = df_station[df_station.Distance < grid_size]
    else:
        df_station2 = df_station[df_station.Cloudsat_Lat < rect[3][0]]
        df_station3 = df_station2[df_station2.Cloudsat_Lat > rect[2][0]]
        df_station4 = df_station3[df_station3.Cloudsat_Lon > rect[3][1]]
        df_station5 = df_station4[df_station4.Cloudsat_Lon < rect[0][1]]

    # Add file name to keep in track of date
    df_station5['File_Name'] = filename
    df_station5 = df_station5[["Year", "Day_of_Year", "Cloudsat_Lat", "Cloudsat_Lon", "UTC_Time", "Height", "Data_status", "cloud_scenario", "File_Name"]]

    # Check for empty result
    if not df_station5.empty:
        print ("Overpass hit")
        df_station5.to_csv((outpath + "/" + station_name + "/" + filename + ".csv"), index=True)



