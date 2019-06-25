#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@author: Fraser King
   @date: 2019

   @desc:
   Script used for parsing the raw 
   ECCC csv data into an easier to use
   format in our comparison with CloudSat
"""

##################################################
########## IMPORTS

import sys,os,csv,glob
import pandas as pd 

##################################################
########## GLOBALS

mask_missing = False
delete_flag_cols = True

# Lookup table def (values based on ECCC climatic descriptions)
dict = {262: ('Precip @00-60 (mm)', 0.1),
		263: ('Precip @00-15 (mm)', 0.1),
		264: ('Precip @15-30 (mm)', 0.1),
		265: ('Precip @30-45 (mm)', 0.1),
		266: ('Precip @45-60 (mm)', 0.1),
		267: ('Precip weight @15 (kg/m2)', 0.1),
		268: ('Precip weight @30 (kg/m2)', 0.1),
		269: ('Precip weight @45 (kg/m2)', 0.1),
		270: ('Precip weight @60 (kg/m2)', 0.1),
		271: ('Wind speed @00-15 (kg/h)', 0.1),
		272: ('Wind speed @15-30 (kg/h)', 0.1),
		273: ('Wind speed @30-45 (kg/h)', 0.1),
		274: ('Wind speed @45-60 (kg/h)', 0.1),
		275: ('Snow depth @60 (cm)', 1),
		276: ('Snow depth @15 (cm)', 1),
		277: ('Snow depth @30 (cm)', 1),
		278: ('Snow depth @45 (cm)', 1),
		279: ('Wind dir @50-60 (degrees)', 1),
		280: ('Wind speed @50-60 (km/h)', 0.1),
		1: ('Daily Max Temp (Celcius)', 0.1),
		2: ('Daily Min Temp (Celcius)', 0.1),
		3: ('Daily Mean Temp (Celcius)', 0.1),
		4: ('Daily Max Humid (%)', 1),
		5: ('Daily Min Humid (%)', 1),
		6: ('Hourly Precip. 1200 UTC (mm)', 0.1),
		7: ('Hourly Precip. 1800 UTC (mm)', 0.1),
		8: ('Hourly Precip. 0000 UTC (mm)', 0.1),
		9: ('Hourly Precip. 600 UTC (mm)', 0.1),
		10: ('Total Rain (mm)', 0.1),
		11: ('Total Snow (cm)', 0.1),
		12: ('Total Precip (mm)', 0.1),
		13: ('Ground Snow (cm)', 1),
		14: ('Thunderstorms (Y/N)', 1),
		15: ('Freezing Rain (Y/N)', 1),
		16: ('Hail (Y/N)', 1),
		17: ('Fog (Y/N)', 1),
		18: ('Smoke/Haze (Y/N)', 1),
		19: ('Dust/Sand (Y/N)', 1),
		20: ('Blowing Snow (Y/N)', 1),
		21: ('Wind speed > 28 knots (Y/N)', 1),
		22: ('Wind speed > 34 knots (Y/N)', 1),
		23: ('Gust Dir 1976', 1),
		24: ('Gust Speed (km/h)', 1),
		25: ('Gust Hour (UTC)', 1),
		157: ('Gust Dir 1977', 1),
		179: ('Daily bright sun (hrs)', 0.1)}

# These are the CLIM_VARs that we will keep when we reduce the csv
desirable_climate_vars = {10} # Candidates: {262, 263, 264, 265, 266, 267, 268, 269, 270, 6, 7, 8, 9, 10, 11, 12}

##################################################
########## ARGS (INPUT) - filename

if len(sys.argv) < 2:
	print "ERROR: Please include a file (no extension) to clean as an argument to this script"
	sys.exit()

# Read in the file
filename = sys.argv[1]
dir = os.path.dirname(__file__)
file = filename
df = pd.read_csv(file + '.csv', index_col = False)
df.sort_values(by=['YYYY', 'MM'], inplace = True)
df.to_csv(filename + '_CLEAN.csv', index = False)
df = pd.read_csv(file + '_CLEAN.csv', index_col = False)

##################################################
########## HELPER FUNCTIONS

def insert_desc_col():
	"""Inserts a column into the csv to display the readable CLIM_VAR translations"""     
	print "Creating DESC column..."  
	df.insert(df.columns.get_loc("CLIM_VAR")+1, 'DESC', len(df.axes[0]))

def fill_descriptions():
	"""Fill the DESC column with the translations"""     
	print "Matching DESC column with CLIM_VAR translations..."  
	count = 0
	for row in df.itertuples(index=False, name='Pandas'):
	    key = getattr(row, "CLIM_VAR")
	    if key in dict:
	    	df.loc[count, "DESC"] = dict[key][0]
	    count += 1

def clean_rows():
	"""Move through each row in the df and remove all 
	rows that are composed of empty or missing data"""     
	print "Removing unnecessary rows from csv..."  
	count = 0
	starting_loc = df.columns.get_loc("DESC")+2
	ending_loc = len(df.columns)

	for row in df.itertuples(index=False, name='Pandas'):
		remove = True
		for i in range(starting_loc, ending_loc):
			if not(df.loc[count, "CLIM_VAR"] in desirable_climate_vars):
				break
			elif row[i] != ' ' and row[i] != "M":
				remove = False				 

		if remove == True:
			df.drop(count, inplace=True)
		count += 1

def convert_values():
	"""Convert each data value in the csv to use
	the correct units as defined in the documentation.""" 
	print "Converting data to readable values..."  
	count = 0
	starting_loc = df.columns.get_loc("DESC")+2
	ending_loc = len(df.columns)

	total_missing = 0

	for row in df.itertuples(index=False, name='Pandas'):
		key = getattr(row, "CLIM_VAR")

		x = starting_loc 
		while x <= ending_loc:
			if mask_missing and row[x-1] <= -9999:
				df.loc[count, df.columns[x-1]] = ""
				total_missing += 1
			else:
				if not(isinstance(row[x-1], str)):
					df.loc[count, df.columns[x-1]] = float(row[x-1]) * dict[key][1]
			x += 1
		count += 1
	print("Total Missing", total_missing)

def delete_cols():
	"""Removing potentially unwanted flag columns.""" 
	print "Removing unwanted columns..."  
	starting_loc = df.columns.get_loc("DESC")+2
	ending_loc = len(df.columns)
	cols_to_remove = []

	x = starting_loc 
	while x < ending_loc:
		cols_to_remove.append(x)
		x += 2

	print(cols_to_remove)
	df.drop(df.columns[cols_to_remove], axis=1, inplace = True)

##################################################
########## MAIN 

insert_desc_col()
fill_descriptions()
clean_rows()

if delete_flag_cols:
	delete_cols()

df.to_csv(filename + '_CLEAN.csv', index = False)
df = pd.read_csv(file + '_CLEAN.csv', index_col = False)
convert_values()

print "Finishing up..."  
df.to_csv(filename + '_CLEAN.csv', index = True)
print "Cleanup complete!"  





























