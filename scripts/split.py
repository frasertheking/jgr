#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@author: Fraser King
   @date: 2019

   @desc:
   Script that breaks up one large
   csv with all ECCC station data into
   more managable files.
"""

##################################################
########## IMPORTS

import sys,os,csv,glob
import pandas as pd 

##################################################
########## ARGS (INPUT) - filename, ("DLY" ir "HYL")

if len(sys.argv) < 3:
	print "ERROR: Please include a file (no extension) to clean as an argument to this script followed either DLY type or HLY type"
	sys.exit()

# Read in the file
filename = sys.argv[1]
data_type = sys.argv[2]
dir = os.path.dirname(__file__)
file = filename
df = pd.read_csv(file, index_col = False)

def split_stations():
	"""Split all stations in the provided ECCC file
	into their separate stations"""  
	df.sort_values(by=['STATION_ID'], inplace=True)

	start_count = 0
	end_count = 2
	previous_id = -1
	
	# Split by unique ID
	for row in df.itertuples(index=False, name='Pandas'):
		current_id = getattr(row, "STATION_ID")
		if (previous_id != -1 and current_id != previous_id) or end_count == len(df)+1:
			df.iloc[start_count:end_count-2].to_csv(str(previous_id) + '_' + str(data_type) + '.csv', index=False, header=True)
			start_count = end_count + 1
		end_count += 1
		previous_id = current_id

split_stations()





























