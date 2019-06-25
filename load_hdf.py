#################################################################
# 	load_hdf.py - extract data from HDF4 files 					#
#	created by Alex Cabaj, partially adapted from 				#
# 	documentation at http://hdfeos.github.io/pyhdf/index.html	#
#################################################################

import numpy as np
from pyhdf.HDF import *
from pyhdf.VS import *
from pyhdf.SD import SD, SDC

def load_sd(fname, varnames):
	'''return list containing SD (Scientific Dataset) structures specified by 
	list varnames, contained in hdf4 file with filename fname'''
	dataf = SD(fname)
	data_list = []
	for name in varnames:
		data_list.append(dataf.select(name)[:])
	dataf.end()
	return data_list

def load_vd(fname, varnames):
	'''return list containing vdata structures specified by list varnames,
	contained in hdf4 file with filename fname'''
	f = HDF(fname)
	data_list = []
	vs = f.vstart()
	for name in varnames:
		vd = vs.attach(name)
		data_list.append(vd[:])
		vd.detach()
	vs.end()
	f.close()
	return data_list
