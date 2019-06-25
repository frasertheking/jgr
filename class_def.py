#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@author: Fraser King
   @date: 2019

   @desc:
   Class declarations used in the processesing and 
   comparison of CloudSat data.
"""

import numpy as np

class BaseStation(object):
    station = 0
    lc_frac = 0
    grid_cell_x = 0
    grid_cell_y = 0
    bounds = 0
    ymax_clim = 0
    ymax_time = 0
    station_x = 0
    station_y = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, station, lc_frac, grid_cell_x, grid_cell_y, bounds, ymax_clim, ymax_time, station_x, station_y):
        self.station = station
        self.lc_frac = lc_frac
        self.grid_cell_x = grid_cell_x
        self.grid_cell_y = grid_cell_y
        self.bounds = bounds
        self.ymax_clim = ymax_clim
        self.ymax_time = ymax_time
        self.station_x = station_x
        self.station_y = station_y
        
class Period(object):
    name = 0
    scale = ""
    days_in_period = 0
    snowfall_events = [[]]
    event_count = 0
    mean_snowfall_rate = 0
    sd = 0
    se = 0
    CI_upper = 0
    CI_lower = 0
    accumulation = 0
    accumulation_upper = 0
    accumulation_lower = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, name, scale, days_in_period):
        self.name = name
        self.scale = scale
        self.snowfall_events = [[]]
        self.days_in_period = days_in_period
        self.event_count = 0
        self.mean_snowfall_rate = 0
        self.sd = 0
        self.se = 0
        self.CI_upper = 0
        self.CI_lower = 0
        self.accumulation = 0
        self.accumulation_upper = 0
        self.accumulation_lower = 0

class BlendedObject(object):
    name = ""
    blended_swe_values = [[]]
    blended_var_values = [[]]
    yearly_accumulations = [[]]
    accumulations = []
    stds = []
    days_in_period = 0
    event_count = 0
    total_accumulation = 0
    sd = 0
    se = 0
    CI_upper = 0
    CI_lower = 0
    accumulation_upper = 0
    accumulation_lower = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, name, days_in_period):
        self.name = name
        self.blended_swe_values = [[] for x in range(9)]
        self.blended_var_values = [[] for x in range(9)]
        self.yearly_accumulations = np.zeros(9)
        self.accumulations = []
        self.stds = []
        self.days_in_period = days_in_period
        self.event_count = 0
        self.total_accumulation = 0
        self.sd = 0
        self.se = 0
        self.CI_upper = 0
        self.CI_lower = 0
        self.accumulation_upper = 0
        self.accumulation_lower = 0