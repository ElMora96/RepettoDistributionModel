# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:36:09 2021

@author: giamm
"""

import numpy as np
import pandas as pd
import warnings

from pathlib import Path


'''
The module contains a method that evaluates the monthly and yearly energy 
performances of a configuration where renewable electricity is shared among
an aggregate of users. The performances are evaluated for a configuration with
given sizes of the components (photovoltaic system and battery).
'''

def yearly_series(time_dict, profile_months, auxiliary_dict):
	'''
	The method converts an array of daily power profiles (time-series) grouped 
	by months and day-types into a single hourly time-series for the whole year
	
	Parameters
	----------
	time_dict : dict
		Contains all the data needed for the time-discretisation of one day
	profile_months : np.array
		Contains the daily profiles grouped by typical days
	auxiliary_dict : dict
		Contains all the elements about the reference year.

	Returns
	-------
	profile_series : np.array
		Hourly year time-series associated to the input power profile.
	'''
	
	### Inputs
	
	## Time discretization
	# Time-step (h)
	time_sim = time_dict['time_sim']
	time_length = time_sim.size
	
	## Refence year (for each day, season month and day-type's IDs)
	reference_year = auxiliary_dict['reference_year']
	n_days = len(reference_year)
	n_timesteps = n_days * time_length
	
	### Evaluation
	
	profile_series = np.empty((n_timesteps,))
	
	for index, day in reference_year.iterrows():
		mm = day['Month']
		dd = day['Day-type']
		start = index * time_length
		stop = start + time_length
		
		profile_series[start : stop] = \
			profile_months[:, mm, dd]
	
	return profile_series
	
##############################################################################

def configuration_evaluation(
		profiles_months, technologies_dict):
	'''
	The method evaluate the monthly/yearly performances of a configuration
	with fixed components' sizes, where energy is virtually shared.
	
	Parameters
	----------
	profiles_months : dict of np.arrays
		Contains the arrays of the daily profiles (consumption and production).
	technologies_dict : dict
		Contains all the data about the components in the configuration (PV 
		and BESS)

	Returns
	-------
	series : dict
		Contains the arrays of production, consumption and shared energy.
	'''

	#### Inputs
	
	### Systems sizes and specifications 

	## Photovoltaic system (pv)
	pv = technologies_dict['pv']
	# Size (kW, electric)
	pv_size = pv['size']
	
	## Battery energy storage system (bess)
	bess_flag = 'bess' in technologies_dict
	if bess_flag:
		bess = technologies_dict['bess']
		bess_flag = bess['size']  > 0
	
	### Users' thermal/electric loads and production from the pv system
	
	## Photovoltaic system's production
	# Unit production (kWh/h/kWp) evaluated from PVGIS
	pv_production_unit = profiles_months['pv_production_unit']

	# Actual production (kWh/h)
	pv_production = pv_production_unit * pv_size
	
	## Users' electricity demand
	# Electric load (kWh/h) evaluated using the routine
	ue_demand = profiles_months['ue_demand']
	
	### Evaluation
	
	## No optimisation of the power flows
	# If there is no battery, no optimisation is needed, therefore the 
	# evaluation of the shared energy can be done using numpy
	if bess_flag:
		warnings.warn("Presence of the BESS ignored")
	
	# Shared energy as the minimum between production and consumption
	shared = np.minimum(pv_production,ue_demand)
	series = {
		'pv_production': pv_production,
		'ue_demand': ue_demand,
		'shared': shared,
		}
	
	return series

##############################################################################

# =============================================================================
# ## Uncomment to test
# 
# import os
# from storage_system_specifications import storage_system_specifications
# 
# # Auxiliary variables
# # Seasons, storing for each season an id number and nickname
# seasons = {
# 	'winter': {'id': 0, 'nickname': 'w'},
# 	'spring': {'id': 1, 'nickname': 'ap'},
# 	'summer': {'id': 2, 'nickname': 's'},
# 	'autumn': {'id': 3, 'nickname': 'ap'},
# 	}
# # months, storing for each month an id number and nickname				
# months = {
# 	'january': {'id': 0, 'nickname': 'jan', 'season': 'winter'},
# 	'february': {'id': 1, 'nickname': 'feb', 'season': 'winter'},
# 	'march': {'id': 2, 'nickname': 'mar', 'season': 'winter'},
# 	'april': {'id': 3, 'nickname': 'apr', 'season': 'spring'},
# 	'may': {'id': 4, 'nickname': 'may', 'season': 'spring'},
# 	'june': {'id': 5, 'nickname': 'jun', 'season': 'spring'},
# 	'july': {'id': 6, 'nickname': 'jul', 'season': 'summer'},
# 	'august': {'id': 7, 'nickname': 'aug', 'season': 'summer'},
# 	'september': {'id': 8, 'nickname': 'sep', 'season': 'summer'},
# 	'october': {'id': 9, 'nickname': 'oct', 'season': 'autumn'},
# 	'november': {'id': 10, 'nickname': 'nov', 'season': 'autumn'},
# 	'december': {'id': 11, 'nickname': 'dec', 'season': 'autumn'},
# 	}
# # Day types, storing for each day type an id number and nickname
# day_types = {
# 	'work-day': {'id': 0, 'nickname': 'wd'},
# 	'weekend-day': {'id': 1, 'nickname': 'we'},
# 	}
# # Distribution of both day types among all months
# days_distr_months = {
# 	'january': {'work-day': 21, 'weekend-day': 10},
# 	'february': {'work-day': 20, 'weekend-day': 8},
# 	'march': {'work-day': 23, 'weekend-day': 8},
# 	'april': {'work-day': 22, 'weekend-day': 8},
# 	'may': {'work-day': 21, 'weekend-day': 10},
# 	'june': {'work-day': 22, 'weekend-day': 8},
# 	'july': {'work-day': 22, 'weekend-day': 9},
# 	'august': {'work-day': 22, 'weekend-day': 9},
# 	'september': {'work-day': 22, 'weekend-day': 8},
# 	'october': {'work-day': 21, 'weekend-day': 10},
# 	'november': {'work-day': 22, 'weekend-day': 8},
# 	'december': {'work-day': 23, 'weekend-day': 8},
# 	}
# 
# n_months = len(months)
# n_day_types = len(day_types)
# 
# # Distribution of both day types among all months
# days_distr_months = {
# 	'january': {'work-day': 21, 'weekend-day': 10},
# 	'february': {'work-day': 20, 'weekend-day': 8},
# 	'march': {'work-day': 23, 'weekend-day': 8},
# 	'april': {'work-day': 22, 'weekend-day': 8},
# 	'may': {'work-day': 21, 'weekend-day': 10},
# 	'june': {'work-day': 22, 'weekend-day': 8},
# 	'july': {'work-day': 22, 'weekend-day': 9},
# 	'august': {'work-day': 22, 'weekend-day': 9},
# 	'september': {'work-day': 22, 'weekend-day': 8},
# 	'october': {'work-day': 21, 'weekend-day': 10},
# 	'november': {'work-day': 22, 'weekend-day': 8},
# 	'december': {'work-day': 23, 'weekend-day': 8},
# 	}
# 
# # Days distributions as arrays useful for quicker calculations
# days_distr_months_array = np.zeros((n_months, n_day_types))
# for month in months:
# 	mm = months[month]['id']
# 	for day_type in day_types:
# 		dd = day_types[day_type]['id']
# 		days_distr_months_array[mm, dd] = days_distr_months[month][day_type]
# 		
# ## Refence year (for each day, season month and day-type's IDs)
# basepath = Path(__file__).parent
# filename = 'year.csv'
# reference_year = pd.read_csv(basepath / filename, sep=';')
# 
# # Auxiliary time dictionary
# auxiliary_dict = {
# 	'seasons': seasons,
# 	'months': months,
# 	'day_types': day_types,
# 	'reference_year': reference_year,
# 	}
# 
# # Time discretization
# # Time-step (h)
# dt = 1
# # Total time of simulation (h)
# time = 24
# # Vector of time (h)
# time_sim = np.arange(0, time, dt)
# time_dict = {
# 	'dt': dt,
# 	'time': time,
# 	'time_sim': time_sim,
# 	}
# 
# n_days = len(reference_year)
# time_length = time_sim.size
# n_timesteps = n_days*time_length
# 
# # PV data
# basepath = Path(__file__).parent
# filename = 'pv_production_unit.csv'
# pv_data = np.array(pd.read_csv(basepath / filename, sep=';'))
# pv_production_unit_months = pv_data[:, 1:]
# # Broadcasting the array in order to account for different day types
# pv_production_unit_months = pv_production_unit_months[:, :, np.newaxis]
# broadcaster = np.zeros((n_day_types,))
# pv_production_unit_months = pv_production_unit_months + broadcaster
# 
# pv_production_unit_series = pv_production_unit_months.copy()
# # Check shape
# series_shape = pv_production_unit_series.shape
# if not series_shape == (n_timesteps,):
# 	if series_shape == (time_length, n_months, n_day_types):
# 		pv_production_unit_series = \
# 			yearly_series(time_dict, pv_production_unit_series, 
# 			  auxiliary_dict)
# 	else:
# 		raise ValueError('Wrong size of the production array')
# 
# # UE demand
# basepath= Path(__file__).parent
# datapath = basepath / 'Data'
# player_data_folders = os.listdir(datapath)
# player_list = [] #Store _Player objects
# ue_demand_months = np.empty((time_length, n_months, n_day_types))
# for ix, folder in enumerate(player_data_folders):
# 	#Workday
# 	wd_path = datapath / folder / "consumption_profiles_month_wd.csv"
# 	wd_data = pd.read_csv(wd_path,
# 						  sep = ';',
# 						  decimal= ',',
# 						  ).dropna().values[:,1:]
# 	ue_demand_months[:, :, 0] += wd_data
# 	
# 	#Weekend
# 	we_path = datapath / folder / "consumption_profiles_month_we.csv"
# 	we_data = pd.read_csv(we_path,
# 						  sep = ';',
# 						  decimal= ',',
# 						  ).dropna().values[:,1:]
# 	ue_demand_months[:, :, 1] += we_data
# 	
# 	print(np.max(ue_demand_months))
# 	
# ue_demand_series = ue_demand_months.copy()
# # Check shape
# series_shape = ue_demand_series.shape
# if not series_shape == (n_timesteps,):
# 	if series_shape == (time_length, n_months, n_day_types):
# 		ue_demand_series = \
# 			yearly_series(time_dict, ue_demand_series, 
# 			  auxiliary_dict)
# 	else:
# 		raise ValueError('Wrong size of the consumption array')
# 	
# profiles_months = {
# 	'pv_production_unit': pv_production_unit_series,
# 	'ue_demand': ue_demand_series}
# 
# # Battery specs (constant)
# # Maximum and minimum states of charge (SOC) (%)
# # Minimum time of charge/discharge (h)
# # Charge, discharge and self-discharge efficiencies (-)
# # Size is not present as it varies according to subconfiguration
# bess = storage_system_specifications(default_flag=False)['bess']
# bess['size'] = 0
# technologies_dict = {
# 	'bess': bess,
# 	'pv': {'size': 55},
# 	}
# 
# series = \
# 	configuration_evaluation(profiles_months, technologies_dict)
# 
# pv_production_months = \
# 	pv_production_unit_months * technologies_dict['pv']['size']
# pv_energy_months = \
# 	np.sum(pv_production_months, axis=0) *dt * \
# 	days_distr_months_array
# pv_energy_months = np.sum(pv_energy_months, axis=1)
# 
# ue_energy_months = \
# 	np.sum(ue_demand_months, axis=0) *dt * \
# 	days_distr_months_array
# ue_energy_months = np.sum(ue_energy_months, axis=1)
# 
# shared_months = \
# 	np.minimum(
# 		pv_production_months, ue_demand_months)
# 
# shared_energy_months = \
# 	np.sum(shared_months, axis=0) *dt * \
# 	days_distr_months_array
# shared_energy_months = np.sum(shared_energy_months, axis=1)
# 
# pv_energy_year = np.sum(pv_energy_months)
# ue_energy_year = np.sum(ue_energy_months)
# shared_energy_year = np.sum(shared_energy_months)
# 
# pv_production_series = np.array(series['pv_production'])
# ue_demand_series = np.array(series['ue_demand'])
# shared_series = np.array(series['shared'])
# 
# pv_energy_year2 = np.sum(pv_production_series) * dt
# ue_energy_year2 = np.sum(ue_demand_series) * dt
# shared_energy_year2 = np.sum(shared_series) * dt
# 
# pv_energy_months2 = np.zeros((n_months,))
# ue_energy_months2 = np.zeros((n_months,))
# shared_energy_months2 = np.zeros((n_months,))
# 
# start = 0
# for month in months:
# 	mm = months[month]['id']
# 	n_days = np.sum(days_distr_months_array[mm, :])
# 	stop = int(start + n_days*time_length)
# 	
# 	pv_energy_months2[mm] = \
# 		np.sum(pv_production_series[start : stop])*dt
# 		
# 	ue_energy_months2[mm] = \
# 		np.sum(ue_demand_series[start : stop])*dt
# 		
# 	shared_energy_months2[mm] = \
# 		np.sum(shared_series[start : stop])*dt
# 	
# 	start = stop
# 	
# print(42*pv_energy_year/1000 + 118.56*shared_energy_year/1000)
# =============================================================================
