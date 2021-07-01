# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:36:09 2021

@author: giamm
"""

import numpy as np
import pandas as pd

from optimisation_quicky import power_flows_optimisation
from systems_specifications import storage_system_specifications

'''
The module contains a method that evaluates the monthly and yearly energy 
performances of a configuration where renewable electricity is shared among
an aggregate of users. The performances are evaluated for a configuration with
given sizes of the components (photovoltaic system and battery).
'''

def configuration_evaluation(
		time_dict, profiles_months, technologies_dict, auxiliary_dict):
	'''
	The method evaluate the monthly/yearly performances of a configuration
	with fixed components' sizes, where energy is virtually shared.
	'''

	#### Inputs

	## Time discretization
	# Time-step (h)
	dt = time_dict['dt']
	
	## Auxiliary variable (seasons/month/days/days_distr dictionaries)

	# Months dict (for each month, id number, nickname and season)
	months = auxiliary_dict['months']
	n_months = len(months)

	# Days dict (for each day-type, id number and nickname)
	day_types = auxiliary_dict['day_types']
	n_day_types = len(day_types)
	# Days distribution as array
	days_distr_months_array = auxiliary_dict['days_distr_months_array']
	
	### Systems sizes and specifications 

	## Photovoltaic system (pv)
	pv = technologies_dict['pv']
	# Size (kW, electric)
	pv_size = pv['size']
	
	## Battery energy storage system (bess)
	bess_flag = 'bess' in technologies_dict
	
	### Users' thermal/electric loads and production from the pv system
	
	## Photovoltaic system's production
	# Unit production (kWh/h/kWp) evaluated from PVGIS
	pv_production_unit_months = profiles_months['pv_production_unit_months']
	# Actual production (kWh/h)
	pv_production_months = pv_production_unit_months * pv_size
	
	## Users' electricity demand
	# Electric load (kWh/h) evaluated using the routine
	ue_demand_months = profiles_months['ue_demand_months']
	
	## Monthly values of energy 
	pv_energy_months = \
		(np.sum(pv_production_months, axis=0) * dt) * days_distr_months_array
	pv_energy_months = np.sum(pv_energy_months, axis=1)
	ue_energy_months = \
		(np.sum(ue_demand_months, axis=0) * dt) * days_distr_months_array
	ue_energy_months = np.sum(ue_energy_months, axis=1)

	monthly_energy = pd.DataFrame({
		'month': list(months.keys()),
		'pv_production': pv_energy_months,
		'ue_demand': ue_energy_months,
		})
	 
	### Evaluation
	
	shared_energy_months = np.zeros((n_months,))
	
	# (comment this if you don't care about bess's efficiencies)
	injections_months = np.zeros((n_months,))
	withdrawals_months = np.zeros((n_months,))
	
	## No optimisation of the power flows
	# If there is no battery, no optimisation is needed, therefore the 
	# evaluation of the shared energy can be done using numpy
	if not bess_flag:
		shared_power_months = \
			np.minimum(pv_production_months, ue_demand_months)
		
		shared_energy_months = (np.sum(shared_power_months, axis=0) * dt) * \
			days_distr_months_array
		shared_energy_months = np.sum(shared_energy_months, axis=1)
		
		monthly_energy['shared_power'] = shared_energy_months

		# (comment this if you don't care about bess's efficiencies)
		injections_months = pv_energy_months
		monthly_energy['injections'] = injections_months
		withdrawals_months = ue_energy_months
		monthly_energy['withdrawals'] = withdrawals_months


	### Optimisation of the power flows
	# Otherwise, the optimisation is performed         
	else:
		
		# If the optimisation fails in any typical day, a flag is activated
		# in correspondance of that day in order to fix the values using
		# adjacent days or months
		failed_opt_flags = np.zeros((n_months, n_day_types))
		
		for month in months:
			mm = months[month]['id']
			
			for day_type in day_types:
				dd = day_types[day_type]['id']   
				
				# Input profiles for the optimisation
				ue_demand = ue_demand_months[:, mm, dd]
				pv_production = pv_production_months[:, mm, dd]
				
				profiles = pd.DataFrame({
					'ue_demand': ue_demand,
					'pv_production': pv_production,
					})
				
				# Optimised power flows
				opt_status, results = power_flows_optimisation(
					time_dict, profiles, technologies_dict)
				
				# If the optimisation fails, the procedure continues to the
				# following typical day and the values are fixed later on
				if opt_status == 'failed':
					failed_opt_flags[mm, dd] = 1
					continue

				# Evaluation of the energy values
				n_days = days_distr_months_array[mm, dd]
				
				shared_power = results['shared_power']
				shared_energy_month = np.sum(shared_power)*dt * n_days
				shared_energy_months[mm] += shared_energy_month
				
				# (comment this if you don't care about bess's efficiencies)
				injections = results['injections']
				injections_month = np.sum(injections)*dt * n_days
				injections_months[mm] += injections_month
				
				# (comment this if you don't care about bess's efficiencies)
				withdrawals = results['withdrawals']
				withdrawals_month = np.sum(withdrawals)*dt * n_days
				withdrawals_months[mm] += withdrawals_month
				
		## Fixing failed optimisations
		# If any failed_opt flag is active, the values of energy are fixed
		if (failed_opt_flags != 0).any():
			indices = np.nonzero(failed_opt_flags)
			for mm, dd in zip(indices[0], indices[1]):

				# If the optimisation failed in both typical days of a month,
				# the values are going to be fixed later on, interpolating
				# between adjacent months
				if np.sum(failed_opt_flags[mm]) > 1:
					
					shared_energy_months[mm] = np.nan
					injections_months[mm] = np.nan
					withdrawals_months[mm] = np.nan
				
				# Otherwise, the values of the adjacent typical days (same
				# month, different day-type) are used
				else:  
					# ID of the adjacent day-type (this works only if two
					# day-types are considered)
					dd_adj = int(1 - dd)

					# THe number of days in both this day and the adjacent one
					# are needed to fix the energy values
					n_days = days_distr_months_array[mm, dd]
					n_days_adj = days_distr_months_array[mm, dd_adj]
					
					# The monthly energy accounts only for the previous 
					# typical day, therefore the value can be divided by the 
					# number of days in the adjacent typical day and then 
					# multiplied by the number of days of this day and added
					shared_energy_months[mm] += \
						shared_energy_months[mm] / n_days_adj * n_days
					injections_months[mm] += \
						injections_months[mm] / n_days_adj * n_days 
					withdrawals_months[mm] += \
						withdrawals_months[mm] / n_days_adj * n_days
					
					# The flag is deactivated
					failed_opt_flags[mm, dd] = 0
			
			# If the optimisation failed in both day types of a month, energy
			# values are interpolated using adjacent months whose energy is 
			# not nan
			if (np.isnan(shared_energy_months)).any():

				nans = np.isnan(shared_energy_months)
				fnan = lambda z: z.nonzero()[0]
				shared_energy_months[nans] = \
					np.interp(fnan(nans), fnan(~nans), 
							  shared_energy_months[~nans], period = n_months)
				
				nans = np.isnan(injections_months)
				fnan = lambda z: z.nonzero()[0]
				injections_months[nans] = \
					np.interp(fnan(nans), fnan(~nans), 
							  injections_months[~nans], period = n_months)
					
				nans = np.isnan(withdrawals_months)
				fnan = lambda z: z.nonzero()[0]
				withdrawals_months[nans] = \
					np.interp(fnan(nans), fnan(~nans), 
							  withdrawals_months[~nans], period = n_months)
				
				failed_opt_flags[nans, :] = [0, 0]
					
		monthly_energy['shared_power'] = shared_energy_months 
			 
		# (comment this if you don't care about bess's efficiencies)
		monthly_energy['injections'] = injections_months
		monthly_energy['withdrawals'] = withdrawals_months

	# Uncomment to write results	
	#monthly_energy.to_excel('monthly_energy.xlsx')
	
	yearly_energy = {
		'pv_production': np.sum(pv_energy_months),
		'ue_demand': np.sum(ue_energy_months),
		'shared power': np.sum(shared_energy_months),
		'injections': np.sum(injections_months),
		'withdrawals': np.sum(withdrawals_months),
		}
	
	return yearly_energy
	

	
	

