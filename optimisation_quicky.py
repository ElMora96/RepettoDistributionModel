# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:34:27 2021

@author: giamm
"""

import pulp as plp
import numpy as np
import pandas as pd
from tictoc import tic, toc
from pathlib import Path
import warnings
'''
The module contains a method that performs the optimisation of the power flows
in a configuration where renewable electricity is shared among an aggregate of
customers, also using a battery energy storage. The optimisation is performed 
during one day of simulation. 
The method receives the electric load and production profiles as inputs, and 
returns the optimised power flows.
'''

def power_flows_optimisation(
		time_dict, profiles, technologies_dict):
	'''
	The method performs the optimisation of the power flows (thermal and 
	electric), considering the coupling between a PV system, an heat pump 
	and a battery and thermal energy storage system.
	
	Inputs:
		- time_dict, dict, containing useful elements for the discretisation 
		of the one-day simulation time;
		- profiles, df, containing the input production and load 
		profiles from the users (electric);
		- technologies_dict, dict, containing the sizes and other 
		specifications about the technologies involved;
		
	Outputs:
		- results, df, containing the optimised powers;
		- optimisation_status, str, showing the status of the optimisation. 
	'''
	
	# tic()
	#### Inputs

	## Time discretization
	# Time-step (h)
	dt = time_dict['dt']
	# Total time of simulation (h)
	time = time_dict['time']
	# Vector of time (h)
	time_sim = time_dict['time_sim']
	time_length = time_sim.size
	
	### Systems sizes and specifications   

	## Photovoltaic system (pv)
	pv = technologies_dict['pv']
	# Size (kW, electric)
	pv_size = pv['size']
	
	## Electricicty distribution grid (grid)
	grid = technologies_dict['grid']
	# Maximum purchasable power (kW, electric)
	grid_purchase_max = grid['purchase_max']
	# Maximum feedable power (kW, electric)
	grid_feed_max = grid['feed_max']
	
	## Battery energy storage system (bess)
	# If the battery is not present, the optimisation is not needed
	easy_solve_flag = False
	bess_flag = 'bess' in technologies_dict
	if bess_flag:
		bess = technologies_dict['bess']
		# Size (kWh, electric)
		bess_size = bess['size']
		# Capacity (kWh, electric)
		bess_capacity = bess_size
		# Maximum and minimum states of charge (SOC) (%)
		bess_soc_max = bess['soc_max']
		bess_soc_min = bess['soc_min']
		# Minimum time of charge/discharge (h)
		bess_t_cd_min = bess['t_cd_min']
		# Charge, discharge and self-discharge efficiencies (-)
		bess_eta_charge = bess['eta_charge']
		bess_eta_discharge = bess['eta_discharge']
		bess_eta_self_discharge = bess['eta_self_discharge']
		# Maximum and minimum energy in the battery (kWh)
		bess_energy_max = bess_soc_max*bess_capacity 
		bess_energy_min = bess_soc_min*bess_capacity 
		# Maximum power of discharge and charge (kW)
		bess_discharge_max = (bess_energy_max - bess_energy_min)/bess_t_cd_min 
		bess_charge_max = bess_discharge_max

	else:
		easy_solve_flag = True
		
	### Users' thermal/electric loads and production from the pv system

	## Photovoltaic system's production
	# Unit production (kWh/h) evaluated from PVGIS
	pv_production = np.array(profiles['pv_production'])
	
	## Users' electricity demand
	# Electric load (kWh/h) evaluated using the routine for both work- and 
	# weekend-days
	ue_demand = np.array(profiles['ue_demand'])
	
	# If there is no excess production from the PV at all time-steps, the 
	# optimisation is not needed.
	# Defining a tolerance on the excess power (since it is given in kW,
	# a tolerance of 1e-4 is a tenth of a W)
	tol = 1e-4
	excess = np.maximum(pv_production - ue_demand, 0)
	if (excess < tol).all():
		easy_solve_flag = True
	 
	#### Easy solution
	# The easy solution does not require any optimisation
	if easy_solve_flag:
		opt_status = 'Unneeded'
		# grid_feed = excess
		# grid_purchase = np.maximum(ue_demand - pv_production, 0)
		# bess_charge = np.zeros((time_length,))
		# bess_discharge = np.zeros((time_length,))
		# bess_energy = np.zeros((time_length,))
		injections = pv_production
		withdrawals = ue_demand
		shared_power = np.minimum(pv_production, ue_demand)
		
		# # If the battery is present in the configuration, some energy is to be
		# # charged in it to account for self-discharge. The battery is kept to
		# # the minimum energy state.
		# if bess_flag:
		#     bess_energy = bess_energy_min * np.ones((time_length,))
		#     bess_charge = \
		#         (np.roll(bess_energy, -1) -\
		#          bess_energy*bess_eta_self_discharge) / (dt * bess_eta_charge)
		#     bess_discharge = np.zeros((time_length,))
		#     grid_purchase += bess_charge
			
		results = pd.DataFrame({
			# 'pv_production': pv_production,
			# 'ue_demand': ue_demand,
			# 'grid_purchase': grid_purchase,
			# 'grid_feed': grid_feed,
			# 'bess_charge': bess_charge,
			# 'bess_discharge': bess_discharge,
			# 'bess_energy': bess_energy,
			'injections': injections,
			'withdrawals': withdrawals,
			'shared_power': shared_power,
			 })  
	
		return opt_status, results

	#### Optimisation procedure

	### Definition of the problem
	
	## Initializing the optimisation problem using Pulp
	opt_problem = plp.LpProblem('Pulp', plp.LpMaximize)
	
	## Assigning the variables to each time-step
	# Powers are represented by LpContinuous variables, bounded to be positive,
	# the state of some components (grid, bess and tess charge/discharge) is 
	# represented through binary variables (states).
	
	# Grid
	time_set = range(time_length)
	grid_feed = \
		[plp.LpVariable('Pf_{}'.format(i), lowBound=0) for i in time_set]
	grid_purchase = \
		[plp.LpVariable('Pp_{}'.format(i), lowBound=0) for i in time_set]
	grid_feed_state = \
		[plp.LpVariable('Df_{}'.format(i), cat=plp.LpBinary) 
		 for i in time_set]
	grid_purchase_state = \
		[plp.LpVariable('Dp_{}'.format(i), cat=plp.LpBinary) 
		 for i in time_set]
	
	# Battery energy storage system 
	bess_charge = \
		[plp.LpVariable('Pbat_in_{}'.format(i), lowBound=0) for i in time_set]
	bess_discharge = \
		[plp.LpVariable('Pbat_out_{}'.format(i), lowBound=0)
		 for i in time_set]
	bess_energy = \
		[plp.LpVariable('Ebat_{}'.format(i), lowBound=0) for i in time_set]
	bess_charge_state = \
		[plp.LpVariable('Dbat_in_{}'.format(i), cat=plp.LpBinary)
		 for i in time_set]
	bess_discharge_state = \
		[plp.LpVariable('Dbat_out_{}'.format(i), cat=plp.LpBinary)
		 for i in time_set]
	
	# Shared power
	shared_power = \
		[plp.LpVariable("Psh_{}".format(i), lowBound=0) for i in time_set]
	y = \
		[plp.LpVariable("y_{}".format(i), cat=plp.LpBinary) for i in time_set]
	
	# TBN, y is a binary variable used to linearise the definition of shared 
	# power. The function min(x1, x2) is not linear, a proper implementation 
	# is needed. Variable y is used to assess which is the minimum between
	# x1 and x2 (inputs of min function).
	# M is a big-parameter used to linearise the function min(x1, x2), defined
	# as a value which is always larger than both x1 and x2.
	# In this case, x1 and x2 are, respectively, the hourly sum between the PV 
	# production and battery discharge minus charge, and the hourly sum 
	# between the users electricity demand and heat pump electric power. 
	injections_max = pv_size + bess_discharge_max
	withdrawals_max = grid_purchase_max   
	M = 10*max(injections_max, withdrawals_max)

	## Adding the constraints to each time-step
	# Constraints can represent energy conservation equations, consitutive 
	# equations of some components, upper bounds for some variables and 
	# mutual exclusivity between variables (using binary states)
	for i in time_set:
		
		# Electric node, energy conservation
		opt_problem += \
			(pv_production[i] + grid_purchase[i] + bess_discharge[i] \
			- grid_feed[i] - bess_charge[i] - ue_demand[i]) * dt == 0, \
			'Electric node energy conservation {}'.format(i)
		
		# Battery energy storage, energy conservation
		if (i < time_length - 1):
			opt_problem += bess_energy[i + 1] - \
				bess_eta_self_discharge*bess_energy[i] \
				+ (bess_discharge[i] * (1/bess_eta_discharge) \
				- bess_charge[i]*bess_eta_charge) * dt == 0, \
				'BESS energy conservation {}'.format(i)
		else:
			opt_problem += \
				bess_energy[0] - bess_eta_self_discharge*bess_energy[i] \
				+ (bess_discharge[i] * (1/bess_eta_discharge) \
				- bess_charge[i]*bess_eta_charge) * dt == 0, \
				'BESS energy conservation {}'.format(i)
	   
		# Grid feed and purchase, constraint on maximum power
		opt_problem += grid_feed[i] <= grid_feed_state[i]*grid_feed_max, \
			'Grid feed maximum power {}'.format(i)
		opt_problem += \
			grid_purchase[i] <= grid_purchase_state[i]*grid_purchase_max, \
			'Grid purchase maximum power {}'.format(i)
	
		# Grid feed and purchase, constraint on mutual exclusivity
		# opt_problem += grid_feed_state[i] + grid_purchase_state[i] >= 0
		opt_problem += grid_feed_state[i] + grid_purchase_state[i] <= 1, \
			'Grid feed and purchase mutual exclusivity {}'.format(i)
	
		# Bess charge and discharge, constraint on maximum power and 
		# on mutual exclusivity, constraint on minimum/maximum storable energy
		opt_problem += \
			bess_charge[i] <= bess_charge_state[i]*bess_charge_max, \
			'BESS charge maximum power {}'.format(i)
		opt_problem += bess_discharge[i] <= \
			bess_discharge_state[i]*bess_discharge_max, \
			'BESS discharge maximum power {}'.format(i)
		# opt_problem += bess_charge_state[i] + bess_discharge_state[i] >= 0
		opt_problem += \
			bess_charge_state[i] + bess_discharge_state[i] <= 1, \
			'BESS chrage and discharge mutual exclusivity {}'.format(i)
		opt_problem += bess_energy[i] <= bess_energy_max, \
			'BESS minimum energy {}'.format(i)
		opt_problem += bess_energy[i] >= bess_energy_min, \
			'BESS maximum energy {}'.format(i)
	
		# Grid feed, battery cannot be discharged to feed energy into the grid
		opt_problem += (grid_feed[i] <= pv_production[i]), \
			'Grid feed not using battery discharge {}'.format(i)
	
		# Grid purchase, battery cannot be charged purchasing from the grid
		opt_problem += (bess_charge[i] <= pv_production[i]), \
			'Battery charge not using grid purchase {}'.format(i)
		
		# Linearization of shared energy definition
		# shared_power = min(
		#       pv_production + bess_discharge - bess_charge, <--- injections
		#       ue_demand + hp_electric)                      <--- withdrawals
	
		# Constraint on the shared energy, that must be smaller than both the
		# injections in the grid (pv_production + bess_discharge - bess_charge)
		# and the withdrawals from the grid (ue_demand + hp_electric)
		opt_problem += shared_power[i] <= \
			pv_production[i] + bess_discharge[i] - bess_charge[i], \
			'Shared power linearisation (1/6) {}'.format(i)
		opt_problem += \
			shared_power[i] <= ue_demand[i], \
			'Shared power linearisation (2/6) {}'.format(i)
	
		# Definition of y
		# y = 1 if pv_production + bess_discharge - bess_charge <= 
		# ue_demand + hp_electric, 0 otherwise
		# The definition of y is introduced as a constraint
		opt_problem += M*y[i] >= (ue_demand[i]) \
			- (pv_production[i] + bess_discharge[i] - bess_charge[i]), \
			'Shared power linearisation (3/6) {}'.format(i)
		opt_problem += M * (1-y[i]) >= - (ue_demand[i]) \
			+ (pv_production[i] + bess_discharge[i] - bess_charge[i]), \
			'Shared power linearisation (4/6) {}'.format(i)
	
		# Constraint on the shared energy, that must be not only smaller than 
		# both injections and withdrawals but also equal to the minimum value.
		# When y == 1:
		#   shared_power = pv_production + bess_discharge - bess_charge
		# for this constraint and smaller-equal for the previous one. 
		# When y == 0, the other way around.
		opt_problem += shared_power[i] >= \
			(pv_production[i] + bess_discharge[i] - bess_charge[i]) \
			- M*(1-y[i]), \
			'Shared power linearisation (5/6) {}'.format(i)
		opt_problem += \
			shared_power[i] >= (ue_demand[i]) - M*y[i], \
			'Shared power linearisation (6/6) {}'.format(i)
			
	## Adding the objective of maximizing the shared energy
	opt_problem += \
		plp.lpSum([(shared_power[i]) * dt for i in time_set])
	
	### Solving the problem and post-processing
	
	## Solving the problem
	# For each time-step the variables are evaluated in order to reach the 
	# specified objective.
	# A try-except-else block is used, in case pulp fails at the optimisation
	# In this case, nans are returned. The absence of data due to the failed
	# optimisation is handled in another module.
	
	try:
		# tic()
		basepath = Path(__file__).parent
		solver_path = basepath / "Solvers" / "cbc.exe"  #Path to CBC solver   
		solver_path = solver_path.as_posix() #Windows path requires being casted to posix
		solver = plp.COIN_CMD(path = solver_path, msg=1)
		opt_problem.solve(solver) #PULP_CBC_CMD(msg=True)  
		# print('Time to solve the opt problem: {} s'.format(toc()))
		opt_status = plp.LpStatus[opt_problem.status]
	except:
		warnings.warn("Optimization failed")
		opt_status = 'Failed'
		grid_feed = np.empty((time_length,)) * np.nan
		grid_purchase = np.empty((time_length,)) * np.nan
		bess_charge = np.empty((time_length,)) * np.nan
		bess_discharge = np.empty((time_length,)) * np.nan
		bess_energy = np.empty((time_length,)) * np.nan
		injections =  np.empty((time_length,)) * np.nan
		withdrawals =  np.empty((time_length,)) * np.nan
		shared_power = np.empty((time_length,)) * np.nan
	
	## Post-processing of the results
	# Storing the optimised values of the variables to be returned
	for i in time_set:
		grid_feed[i] = plp.value(grid_feed[i])
		grid_purchase[i] = plp.value(grid_purchase[i])
		bess_charge[i] = plp.value(bess_charge[i])
		bess_discharge[i] = plp.value(bess_discharge[i])
		bess_energy[i] = plp.value(bess_energy[i])
		shared_power[i] = plp.value(shared_power[i])
		
	injections = \
		pv_production + np.asarray(bess_discharge) - \
		np.asarray(bess_charge)
	withdrawals =  ue_demand
	
	# Returning the lists as arrays    
	results = pd.DataFrame({
		# 'pv_production': pv_production,
		# 'ue_demand': ue_demand,
		# 'grid_purchase': np.asarray(grid_purchase),
		# 'grid_feed': np.asarray(grid_feed),
		# 'bess_charge': np.asarray(bess_charge),
		# 'bess_discharge': np.asarray(bess_discharge),
		# 'bess_energy': np.asarray(bess_energy),
		'injections': injections,
		'withdrawals': withdrawals,
		'shared_power': np.asarray(shared_power),
		 })    
	
	# print('Total time: {} s'.format(toc()))
	
	return opt_status, results


