# ___________                                ________  ________  
# \__    ___/  ____  _____     _____         \_____  \ \_____  \ 
#   |    |   _/ __ \ \__  \   /     \         /  ____/   _(__  < 
#   |    |   \  ___/  / __ \_|  Y Y  \       /       \  /       \
#   |____|    \___  >(____  /|__|_|  / ______\_______ \/______  /
#                 \/      \/       \/ /_____/        \/       \/
import pandas as pd
import numpy as np

from evaluation_quicky import configuration_evaluation #To compute value function
from systems_specifications import storage_system_specifications #Read battery specs
from plotter import shares_pie_plot

import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#Plot configuration
sns.set_theme() #use to reset955
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale= 2.5)
plt.rc('figure',figsize=(32,24))
plt.rc('font', size=20)
plt.ioff()


class DistributionModel:
	"""Benefit distribution model according to Repetto's basic formulation:
	No storage is considered."""

	#----------------------------Nested Participant class----------------------
	class _Participant:
		"""Constructor should not be invoked by user"""
		def __init__(self, participant_number, name, profile_we = None, profile_wd = None):
			"""Parameters:
				player_number: int. 
			"""
			self._profile_wd = profile_wd
			self._profile_we = profile_we

			#Assign player number
			self.participant_number = participant_number
			#Player name
			self.name = name
			#Display dumb menu
			print("Creating Participant {}: {}".format(participant_number, name))
			#Insert PV Power and Battery Size for this user.
			self._pv_size = input("Insert PV Power for this participant: ")
			if self._pv_size == '': 
				self._pv_size = 0
			else:
				self._pv_size = int(self._pv_size)
			
			#No battery in this base formulation

			#Compute player max power
			self._grid_purchase_max = np.ceil(profile_wd.max())

			#Consumer / Producer flags
			self.is_consumer = (self._grid_purchase_max != 0)
			self.is_producer = (self._pv_size > 0)

			#Compute hourly profiles (production / consumption)
			#np.array format
			#Call to Lorenti's Module - replace np.empty
			self.consumption = np.empty() 
			self.production = np.empty()


	#----------------------------------Private Methods---------------------------------
	
	def _config_inputs(self):
		"""Generate all configuration inputs.
		Returns:
		profile_wd: np.array of shape (24,12)
		profile_we: np.array of shape (24,12)
		pv_size: float
		"""
		profile_wd = np.zeros((24,12))
		profile_we = np.zeros((24,12))
		pv_size = 0
		grid_purchase_max = 0
		for participant in self.participants:
			profile_wd += participant._profile_wd
			profile_we += participant._profile_we
			pv_size += participant._pv_size
			grid_purchase_max += participant._grid_purchase_max

		return profile_wd, profile_we, pv_size, grid_purchase_max	


	def _run_recopt(self, ):
		"""Optimize configuration and return power fluxes"""
		#Compute inputs
		profile_wd, profile_we, pv_size, grid_purchase_max = self._config_inputs()
		# Optimization Setup
		# Size (kW, electric)
		pv = {'size': pv_size,}
		## Electricicty distribution grid (grid)
		# Maximum purchasable power (kW, electric)
		# Maximum feedable power (kW, electric) - Set equal to pv_size
		grid = {'purchase_max': grid_purchase_max,
				'feed_max': pv_size,
				}

		#Create technologies dictionary
		technologies_dict = {'pv': pv,
					 'grid': grid,                      
					}

		# Electric load (kWh/h) evaluated using the routine for work- and weekend-days
		ue_demand_months = np.zeros((self._time_length, self._n_months, self._n_day_types))
		for day_type, data in zip(self._day_types, [profile_wd, profile_we]):
			dd = self._day_types[day_type]['id']
			ue_demand_months[:, :, dd] = data

		# Load & Production dictionary
		profiles_months = {'ue_demand_months': ue_demand_months, #(24x12x2)
							'pv_production_unit_months': self._pv_production_unit_months,
							}	
		# Run Optimization

		#Modify this to return hourly series
		yearly_energy = configuration_evaluation(self._time_dict, 
													profiles_months,
													technologies_dict,
													self._auxiliary_dict
													)

		# Compute economic value (hourly)
		# Return it		


	def _economic_value(self, shared_energy, grid_feed, PR3 = 42,  CUAF = 8.56 , TP = 110):
		"""Return economic value of shared energy plus energy sales"""
		ritiro_energia = grid_feed/1000 * PR3 #KWh to MWh
		incentivo = shared_energy/1000 * (CUAF + TP)
		return ritiro_energia + incentivo

	def _compute_shares(self, participant):
		"""Compute % shares (suitable for both producers and consumers).
		Runs on hourly basis, returns yearly values"""
		p_production = participant.production
		p_consumption = participant.consumption

		#Phase 1: Producer share
		p_share = np.zeros(len(p_production)) #producer share for this participant 
		if participant.is_producer:
			for ix, single, total in enumerate(zip(p_production, self._total_production)):
				#Look before you leap
				if total == 0:
					continue #p_share remain zero
				p_share[ix] = single/total
		
		#Producer cash amount
		producer_cash = self.eta * np.multiply(self._total_value, p_share)

		#Phase 2: Consumer share
		c_share = np.zeros(len(p_production)) #consumer share for this participant 
		if participant.is_consumer:
			for ix, single, total in enumerate(zip(p_consumption, self._total_consumption)):
				#Look before you leap
				if total == 0:
					continue #p_share remain zero
				c_share[ix] = single/total			

		#Consumer cash amount
		consumer_cash = (1 - self.eta) * np.multiply(self._total_value, c_share)

		#Return total cash amount
		return sum(producer_cash + consumer_cash)

	def _create_participants(self):
		"""Create participants to REC.
		Returns:
		List of _Participant objects
		
		"""
		basepath= Path(__file__).parent
		datapath = basepath / 'Data'
		data_folders = os.listdir(datapath)
		participant_list = [] #Store _Participant objects
		for ix, folder in enumerate(data_folders):
			#Workday
			wd_path = datapath / folder / "consumption_profiles_month_wd.csv"
			wd_data = pd.read_csv(wd_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]
			#Weekend
			we_path = datapath / folder / "consumption_profiles_month_we.csv"
			we_data = pd.read_csv(we_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]

			#Instantiate new player
			new = self._Participant(ix, folder, wd_data, we_data) #folder stays for participant name
			participant_list.append(new)

		return player_list
	
	#----------------Public Methods------------------

	def __init__(self, eta = 0.5):
		"""Create an instance of the model.
		Parameters.
		eta: float in [0,1]. Share of Benefit for producers.
		"""
		self.eta = eta
		#Create participants
		self.participants = self._create_participants()
		self._n_participants = len(self.participants)

		################ RECOPT SETUP #############

		#Auxiliary variables
		# Seasons, storing for each season an id number and nickname
		self._seasons = {'winter': {'id': 0, 'nickname': 'w'},
						'spring': {'id': 1, 'nickname': 'ap'},
						'summer': {'id': 2, 'nickname': 's'},
						'autumn': {'id': 3, 'nickname': 'ap'},
						}
		# self._months, storing for each month an id number and nickname				
		self._months = {'january': {'id': 0, 'nickname': 'jan', 'season': 'winter'},
						'february': {'id': 1, 'nickname': 'feb', 'season': 'winter'},
						'march': {'id': 2, 'nickname': 'mar', 'season': 'winter'},
						'april': {'id': 3, 'nickname': 'apr', 'season': 'spring'},
						'may': {'id': 4, 'nickname': 'may', 'season': 'spring'},
						'june': {'id': 5, 'nickname': 'jun', 'season': 'spring'},
						'july': {'id': 6, 'nickname': 'jul', 'season': 'summer'},
						'august': {'id': 7, 'nickname': 'aug', 'season': 'summer'},
						'september': {'id': 8, 'nickname': 'sep', 'season': 'summer'},
						'october': {'id': 9, 'nickname': 'oct', 'season': 'autumn'},
						'november': {'id': 10, 'nickname': 'nov', 'season': 'autumn'},
						'december': {'id': 11, 'nickname': 'dec', 'season': 'autumn'},
						}
		# Day types, storing for each day type an id number and nickname
		self._day_types = {'work-day': {'id': 0, 'nickname': 'wd'},
							'weekend-day': {'id': 1, 'nickname': 'we'},
							}
		# Distribution of both day types among all self._months
		self._days_distr_months = {'january': {'work-day': 21, 'weekend-day': 10},
								'february': {'work-day': 20, 'weekend-day': 8},
								'march': {'work-day': 23, 'weekend-day': 8},
								'april': {'work-day': 22, 'weekend-day': 8},
								'may': {'work-day': 21, 'weekend-day': 10},
								'june': {'work-day': 22, 'weekend-day': 8},
								'july': {'work-day': 22, 'weekend-day': 9},
								'august': {'work-day': 22, 'weekend-day': 9},
								'september': {'work-day': 22, 'weekend-day': 8},
								'october': {'work-day': 21, 'weekend-day': 10},
								'november': {'work-day': 22, 'weekend-day': 8},
								'december': {'work-day': 23, 'weekend-day': 8},
								}
		# Number of seasons, months and day_types
		self._n_seasons, self._n_months, self._n_day_types = \
			len(self._seasons), len(self._months), len(self._day_types)

		# Distribution of both day types among all seasons
		self._days_distr_seasons = {}
		for month in self._months:
			season = self._months[month]['season']
			
			work_days_month = self._days_distr_months[month]['work-day']
			weekend_days_month = self._days_distr_months[month]['weekend-day']
			
			if season not in self._days_distr_seasons: 
				self._days_distr_seasons[season] = {'work-day': work_days_month,
											  'weekend-day':  weekend_days_month}
			else: 
				self._days_distr_seasons[season]['work-day'] += work_days_month
				self._days_distr_seasons[season]['weekend-day'] += weekend_days_month
				
		# Days distributions as arrays useful for quicker calculations
		self._days_distr_months_array = np.zeros((self._n_months, self._n_day_types))
		for month in self._months:
			mm = self._months[month]['id']
			for day_type in self._day_types:
				dd = self._day_types[day_type]['id']
				self._days_distr_months_array[mm, dd] = self._days_distr_months[month][day_type]
				
		self._days_distr_seasons_array = np.zeros((self._n_seasons, self._n_day_types))
		for season in self._seasons:
			ss = self._seasons[season]['id']
			for day_type in self._day_types:
				dd = self._day_types[day_type]['id']
				self._days_distr_seasons_array[ss, dd] = self._days_distr_seasons[season][day_type]

		# Auxiliary time dictionary
		self._auxiliary_dict = {'seasons': self._seasons,
							  'months': self._months,
							  'day_types': self._day_types,
							  'days_distr_seasons': self._days_distr_seasons,
							  'days_distr_months': self._days_distr_months,
							  'days_distr_months_array': self._days_distr_months_array,
				  			}

		# PV data
		filename = 'pv_production_unit.csv'
		self._pv_data = np.array(pd.read_csv(filename, sep=';'))
		self._pv_production_unit_months = self._pv_data[:, 1:]
		# Broadcasting the array in order to account for different day types
		self._pv_production_unit_months = self._pv_production_unit_months[:, :, np.newaxis]
		broadcaster = np.zeros((self._n_day_types,))
		self._pv_production_unit_months = self._pv_production_unit_months + broadcaster		

		# Battery specs (constant)
		# Maximum and minimum states of charge (SOC) (%)
		# Minimum time of charge/discharge (h)
		# Charge, discharge and self-discharge efficiencies (-)
		# Size is not present as it varies according to subconfiguration
		self._bess = storage_system_specifications(default_flag=False)['bess']

		# Time discretization
		# Time-step (h)
		self._dt = 1
		# Total time of simulation (h)
		self._time = 24
		# Vector of time (h)
		self._time_sim = np.arange(0, self._time, self._dt)
		self._time_length = self._time_sim.size
		self._time_dict = {'dt': self._dt,
					  'time': self._time,
					  'time_sim': self._time_sim,
					  }

		#Run recopt and get quantities
		self._run_recopt() 
		self._total_value = np.empty(8760) #Economic value on hourly basis
		#aggregate values over all participants (hourly arrays)
		self._total_production = np.sum([p.production for p in self.participants])
		self._total_consumption = np.sum([p.consumption for p in self.participants])

	def run(self):
		"""Run distribution model, return shares"""
		#Compute shares (cash)
		shares = [self._compute_shares(participant) for participant in self.participants]
		# Distribution of shares
		distribution = shares / sum(shares)
		# Plot results
		# Use Lorenti's module
		#Create input dataframe
		names = [participant.name for participant in self.participants] #participant names
		types = [] #producers/consumers/prosumers
		for participant in self.players:
			if participant._pv_size > 0:
				if participant._grid_purchase_max == 0:
					types.append("producer")
				else:
					types.append("prosumer")
			else:
				types.append("consumer")
		plot_input = pd.DataFrame({'player': names,
								  'share': distribution,
								  'type': types
							 	})
		figure = shares_pie_plot(plot_input) #Plot data 
		plt.show(figure)		

			


